"""Real-time particle flow animation for JoseCast Analyzer.

The animator uses the 3-D Darcy velocity vector and fill-time field from
`FillingResult` to emit, advect and render particles that mimic a stream of
liquid metal through the gating system.  It is fully integrated with the
PyVistaQt `Analyzer3DViewer` and driven by a `QTimer` so play/pause/speed
controls can be wired to the Qt UI.
"""

from typing import List, Optional, Tuple

import numpy as np
import pyvista as pv
from PyQt6 import QtCore
from scipy import ndimage

from core.types import AnalysisResult


class FlowAnimator(QtCore.QObject):
    """Animate metal flow as a particle stream inside the 3-D viewer."""

    # Time step for one animation frame in real-time seconds.  Smaller values
    # give smoother advection; the effective physical dt per frame is
    # FRAME_DT * speed_multiplier.
    FRAME_DT = 0.02
    # Default number of particles kept alive in the scene.
    DEFAULT_PARTICLE_COUNT = 6000
    # Maximum particle age as a fraction of total fill time.
    MAX_AGE_FACTOR = 1.5

    def __init__(self, viewer):
        super().__init__(parent=viewer)
        self._viewer = viewer
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_timer)

        self._result: Optional[AnalysisResult] = None
        self._velocity: Optional[np.ndarray] = None
        self._fill_time: Optional[np.ndarray] = None
        self._valid_mask: Optional[np.ndarray] = None
        self._inlet_positions: Optional[np.ndarray] = None
        self._origin: Optional[np.ndarray] = None
        self._dx: float = 1.0
        self._shape: Optional[Tuple[int, int, int]] = None
        self._max_time: float = 0.0

        self._is_running: bool = False
        self._current_time: float = 0.0
        self._speed_multiplier: float = 1.0
        self._target_particle_count: int = self.DEFAULT_PARTICLE_COUNT
        self._emit_rate: float = 0.0
        self._emit_accumulator: float = 0.0
        self._show_surface: bool = False

        self._particle_pos: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self._particle_vel: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self._particle_age: np.ndarray = np.empty(0, dtype=np.float64)
        self._particle_active: np.ndarray = np.empty(0, dtype=bool)

        self._particle_actor = None
        self._surface_actor = None

    def set_result(self, result: Optional[AnalysisResult]) -> None:
        """Attach a completed analysis result and pre-compute flow data."""
        self.stop()
        self._clear_actors()
        self._result = result
        if result is None or result.flow_result is None:
            self._velocity = None
            self._fill_time = None
            return

        fr = result.flow_result
        velocity = fr.velocity
        fill_time = fr.fill_time
        if velocity is None or fill_time is None or fill_time.size == 0:
            self._velocity = None
            self._fill_time = None
            return

        self._velocity = np.asarray(velocity, dtype=np.float32)
        self._fill_time = np.asarray(fill_time, dtype=np.float64)
        self._origin = np.asarray(result.origin_mm, dtype=np.float64)
        self._dx = float(result.dx_mm)
        self._shape = tuple(int(s) for s in result.grid.shape)

        metal = result.grid > 0
        finite_fill = np.isfinite(self._fill_time)
        self._valid_mask = metal & finite_fill
        self._max_time = float(
            np.nanmax(self._fill_time[self._valid_mask])
            if self._valid_mask.any()
            else 0.0
        )

        # Inlet = cells whose fill_time is essentially zero (the seeded source).
        inlet_mask = finite_fill & (self._fill_time <= 1e-9) & metal
        if not inlet_mask.any():
            # Fallback: take the cell(s) with the smallest finite fill time.
            finite_vals = np.where(finite_fill & metal, self._fill_time, np.inf)
            min_t = float(np.min(finite_vals))
            inlet_mask = finite_fill & (self._fill_time <= min_t + 1e-9) & metal

        inlet_idx = np.argwhere(inlet_mask)
        self._inlet_positions = inlet_idx * self._dx + self._origin + 0.5 * self._dx

        # Emit enough particles to keep the target count alive over the fill.
        self._emit_rate = (
            self._target_particle_count / max(self._max_time, 1e-6)
            if self._max_time > 0
            else 1000.0
        )
        self._emit_accumulator = 0.0

        # If there are no inlet cells, animation cannot run.
        if self._inlet_positions.shape[0] == 0:
            self._velocity = None
            self._fill_time = None
            return

        self._current_time = 0.0
        self._reset_particles()

    def _reset_particles(self) -> None:
        """Drop all live particles and start from a clean state."""
        self._particle_pos = np.empty((0, 3), dtype=np.float64)
        self._particle_vel = np.empty((0, 3), dtype=np.float64)
        self._particle_age = np.empty(0, dtype=np.float64)
        self._particle_active = np.empty(0, dtype=bool)
        self._emit_accumulator = 0.0
        self._update_particle_count_target()

    def set_particle_count(self, count: int) -> None:
        """Set the number of particles to keep in the scene."""
        count = max(100, int(count))
        self._target_particle_count = count
        if self._max_time > 0:
            self._emit_rate = count / max(self._max_time, 1e-6)
        if self._particle_pos.shape[0] > count:
            # Trim oldest particles.
            keep = count
            self._particle_pos = self._particle_pos[-keep:]
            self._particle_vel = self._particle_vel[-keep:]
            self._particle_age = self._particle_age[-keep:]
            self._particle_active = self._particle_active[-keep:]

    def set_speed_multiplier(self, speed: float) -> None:
        """Scale the physical time step (1.0 = real time)."""
        self._speed_multiplier = max(0.01, float(speed))

    def set_show_surface(self, show: bool) -> None:
        """Toggle the reconstructed fluid surface (currently placeholder)."""
        self._show_surface = bool(show)

    def set_current_time(self, t: float) -> None:
        """Jump to a specific fill time and update the view."""
        self._current_time = float(np.clip(t, 0.0, self._max_time))
        if not self._is_running:
            self._update_scene()

    def play(self) -> None:
        """Start/pause the timer."""
        if self._velocity is None:
            return
        if self._is_running:
            self._timer.stop()
            self._is_running = False
        else:
            # One animation step per timer tick; speed multiplier changes dt.
            self._timer.start(int(self.FRAME_DT * 1000))
            self._is_running = True

    def pause(self) -> None:
        """Pause the animation."""
        if self._is_running:
            self._timer.stop()
            self._is_running = False

    def stop(self) -> None:
        """Stop and remove all flow actors."""
        self.pause()
        self._clear_actors()
        self._reset_particles()

    def _clear_actors(self) -> None:
        if self._particle_actor is not None:
            try:
                self._viewer.remove_actor(self._particle_actor)
            except Exception:
                pass
            self._particle_actor = None
        if self._surface_actor is not None:
            try:
                self._viewer.remove_actor(self._surface_actor)
            except Exception:
                pass
            self._surface_actor = None

    def _on_timer(self) -> None:
        """Advance one animation frame."""
        if self._velocity is None:
            return
        dt = self.FRAME_DT * self._speed_multiplier
        self._current_time = min(self._current_time + dt, self._max_time)
        self._emit_particles(dt)
        self._advect_particles(dt)
        self._update_scene()
        if self._current_time >= self._max_time:
            self.pause()

    def _emit_particles(self, dt: float) -> None:
        """Spawn new particles at the inlet proportional to dt."""
        if self._inlet_positions is None or self._inlet_positions.shape[0] == 0:
            return
        self._emit_accumulator += self._emit_rate * dt
        n_emit = int(self._emit_accumulator)
        if n_emit <= 0:
            return
        self._emit_accumulator -= n_emit

        # Randomly pick inlet cells and add a sub-voxel jitter.
        idx = np.random.randint(
            0, self._inlet_positions.shape[0], size=n_emit
        )
        jitter = (np.random.rand(n_emit, 3) - 0.5) * self._dx * 0.6
        new_pos = self._inlet_positions[idx] + jitter

        # Initial velocity from the vector field at the inlet.
        new_vel = self._sample_velocity(new_pos)
        new_age = np.zeros(n_emit, dtype=np.float64)
        new_active = np.ones(n_emit, dtype=bool)

        self._particle_pos = np.vstack([self._particle_pos, new_pos])
        self._particle_vel = np.vstack([self._particle_vel, new_vel])
        self._particle_age = np.concatenate([self._particle_age, new_age])
        self._particle_active = np.concatenate(
            [self._particle_active, new_active]
        )
        self._update_particle_count_target()

    def _update_particle_count_target(self) -> None:
        """Keep particle pool near the target by discarding oldest."""
        n = self._particle_pos.shape[0]
        if n > self._target_particle_count:
            excess = n - self._target_particle_count
            self._particle_pos = self._particle_pos[excess:]
            self._particle_vel = self._particle_vel[excess:]
            self._particle_age = self._particle_age[excess:]
            self._particle_active = self._particle_active[excess:]

    def _advect_particles(self, dt: float) -> None:
        """Move particles with the local Darcy velocity vector."""
        if self._particle_pos.shape[0] == 0:
            return

        vel = self._sample_velocity(self._particle_pos)
        speed = np.linalg.norm(vel, axis=1)

        # Kill particles that are nearly stagnant or have left the domain.
        old_active = self._particle_active.copy()
        self._particle_active &= (
            (speed > 1e-6)
            & self._inside_domain(self._particle_pos)
        )

        # Apply velocity to still-active particles.
        active = self._particle_active
        # pos update: v [m/s] * dt [s] -> m, then *1000 -> mm.
        self._particle_pos[active] += vel[active] * dt * 1000.0
        self._particle_vel[active] = vel[active]
        self._particle_age += dt

        # Also kill particles that have exceeded a reasonable lifetime.
        max_age = max(self._max_time * self.MAX_AGE_FACTOR, 0.1)
        self._particle_active &= self._particle_age < max_age

        # If a particle was active and became inactive, it has exited or stopped;
        # remove it so it does not accumulate.
        if not np.array_equal(old_active, self._particle_active):
            self._compact_particles()

    def _compact_particles(self) -> None:
        """Remove dead particles from the arrays."""
        active = self._particle_active
        if active.all():
            return
        self._particle_pos = self._particle_pos[active]
        self._particle_vel = self._particle_vel[active]
        self._particle_age = self._particle_age[active]
        self._particle_active = self._particle_active[active]

    def _inside_domain(self, pos: np.ndarray) -> np.ndarray:
        """Return mask of points inside the voxel bounding box."""
        ijk = (pos - self._origin) / self._dx
        return (
            (ijk >= 0.0).all(axis=1)
            & (ijk[:, 0] < self._shape[0])
            & (ijk[:, 1] < self._shape[1])
            & (ijk[:, 2] < self._shape[2])
        )

    def _sample_velocity(self, pos: np.ndarray) -> np.ndarray:
        """Trilinear interpolation of the velocity vector at world points (mm)."""
        if self._velocity is None or pos.shape[0] == 0:
            return np.empty_like(pos)
        ijk = (pos - self._origin) / self._dx
        # velocity has shape (3, nx, ny, nz); map_coordinates needs [comp, x, y, z].
        i = ijk[:, 0].astype(np.float64)
        j = ijk[:, 1].astype(np.float64)
        k = ijk[:, 2].astype(np.float64)
        coords = np.stack([np.zeros_like(i), i, j, k], axis=0)
        sampled = np.empty_like(pos, dtype=np.float64)
        for comp in range(3):
            coords[0, :] = comp
            sampled[:, comp] = ndimage.map_coordinates(
                self._velocity,
                coords,
                order=1,
                mode="nearest",
                cval=0.0,
            )
        return sampled

    def _sample_fill_time(self, pos: np.ndarray) -> np.ndarray:
        """Trilinear interpolation of fill_time at world points (mm)."""
        if self._fill_time is None or pos.shape[0] == 0:
            return np.full(pos.shape[0], np.inf, dtype=np.float64)
        ijk = (pos - self._origin) / self._dx
        # fill_time has shape (nx, ny, nz); map_coordinates needs [x, y, z].
        coords = np.stack(
            [ijk[:, 0], ijk[:, 1], ijk[:, 2]], axis=0
        )
        return ndimage.map_coordinates(
            self._fill_time,
            coords,
            order=1,
            mode="nearest",
            cval=np.inf,
        )

    def _update_scene(self) -> None:
        """Render the particle cloud respecting the current fill front."""
        if self._particle_pos.shape[0] == 0:
            self._clear_actors()
            return

        # A particle is visible only where the metal front has already passed.
        fill_t = self._sample_fill_time(self._particle_pos)
        visible = (
            self._particle_active
            & np.isfinite(fill_t)
            & (fill_t <= self._current_time + 1e-6)
        )
        if not visible.any():
            self._clear_actors()
            return

        pts = self._particle_pos[visible]
        vel = self._particle_vel[visible]
        speed = np.linalg.norm(vel, axis=1)

        poly = pv.PolyData(pts)
        poly["velocity_m_s"] = speed

        try:
            if self._particle_actor is None:
                self._particle_actor = self._viewer.add_mesh(
                    poly,
                    scalars="velocity_m_s",
                    cmap="turbo",
                    point_size=4,
                    render_points_as_spheres=True,
                    opacity=0.9,
                    show_scalar_bar=True,
                    scalar_bar_args={
                        "color": "#00ffff",
                        "title_font_size": 10,
                        "label_font_size": 9,
                        "fmt": "%.2f",
                        "vertical": False,
                        "position_x": 0.02,
                        "position_y": 0.02,
                        "width": 0.12,
                        "height": 0.06,
                        "title": "Akış hızı (m/s)",
                    },
                    name="flow_animation_particles",
                )
            else:
                self._particle_actor.mapper.dataset = poly
        except Exception:
            # Fallback: rebuild the actor if the mapper update fails.
            if self._particle_actor is not None:
                try:
                    self._viewer.remove_actor(self._particle_actor)
                except Exception:
                    pass
            self._particle_actor = self._viewer.add_mesh(
                poly,
                scalars="velocity_m_s",
                cmap="turbo",
                point_size=4,
                render_points_as_spheres=True,
                opacity=0.9,
                show_scalar_bar=False,
                name="flow_animation_particles",
            )

        self._viewer.render()
