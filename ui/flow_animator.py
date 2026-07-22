"""Real-time particle flow animation for JoseCast Analyzer.

The animator uses the 3-D Darcy velocity vector from `FillingResult` to emit,
advect and render particles that mimic a stream of liquid metal through the
gating system.  It is fully integrated with the PyVistaQt `Analyzer3DViewer`
and driven by a `QTimer` so play/pause/speed controls can be wired to the
Qt UI.
"""

from typing import List, Optional, Tuple

import numpy as np
import pyvista as pv
from PyQt6 import QtCore
from scipy import ndimage

from core.types import AnalysisResult


try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False
    njit = prange = None  # type: ignore


def _make_sample_velocity_numba():
    @njit(parallel=True, cache=True, fastmath=True)
    def sample_velocity_numba(pos, origin, inv_dx, velocity, out):
        n = pos.shape[0]
        nx = velocity.shape[1]
        ny = velocity.shape[2]
        nz = velocity.shape[3]
        for idx in prange(n):
            x = (pos[idx, 0] - origin[0]) * inv_dx
            y = (pos[idx, 1] - origin[1]) * inv_dx
            z = (pos[idx, 2] - origin[2]) * inv_dx
            if x < 0.0:
                x = 0.0
            if y < 0.0:
                y = 0.0
            if z < 0.0:
                z = 0.0
            if x > nx - 1:
                x = float(nx - 1)
            if y > ny - 1:
                y = float(ny - 1)
            if z > nz - 1:
                z = float(nz - 1)
            i0 = int(np.floor(x))
            j0 = int(np.floor(y))
            k0 = int(np.floor(z))
            i1 = i0 + 1
            j1 = j0 + 1
            k1 = k0 + 1
            if i1 >= nx:
                i1 = nx - 1
            if j1 >= ny:
                j1 = ny - 1
            if k1 >= nz:
                k1 = nz - 1
            tx = x - i0
            ty = y - j0
            tz = z - k0
            for comp in range(3):
                v000 = velocity[comp, i0, j0, k0]
                v001 = velocity[comp, i0, j0, k1]
                v010 = velocity[comp, i0, j1, k0]
                v011 = velocity[comp, i0, j1, k1]
                v100 = velocity[comp, i1, j0, k0]
                v101 = velocity[comp, i1, j0, k1]
                v110 = velocity[comp, i1, j1, k0]
                v111 = velocity[comp, i1, j1, k1]
                c00 = v000 * (1.0 - tz) + v001 * tz
                c01 = v010 * (1.0 - tz) + v011 * tz
                c10 = v100 * (1.0 - tz) + v101 * tz
                c11 = v110 * (1.0 - tz) + v111 * tz
                c0 = c00 * (1.0 - ty) + c01 * ty
                c1 = c10 * (1.0 - ty) + c11 * ty
                out[idx, comp] = c0 * (1.0 - tx) + c1 * tx
    return sample_velocity_numba


def _make_is_metal_numba():
    @njit(parallel=True, cache=True)
    def is_metal_numba(pos, origin, inv_dx, grid, out):
        n = pos.shape[0]
        nx = grid.shape[0]
        ny = grid.shape[1]
        nz = grid.shape[2]
        for idx in prange(n):
            x = (pos[idx, 0] - origin[0]) * inv_dx
            y = (pos[idx, 1] - origin[1]) * inv_dx
            z = (pos[idx, 2] - origin[2]) * inv_dx
            i = int(round(x))
            j = int(round(y))
            k = int(round(z))
            if i < 0 or i >= nx or j < 0 or j >= ny or k < 0 or k >= nz:
                out[idx] = False
            else:
                out[idx] = grid[i, j, k] > 0
    return is_metal_numba


if _HAVE_NUMBA:
    _sample_velocity_numba = _make_sample_velocity_numba()
    _is_metal_numba = _make_is_metal_numba()
else:
    _sample_velocity_numba = None
    _is_metal_numba = None


class FlowAnimator(QtCore.QObject):
    """Animate metal flow as a particle stream inside the 3-D viewer."""

    # Time step for one animation frame in real-time seconds.  Smaller values
    # give smoother advection; the effective physical dt per frame is
    # FRAME_DT * speed_multiplier.
    FRAME_DT = 0.02
    # Maximum number of live particles.  Large enough to look like a continuous
    # stream on a mid-range GPU (e.g. GTX 1050 Ti 4 GB) while staying interactive.
    MAX_PARTICLE_BUDGET = 150_000
    # Minimum particle diameter in voxels.  2 voxels ensures neighbouring
    # particles overlap and the stream looks solid rather than sparse.
    MIN_PARTICLE_DIAMETER_VOXELS = 2.0
    # Hard lower limit on particle size (mm) so very fine grids do not explode.
    MIN_PARTICLE_DIAMETER_MM = 0.25
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
        self._grid: Optional[np.ndarray] = None
        self._inlet_positions: Optional[np.ndarray] = None
        self._origin: Optional[np.ndarray] = None
        self._dx: float = 1.0
        self._shape: Optional[Tuple[int, int, int]] = None
        self._max_time: float = 0.0

        self._is_running: bool = False
        self._current_time: float = 0.0
        self._speed_multiplier: float = 1.0

        # Particle budget and sizing are derived automatically from the metal
        # volume.  The user never sets these directly.
        self._metal_volume_mm3: float = 0.0
        self._particle_diameter_mm: float = 1.0
        self._particle_volume_mm3: float = 1.0
        self._target_particle_count: int = 0
        self._emit_rate: float = 0.0
        self._emit_accumulator: float = 0.0
        self._point_size: int = 4
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
            self._grid = None
            self._inlet_positions = None
            return

        fr = result.flow_result
        velocity = fr.velocity
        fill_time = fr.fill_time
        if velocity is None or fill_time is None or fill_time.size == 0:
            self._velocity = None
            self._fill_time = None
            self._grid = None
            self._inlet_positions = None
            return

        self._velocity = np.ascontiguousarray(velocity, dtype=np.float32)
        self._fill_time = np.ascontiguousarray(fill_time, dtype=np.float64)
        self._grid = np.ascontiguousarray(result.grid, dtype=np.int16)
        self._origin = np.ascontiguousarray(result.origin_mm, dtype=np.float64)
        self._dx = float(result.dx_mm)
        self._shape = tuple(int(s) for s in result.grid.shape)

        metal = result.grid > 0
        finite_fill = np.isfinite(self._fill_time)
        self._max_time = float(
            np.nanmax(self._fill_time[metal & finite_fill])
            if (metal & finite_fill).any()
            else 0.0
        )

        # Total metal volume (mm^3) drives the particle budget and size.
        self._metal_volume_mm3 = float(metal.sum()) * (self._dx ** 3)
        self._compute_particle_budget()

        # Inlet = cells whose fill_time is essentially zero (the seeded source).
        inlet_mask = finite_fill & (self._fill_time <= 1e-9) & metal
        if not inlet_mask.any():
            finite_vals = np.where(finite_fill & metal, self._fill_time, np.inf)
            min_t = float(np.min(finite_vals))
            inlet_mask = finite_fill & (self._fill_time <= min_t + 1e-9) & metal

        self._inlet_positions = np.argwhere(inlet_mask)
        if self._inlet_positions.shape[0] == 0:
            self._velocity = None
            self._fill_time = None
            self._grid = None
            return

        # Warm up the numba kernels on a tiny batch so the first real frame
        # is not delayed by compilation.
        if _HAVE_NUMBA:
            try:
                _is_metal_numba(
                    self._origin.reshape(1, 3),
                    self._origin,
                    1.0 / self._dx,
                    self._grid,
                    np.empty(1, dtype=bool),
                )
                _sample_velocity_numba(
                    self._origin.reshape(1, 3),
                    self._origin,
                    1.0 / self._dx,
                    self._velocity,
                    np.empty((1, 3), dtype=np.float64),
                )
            except Exception:
                pass

        self._current_time = 0.0
        self._reset_particles()

    def _compute_particle_budget(self) -> None:
        """Choose particle size/count from the metal volume.

        We want enough particles to make the stream look like a continuous
        volume, but few enough to keep 15-30 FPS on the target GPU.  The
        particle diameter is grown until the budget is respected.
        """
        dx = self._dx
        d_min = max(
            self.MIN_PARTICLE_DIAMETER_MM,
            self.MIN_PARTICLE_DIAMETER_VOXELS * dx,
        )
        v_particle_min = (np.pi / 6.0) * (d_min ** 3)

        if v_particle_min > 0 and self._metal_volume_mm3 > 0:
            n_min = int(np.ceil(self._metal_volume_mm3 / v_particle_min))
        else:
            n_min = 0

        if n_min <= self.MAX_PARTICLE_BUDGET:
            self._particle_diameter_mm = d_min
            self._particle_volume_mm3 = v_particle_min
            self._target_particle_count = max(1000, n_min)
        else:
            d = (
                6.0
                * self._metal_volume_mm3
                / (np.pi * self.MAX_PARTICLE_BUDGET)
            ) ** (1.0 / 3.0)
            self._particle_diameter_mm = max(d, d_min)
            self._particle_volume_mm3 = (np.pi / 6.0) * (d ** 3)
            self._target_particle_count = self.MAX_PARTICLE_BUDGET

        # Pixel point size is a rough visual multiplier.  Larger particles get
        # larger points so they overlap and look like a fluid surface.
        ratio = self._particle_diameter_mm / max(dx, 0.01)
        self._point_size = int(np.clip(ratio, 4, 10))

        # Emission rate (particles per second) follows the volume flow rate Q.
        # If Q is not available, fall back to a uniform emission over fill time.
        fr = self._result.flow_result if self._result else None
        q_mm3_s = 0.0
        if fr and fr.Q_m3_s and fr.Q_m3_s > 0:
            q_mm3_s = float(fr.Q_m3_s) * 1e9
        if q_mm3_s > 0 and self._particle_volume_mm3 > 0:
            self._emit_rate = q_mm3_s / self._particle_volume_mm3
        elif self._max_time > 0:
            self._emit_rate = self._target_particle_count / self._max_time
        else:
            self._emit_rate = 1000.0

        # Do not emit more than the target count in a single frame.
        self._emit_rate = min(
            self._emit_rate,
            self._target_particle_count / self.FRAME_DT,
        )

    def _reset_particles(self) -> None:
        """Drop all live particles and start from a clean state."""
        self._particle_pos = np.empty((0, 3), dtype=np.float64)
        self._particle_vel = np.empty((0, 3), dtype=np.float64)
        self._particle_age = np.empty(0, dtype=np.float64)
        self._particle_active = np.empty(0, dtype=bool)
        self._emit_accumulator = 0.0

    def set_particle_count(self, count: int) -> None:
        """Legacy/manual override retained for API compatibility.

        The preferred path is the volume-driven budget set in set_result().
        """
        count = max(1000, int(count))
        self._target_particle_count = min(count, self.MAX_PARTICLE_BUDGET)
        if self._particle_pos.shape[0] > self._target_particle_count:
            keep = self._target_particle_count
            self._particle_pos = self._particle_pos[-keep:]
            self._particle_vel = self._particle_vel[-keep:]
            self._particle_age = self._particle_age[-keep:]
            self._particle_active = self._particle_active[-keep:]

    def particle_count(self) -> int:
        """Return the current auto-computed target particle count."""
        return self._target_particle_count

    def particle_diameter_mm(self) -> float:
        """Return the current auto-computed particle diameter."""
        return self._particle_diameter_mm

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
        """Spawn new particles across the full inlet cross-section.

        Particles are placed uniformly inside each inlet voxel, not just at
        voxel centres, so the stream fills the sprue/runner throat from the
        very first frame and looks like a continuous liquid jet.
        """
        if self._inlet_positions is None or self._inlet_positions.shape[0] == 0:
            return

        self._emit_accumulator += self._emit_rate * dt
        n_emit = int(self._emit_accumulator)
        if n_emit <= 0:
            return
        self._emit_accumulator -= n_emit

        # Distribute the budget across all inlet voxels.
        idx = np.random.randint(
            0, self._inlet_positions.shape[0], size=n_emit
        )
        cells = self._inlet_positions[idx]
        # Uniform random position anywhere inside the selected voxel.
        jitter = np.random.rand(n_emit, 3)
        new_pos = (cells + jitter) * self._dx + self._origin

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
        self._trim_particle_pool()

    def _trim_particle_pool(self) -> None:
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

        # Advance active particles; v is m/s, dt is s, *1000 -> mm.
        active = self._particle_active.copy()
        self._particle_pos[active] += vel[active] * dt * 1000.0
        self._particle_vel[active] = vel[active]
        self._particle_age += dt

        # Kill particles that hit a wall, left the domain or stopped moving.
        inside = self._inside_metal(self._particle_pos)
        max_age = max(self._max_time * self.MAX_AGE_FACTOR, 0.1)
        self._particle_active &= (
            active
            & inside
            & (speed > 1e-6)
            & (self._particle_age < max_age)
        )

        if not self._particle_active.all():
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

    def _inside_metal(self, pos: np.ndarray) -> np.ndarray:
        """Return mask of points that lie inside a metal voxel."""
        if self._grid is None or pos.shape[0] == 0:
            return np.zeros(pos.shape[0], dtype=bool)
        if _HAVE_NUMBA:
            out = np.empty(pos.shape[0], dtype=bool)
            _is_metal_numba(
                pos, self._origin, 1.0 / self._dx, self._grid, out
            )
            return out
        # Fallback: scipy ndimage nearest-neighbour sampling.
        ijk = (pos - self._origin) / self._dx
        coords = np.stack([ijk[:, 0], ijk[:, 1], ijk[:, 2]], axis=0)
        sampled = ndimage.map_coordinates(
            self._grid.astype(np.float32),
            coords,
            order=0,
            mode="nearest",
            cval=0.0,
        )
        return sampled > 0

    def _sample_velocity(self, pos: np.ndarray) -> np.ndarray:
        """Trilinear interpolation of the velocity vector at world points (mm)."""
        if self._velocity is None or pos.shape[0] == 0:
            return np.empty_like(pos)
        if _HAVE_NUMBA:
            out = np.empty((pos.shape[0], 3), dtype=np.float64)
            _sample_velocity_numba(
                pos, self._origin, 1.0 / self._dx, self._velocity, out
            )
            return out

        # Fallback: scipy ndimage trilinear sampling.
        ijk = (pos - self._origin) / self._dx
        i = ijk[:, 0].astype(np.float64)
        j = ijk[:, 1].astype(np.float64)
        k = ijk[:, 2].astype(np.float64)
        coords = np.stack([np.zeros_like(i), i, j, k], axis=0)
        sampled = np.empty((pos.shape[0], 3), dtype=np.float64)
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

    def _update_scene(self) -> None:
        """Render the particle cloud."""
        if self._particle_pos.shape[0] == 0:
            self._clear_actors()
            return

        active = self._particle_active
        if not active.any():
            self._clear_actors()
            return

        pts = self._particle_pos[active]
        vel = self._particle_vel[active]
        speed = np.linalg.norm(vel, axis=1)

        poly = pv.PolyData(pts)
        poly["velocity_m_s"] = speed

        try:
            if self._particle_actor is None:
                self._particle_actor = self._viewer.add_mesh(
                    poly,
                    scalars="velocity_m_s",
                    cmap="turbo",
                    point_size=self._point_size,
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
                point_size=self._point_size,
                render_points_as_spheres=True,
                opacity=0.9,
                show_scalar_bar=False,
                name="flow_animation_particles",
            )

        self._viewer.render()
