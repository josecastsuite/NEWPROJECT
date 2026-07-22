"""Lightweight advancing-streamline flow animation for JoseCast Analyzer.

The animator uses the pre-computed 3-D Darcy velocity vector field from
`FillingResult` to generate streamlines from the inlet and render the portion
of each streamline that the metal front has reached by the current fill time.
This is much lighter than per-frame particle advection and resembles the
" advancing lines" display in SolidWorks Flow Simulation.
"""

from typing import Optional, Tuple

import numpy as np
import pyvista as pv
from PyQt6 import QtCore

from core.types import AnalysisResult


class FlowAnimator(QtCore.QObject):
    """Animate metal flow as advancing streamlines inside the 3-D viewer."""

    # Time step for one animation frame in real-time seconds.
    FRAME_DT = 0.02
    # Number of seed points on the inlet.  Fewer lines = faster, but a handful
    # is enough to show the flow pattern clearly.
    MAX_STREAMLINES = 120
    # Line width in pixels and color map for velocity magnitude.
    LINE_WIDTH = 4

    def __init__(self, viewer):
        super().__init__(parent=None)
        self._viewer = viewer
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_timer)

        self._result: Optional[AnalysisResult] = None
        self._streamlines: Optional[pv.PolyData] = None
        self._arrival_time: Optional[np.ndarray] = None
        self._max_time: float = 0.0

        self._is_running: bool = False
        self._current_time: float = 0.0
        self._speed_multiplier: float = 1.0
        self._show_surface: bool = False

        self._streamline_actor = None
        self._surface_actor = None

    def set_result(self, result: Optional[AnalysisResult]) -> None:
        """Attach a completed analysis result and build streamlines."""
        self.stop()
        self._clear_actors()
        self._result = result
        self._streamlines = None
        self._arrival_time = None

        if result is None or result.flow_result is None:
            return

        fr = result.flow_result
        velocity = fr.velocity
        fill_time = fr.fill_time
        if (
            velocity is None
            or fill_time is None
            or fill_time.size == 0
            or result.grid is None
            or result.grid.size == 0
        ):
            return

        self._velocity = np.asarray(velocity, dtype=np.float32)
        self._fill_time = np.asarray(fill_time, dtype=np.float64)
        self._origin = np.asarray(result.origin_mm, dtype=np.float64)
        self._dx = float(result.dx_mm)
        self._shape = tuple(int(s) for s in result.grid.shape)
        self._max_time = float(fr.fill_time_s or 0.0)

        # Build the static velocity-field image and seed streamlines from the inlet.
        image = self._build_velocity_image()
        if image is None:
            return

        source = self._build_source_points()
        if source is None or source.n_points == 0:
            return

        try:
            streamlines = self._compute_streamlines(image, source)
        except Exception:
            streamlines = None

        if streamlines is None or streamlines.n_points == 0:
            return

        self._streamlines = streamlines
        self._arrival_time = self._compute_arrival_time(streamlines)

        # Normalise arrival times so the animation finishes at the Darcy fill time.
        if self._arrival_time is not None and self._arrival_time.size > 0:
            t_max = float(np.nanmax(self._arrival_time))
            if t_max > 0 and self._max_time > 0:
                self._arrival_time *= self._max_time / t_max
            elif t_max > 0 and self._max_time <= 0:
                self._max_time = t_max
            self._streamlines['arrival_time'] = self._arrival_time

        self._current_time = 0.0
        self._update_scene()

    def _build_velocity_image(self) -> Optional[pv.ImageData]:
        """Create a PyVista ImageData with the Darcy velocity vector field."""
        nx, ny, nz = self._shape
        n_points = nx * ny * nz

        # Flatten in C order (x fastest, then y, then z) to match PyVista point order.
        vx = self._velocity[0].ravel(order='C')
        vy = self._velocity[1].ravel(order='C')
        vz = self._velocity[2].ravel(order='C')
        vel = np.column_stack([vx, vy, vz])
        vel = np.ascontiguousarray(vel, dtype=np.float64)

        # Mask velocity to zero outside metal so streamlines stay inside the flow domain.
        grid_flat = self._result.grid.ravel(order='C')
        metal = grid_flat > 0
        if not metal.any():
            return None
        vel[~metal] = 0.0

        magnitude = np.linalg.norm(vel, axis=1)

        image = pv.ImageData(dimensions=(nx, ny, nz))
        # Cell-centred grid: origin shifted by half a voxel.
        image.origin = tuple(self._origin + 0.5 * self._dx)
        image.spacing = (self._dx, self._dx, self._dx)
        image['velocity_m_s'] = vel
        image['velocity_magnitude'] = magnitude
        return image

    def _build_source_points(self) -> Optional[pv.PolyData]:
        """Seed points uniformly across the inlet cells."""
        metal = self._result.grid > 0
        finite_fill = np.isfinite(self._fill_time)
        inlet_mask = finite_fill & (self._fill_time <= 1e-9) & metal
        if not inlet_mask.any():
            finite_vals = np.where(finite_fill & metal, self._fill_time, np.inf)
            min_t = float(np.min(finite_vals))
            inlet_mask = finite_fill & (self._fill_time <= min_t + 1e-9) & metal

        inlet_idx = np.argwhere(inlet_mask)
        if inlet_idx.shape[0] == 0:
            return None

        # Down-sample the inlet cells to a fixed number of streamlines.
        n = inlet_idx.shape[0]
        if n > self.MAX_STREAMLINES:
            step = max(1, n // self.MAX_STREAMLINES)
            inlet_idx = inlet_idx[::step][: self.MAX_STREAMLINES]

        # Centre of each selected voxel.
        pts = inlet_idx * self._dx + self._origin + 0.5 * self._dx
        return pv.PolyData(pts)

    def _compute_streamlines(self, image: pv.ImageData, source: pv.PolyData) -> pv.PolyData:
        """Run the VTK stream tracer from the inlet seeds."""
        diagonal = float(np.linalg.norm(self._shape) * self._dx)
        sl = image.streamlines_from_source(
            source,
            vectors='velocity_m_s',
            integration_direction='forward',
            max_length=diagonal * 1.5,
            initial_step_length=self._dx * 0.5,
            min_step_length=self._dx * 0.05,
            max_step_length=self._dx,
        )
        return sl

    @staticmethod
    def _compute_arrival_time(sl: pv.PolyData) -> np.ndarray:
        """Integrate ds / v along each streamline to get physical arrival time."""
        pts = sl.points
        mag = sl['velocity_magnitude']
        lines = sl.lines

        arrival = np.zeros(sl.n_points, dtype=np.float64)
        if lines is None or len(lines) == 0:
            return arrival

        idx = 0
        while idx < len(lines):
            n = int(lines[idx])
            inds = lines[idx + 1 : idx + 1 + n]
            idx += 1 + n
            for k in range(1, n):
                i0 = inds[k - 1]
                i1 = inds[k]
                ds_mm = float(np.linalg.norm(pts[i1] - pts[i0]))
                v_avg = 0.5 * (mag[i0] + mag[i1])
                dt = (ds_mm * 0.001) / max(v_avg, 1e-9)
                arrival[i1] = arrival[i0] + dt
        return arrival

    def _visible_streamlines(self, t: float) -> Optional[pv.PolyData]:
        """Return the portion of each streamline that has arrived by time t."""
        if self._streamlines is None or self._arrival_time is None:
            return None

        if t >= self._max_time:
            # Full streamlines; just ensure the active scalar is velocity.
            vis = self._streamlines.copy()
            vis.set_active_scalars('velocity_magnitude')
            return vis

        pts = self._streamlines.points
        mag = self._streamlines['velocity_magnitude']
        lines = self._streamlines.lines
        arr = self._arrival_time

        out_pts = []
        out_mag = []
        out_lines = []
        cursor = 0

        idx = 0
        while idx < len(lines):
            n = int(lines[idx])
            inds = lines[idx + 1 : idx + 1 + n]
            idx += 1 + n

            # Number of points on this line whose arrival time is <= t.
            m = 0
            while m < n and arr[inds[m]] <= t:
                m += 1
            if m < 2:
                continue

            out_lines.append(m)
            out_lines.extend(range(cursor, cursor + m))
            out_pts.append(pts[inds[:m]])
            out_mag.append(mag[inds[:m]])
            cursor += m

        if not out_pts:
            return None

        vis = pv.PolyData()
        vis.points = np.concatenate(out_pts, axis=0)
        vis.lines = np.array(out_lines, dtype=np.int64)
        vis['velocity_magnitude'] = np.concatenate(out_mag)
        return vis

    def set_speed_multiplier(self, speed: float) -> None:
        """Scale the physical time step (1.0 = real time)."""
        self._speed_multiplier = max(0.01, float(speed))

    def set_show_surface(self, show: bool) -> None:
        """Toggle the reconstructed fluid surface (currently placeholder)."""
        self._show_surface = bool(show)

    def set_current_time(self, t: float) -> None:
        """Jump to a specific fill time and update the view."""
        self._current_time = float(np.clip(t, 0.0, self._max_time))
        self._update_scene()

    def play(self) -> None:
        """Start/pause the timer."""
        if self._streamlines is None:
            return
        if self._is_running:
            self._timer.stop()
            self._is_running = False
        else:
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
        self._current_time = 0.0

    def _clear_actors(self) -> None:
        if self._streamline_actor is not None:
            try:
                self._viewer.remove_actor(self._streamline_actor)
            except Exception:
                pass
            self._streamline_actor = None
        if self._surface_actor is not None:
            try:
                self._viewer.remove_actor(self._surface_actor)
            except Exception:
                pass
            self._surface_actor = None

    def _on_timer(self) -> None:
        """Advance one animation frame."""
        if self._streamlines is None:
            return
        dt = self.FRAME_DT * self._speed_multiplier
        self._current_time = min(self._current_time + dt, self._max_time)
        self._update_scene()
        if self._current_time >= self._max_time:
            self.pause()

    def _update_scene(self) -> None:
        """Render the currently visible part of the streamlines."""
        vis = self._visible_streamlines(self._current_time)
        if vis is None or vis.n_points == 0:
            self._clear_actors()
            return

        try:
            if self._streamline_actor is None:
                self._streamline_actor = self._viewer.add_mesh(
                    vis,
                    scalars='velocity_magnitude',
                    cmap='turbo',
                    line_width=self.LINE_WIDTH,
                    render_lines_as_tubes=True,
                    opacity=0.95,
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
                    name="flow_animation_streamlines",
                )
            else:
                self._streamline_actor.mapper.dataset = vis
        except Exception:
            if self._streamline_actor is not None:
                try:
                    self._viewer.remove_actor(self._streamline_actor)
                except Exception:
                    pass
                self._streamline_actor = None
            self._streamline_actor = self._viewer.add_mesh(
                vis,
                scalars='velocity_magnitude',
                cmap='turbo',
                line_width=self.LINE_WIDTH,
                render_lines_as_tubes=True,
                opacity=0.95,
                show_scalar_bar=False,
                name="flow_animation_streamlines",
            )

        self._viewer.render()

    def particle_count(self) -> int:
        """Return the number of streamlines (API compatibility)."""
        if self._streamlines is None:
            return 0
        return int(self._streamlines.n_cells)

    def particle_diameter_mm(self) -> float:
        """Return 0.0 for streamlines (API compatibility)."""
        return 0.0
