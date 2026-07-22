"""Flow-animation engine for JoséCast Analyzer.

Renders the metal fill front as an animated isosurface of ``fill_time`` and,
optionally, overlaid streamlines/pathlines.  The fill front is the commercial
standard (used by ProCAST, MAGMASOFT, FLOW-3D CAST, Altair Inspire Cast) and
is always consistent with the Darcy solver's ``velocity`` / ``fill_time`` field.
"""

from typing import List, Optional

import numpy as np
import pyvista as pv
from PyQt6 import QtCore
from scipy import ndimage

from core.types import AnalysisResult


class FlowAnimator(QtCore.QObject):
    """Animate metal flow as a 3-D fill front (isosurface of fill_time)."""

    TIMER_INTERVAL = 0.10
    FRAME_DT = 0.02
    MAX_STREAMLINES = 120
    MAX_STEPS = 2000
    CFL_FRACTION = 0.5
    LINE_WIDTH = 3
    FRONT_OPACITY = 0.82

    def __init__(self, viewer):
        super().__init__(parent=None)
        self._viewer = viewer
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_timer)

        self._result: Optional[AnalysisResult] = None
        self._flow_grid: Optional[pv.ImageData] = None
        self._streamlines: Optional[pv.PolyData] = None
        self._max_time: float = 0.0

        self._is_running: bool = False
        self._current_time: float = 0.0
        self._speed_multiplier: float = 1.0
        self._show_front: bool = True
        self._show_streamlines: bool = True

        self._front_actor = None
        self._streamline_actor = None
        self._current_front = None
        self._current_vis = None

    def set_result(self, result: Optional[AnalysisResult]) -> None:
        """Attach a completed analysis result and build the fill front."""
        self.stop()
        self._clear_actors()
        self._result = result
        self._flow_grid = None
        self._streamlines = None

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

        self._velocity = np.asarray(velocity, dtype=np.float64)
        self._fill_time = np.asarray(fill_time, dtype=np.float64)
        self._origin = np.asarray(result.origin_mm, dtype=np.float64)
        self._dx = float(result.dx_mm)
        self._shape = tuple(int(s) for s in result.grid.shape)

        metal = result.grid > 0
        if not metal.any():
            return

        self._max_time = float(fr.fill_time_s or 0.0)
        if self._max_time <= 0.0:
            finite = np.isfinite(self._fill_time) & metal
            if finite.any():
                self._max_time = float(np.max(self._fill_time[finite]))
        if self._max_time <= 0.0:
            return

        self._flow_grid = self._build_flow_grid()
        if self._flow_grid is None:
            return

        # Optional streamlines for line-trace overlay.
        source_pts = self._build_source_points()
        if source_pts is not None and source_pts.shape[0] > 0:
            try:
                self._streamlines = self._integrate_streamlines(source_pts)
            except Exception:
                self._streamlines = None

        self._current_time = 0.0
        self._update_scene()

    def _build_flow_grid(self) -> Optional[pv.ImageData]:
        """Create a PyVista ImageData with the fill_time scalar.

        Outside-metal cells are given a sentinel value larger than max_time so
        the isosurface only appears inside the metal region and naturally stops
        at the metal/air boundary.
        """
        nx, ny, nz = self._shape
        ft = self._fill_time.copy()
        # Use a sentinel comfortably above max_time; contour uses it as the
        # outside domain value.
        sentinel = self._max_time + 1.0
        ft[self._result.grid == 0] = sentinel

        image = pv.ImageData(dimensions=(nx, ny, nz))
        image.origin = tuple(self._origin + 0.5 * self._dx)
        image.spacing = (self._dx, self._dx, self._dx)
        # PyVista/VTK ImageData point IDs use x-fastest (Fortran-order)
        # ordering for the (nx, ny, nz) point array.
        image['fill_time'] = ft.ravel(order='F').astype(np.float64)
        return image

    def _build_source_points(self) -> Optional[np.ndarray]:
        """Seed streamline start points uniformly across the inlet cells."""
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

        n = inlet_idx.shape[0]
        if n > self.MAX_STREAMLINES:
            step = max(1, n // self.MAX_STREAMLINES)
            inlet_idx = inlet_idx[::step][: self.MAX_STREAMLINES]

        return inlet_idx * self._dx + self._origin + 0.5 * self._dx

    def _sample_velocity(self, pos: np.ndarray) -> np.ndarray:
        """Trilinear interpolation of the Darcy velocity field (m/s).

        The velocity array is cell-centred, so we shift by half a voxel when
        converting physical coordinates to map_coordinates indices.
        """
        if pos.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float64)
        inv_dx = 1.0 / self._dx
        ijk = (pos - self._origin) * inv_dx - 0.5
        ijk = np.ascontiguousarray(ijk, dtype=np.float64)
        coords = np.stack([ijk[:, 0], ijk[:, 1], ijk[:, 2]], axis=0)
        out = np.empty((pos.shape[0], 3), dtype=np.float64)
        for comp in range(3):
            out[:, comp] = ndimage.map_coordinates(
                self._velocity[comp],
                coords,
                order=1,
                mode='constant',
                cval=0.0,
            )
        return out

    def _inside_metal(self, pos: np.ndarray) -> np.ndarray:
        """Nearest-neighbour metal mask check for streamline integration."""
        if pos.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        inv_dx = 1.0 / self._dx
        ijk = (pos - self._origin) * inv_dx - 0.5
        coords = np.stack([ijk[:, 0], ijk[:, 1], ijk[:, 2]], axis=0)
        sampled = ndimage.map_coordinates(
            (self._result.grid > 0).astype(np.float32),
            coords,
            order=0,
            mode='constant',
            cval=0.0,
        )
        return sampled > 0.5

    def _integrate_streamlines(self, source_pts: np.ndarray) -> Optional[pv.PolyData]:
        """Integrate streamlines from inlet seeds through the velocity field."""
        n = source_pts.shape[0]
        pos = source_pts.copy().astype(np.float64)
        active = np.ones(n, dtype=bool)
        t = np.zeros(n, dtype=np.float64)

        line_points: List[List[np.ndarray]] = [[] for _ in range(n)]
        line_vel: List[List[float]] = [[] for _ in range(n)]
        line_time: List[List[float]] = [[] for _ in range(n)]

        # Record source points at t=0.
        for i in range(n):
            line_points[i].append(pos[i].copy())
            line_vel[i].append(0.0)
            line_time[i].append(0.0)

        # Adaptive step budget: ensure the fastest trajectory can reach max_time.
        vmag = np.linalg.norm(self._velocity, axis=0)
        max_speed = float(np.nanmax(vmag)) if vmag.size else 0.0
        if max_speed > 0.0:
            dt_min = self.CFL_FRACTION * self._dx / (1000.0 * max_speed)
            needed_steps = int(np.ceil(self._max_time / dt_min))
            max_steps = max(self.MAX_STEPS, needed_steps)
        else:
            max_steps = self.MAX_STEPS
        max_steps = min(max_steps, 20000)

        pos_active = pos[active]
        for _ in range(max_steps):
            m_active = pos_active.shape[0]
            if m_active == 0:
                break

            active_indices = np.nonzero(active)[0]
            v = self._sample_velocity(pos_active)
            speed = np.linalg.norm(v, axis=1)

            # Stop stagnant lines individually.
            stop = speed <= 1e-9
            if stop.any():
                stop_global = active_indices[stop]
                active[stop_global] = False
                keep = ~stop
                pos_active = pos_active[keep]
                v = v[keep]
                speed = speed[keep]
                if pos_active.shape[0] == 0:
                    break
                active_indices = np.nonzero(active)[0]

            # Choose dt so the fastest point moves at most CFL_FRACTION voxels.
            dt_space = self.CFL_FRACTION * self._dx / (1000.0 * speed.max())
            dt_time = self._max_time / max_steps
            dt = float(min(dt_space, dt_time))
            if dt <= 1e-12:
                break

            new_pos = pos_active + v * (dt * 1000.0)
            inside = self._inside_metal(new_pos)

            for local, global_idx in enumerate(active_indices):
                if inside[local]:
                    pos[global_idx] = new_pos[local]
                    line_points[global_idx].append(pos[global_idx].copy())
                    line_vel[global_idx].append(float(speed[local]))
                    t[global_idx] += dt
                    line_time[global_idx].append(t[global_idx])
                else:
                    active[global_idx] = False

            pos_active = pos[active]

        points = []
        magnitudes = []
        arrival = []
        lines = []
        cursor = 0
        for i in range(n):
            lp = line_points[i]
            if len(lp) < 2:
                continue
            pts = np.stack(lp, axis=0)
            points.append(pts)
            magnitudes.extend(line_vel[i])
            arrival.extend(line_time[i])
            m = pts.shape[0]
            lines.append(m)
            lines.extend(range(cursor, cursor + m))
            cursor += m

        if not points:
            return None

        poly = pv.PolyData()
        poly.points = np.concatenate(points, axis=0)
        poly.lines = np.array(lines, dtype=np.int64)
        poly['velocity_magnitude'] = np.array(magnitudes, dtype=np.float64)
        poly['arrival_time'] = np.array(arrival, dtype=np.float64)
        return self._trim_streamlines(poly)

    def _trim_streamlines(self, poly: pv.PolyData) -> Optional[pv.PolyData]:
        """Remove any trailing points of each line that fell just outside metal."""
        if poly is None or poly.n_points == 0:
            return poly
        pts = poly.points
        mag = poly['velocity_magnitude']
        arr = poly['arrival_time']
        lines = poly.lines
        inside = self._inside_metal(pts)

        out_pts = []
        out_mag = []
        out_arr = []
        out_lines = []
        cursor = 0
        idx = 0
        while idx < len(lines):
            n = int(lines[idx])
            inds = lines[idx + 1 : idx + 1 + n]
            idx += 1 + n

            m = n
            while m > 0 and not inside[inds[m - 1]]:
                m -= 1
            if m < 2:
                continue

            out_lines.append(m)
            out_lines.extend(range(cursor, cursor + m))
            out_pts.append(pts[inds[:m]])
            out_mag.append(mag[inds[:m]])
            out_arr.append(arr[inds[:m]])
            cursor += m

        if not out_pts:
            return None

        trimmed = pv.PolyData()
        trimmed.points = np.concatenate(out_pts, axis=0)
        trimmed.lines = np.array(out_lines, dtype=np.int64)
        trimmed['velocity_magnitude'] = np.concatenate(out_mag)
        trimmed['arrival_time'] = np.concatenate(out_arr)
        return trimmed

    def _build_front(self, t: float) -> Optional[pv.PolyData]:
        """Return the metal fill front as an isosurface of fill_time = t."""
        if self._flow_grid is None:
            return None
        t = float(np.clip(t, 0.0, self._max_time))
        # A tiny offset guarantees the inlet surface is visible at t=0.
        t = max(t, 1e-9)
        front = self._flow_grid.contour([t], scalars='fill_time')
        if front.n_points == 0:
            return None
        # Color the front by the local Darcy velocity at its surface points.
        v = self._sample_velocity(front.points)
        mag = np.linalg.norm(v, axis=1)
        front['velocity_magnitude'] = mag.astype(np.float64)
        return front

    def _visible_streamlines(self, t: float) -> Optional[pv.PolyData]:
        """Return the portion of each streamline that has arrived by time t."""
        if self._streamlines is None:
            return None
        if t >= self._max_time:
            vis = self._streamlines.copy()
            vis.set_active_scalars('velocity_magnitude')
            return vis

        pts = self._streamlines.points
        mag = self._streamlines['velocity_magnitude']
        arr = self._streamlines['arrival_time']
        lines = self._streamlines.lines

        out_pts = []
        out_mag = []
        out_lines = []
        cursor = 0

        idx = 0
        while idx < len(lines):
            n = int(lines[idx])
            inds = lines[idx + 1 : idx + 1 + n]
            idx += 1 + n

            m = 0
            while m < n and arr[inds[m]] <= t:
                m += 1
            # Always show at least the source + first step so streamlines are
            # visible from the very start of the animation.
            m = max(m, 2)
            if m < 2 or m > n:
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
        self._speed_multiplier = max(0.01, float(speed))

    def set_show_surface(self, show: bool) -> None:
        """Show/hide the animated fill-front surface."""
        self._show_front = bool(show)
        self._update_scene()

    def set_show_streamlines(self, show: bool) -> None:
        """Show/hide overlaid streamlines."""
        self._show_streamlines = bool(show)
        self._update_scene()

    def set_current_time(self, t: float) -> None:
        self._current_time = float(np.clip(t, 0.0, self._max_time))
        self._update_scene()

    def play(self) -> None:
        if self._flow_grid is None:
            return
        if self._is_running:
            self._timer.stop()
            self._is_running = False
        else:
            self._timer.start(int(self.TIMER_INTERVAL * 1000))
            self._is_running = True

    def pause(self) -> None:
        if self._is_running:
            self._timer.stop()
            self._is_running = False

    def stop(self) -> None:
        self.pause()
        self._clear_actors()
        self._current_time = 0.0

    def _clear_actors(self) -> None:
        if self._front_actor is not None:
            try:
                self._viewer.remove_actor(self._front_actor)
            except Exception:
                pass
            self._front_actor = None
        if self._streamline_actor is not None:
            try:
                self._viewer.remove_actor(self._streamline_actor)
            except Exception:
                pass
            self._streamline_actor = None

    def _on_timer(self) -> None:
        if self._flow_grid is None:
            return
        dt = self.FRAME_DT * self._speed_multiplier
        self._current_time = min(self._current_time + dt, self._max_time)
        self._update_scene()
        if self._current_time >= self._max_time:
            self.pause()

    def _update_scene(self) -> None:
        # A tiny positive time guarantees the inlet surface/streamlines are
        # visible at the very start of the animation.
        t_eff = max(self._current_time, 1e-9)

        front = None
        if self._show_front:
            front = self._build_front(t_eff)

        vis = None
        if self._show_streamlines and self._streamlines is not None:
            vis = self._visible_streamlines(t_eff)

        if (front is None or front.n_points == 0) and (vis is None or vis.n_points == 0):
            self._clear_actors()
            self._viewer.render()
            return

        scalar_bar_args = {
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
        }

        if front is not None and front.n_points > 0:
            try:
                if self._front_actor is None:
                    self._front_actor = self._viewer.add_mesh(
                        front,
                        scalars='velocity_magnitude',
                        cmap='turbo',
                        opacity=self.FRONT_OPACITY,
                        smooth_shading=True,
                        show_scalar_bar=True,
                        scalar_bar_args=scalar_bar_args,
                        name="flow_animation_front",
                    )
                else:
                    self._front_actor.mapper.dataset = front
            except Exception:
                if self._front_actor is not None:
                    try:
                        self._viewer.remove_actor(self._front_actor)
                    except Exception:
                        pass
                    self._front_actor = None
                self._front_actor = self._viewer.add_mesh(
                    front,
                    scalars='velocity_magnitude',
                    cmap='turbo',
                    opacity=self.FRONT_OPACITY,
                    smooth_shading=True,
                    show_scalar_bar=False,
                    name="flow_animation_front",
                )
        elif self._front_actor is not None:
            try:
                self._viewer.remove_actor(self._front_actor)
            except Exception:
                pass
            self._front_actor = None

        if vis is not None and vis.n_points > 0:
            try:
                if self._streamline_actor is None:
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
        elif self._streamline_actor is not None:
            try:
                self._viewer.remove_actor(self._streamline_actor)
            except Exception:
                pass
            self._streamline_actor = None

        self._current_front = front
        self._current_vis = vis
        self._viewer.render()

    def particle_count(self) -> int:
        """Return a representative element count for the UI label."""
        count = 0
        if self._current_front is not None:
            count += self._current_front.n_points
        if self._current_vis is not None:
            count += self._current_vis.n_cells
        return int(count)

    def particle_diameter_mm(self) -> float:
        return 0.0
