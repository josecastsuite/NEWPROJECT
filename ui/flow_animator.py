"""Lightweight flow-animation engine for JoséCast Analyzer.

Shows 20 red streamlines as thickness-matched tubes and a moving red marker
along each path.  The Darcy solver's ``velocity`` / ``fill_time`` fields are
used for the path and timing, so the visual is consistent with the engineering
values but adds no extra physics of its own.
"""

from typing import List, Optional

import numpy as np
import pyvista as pv
from PyQt6 import QtCore
from scipy import ndimage

from core.types import AnalysisResult


class FlowAnimator(QtCore.QObject):
    """Animate metal flow as 20 red streamlines with channel-aware thickness."""

    TIMER_INTERVAL = 0.05
    FRAME_DT = 0.02
    MAX_STREAMLINES = 20
    MAX_STEPS = 2000
    CFL_FRACTION = 0.5
    N_SIDES = 4
    MARKER_SIZE = 10

    def __init__(self, viewer):
        super().__init__(parent=None)
        self._viewer = viewer
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_timer)

        self._result: Optional[AnalysisResult] = None
        self._streamlines: Optional[pv.PolyData] = None
        self._tube_mesh: Optional[pv.PolyData] = None
        self._edt: Optional[np.ndarray] = None
        self._max_time: float = 0.0

        self._is_running: bool = False
        self._current_time: float = 0.0
        self._speed_multiplier: float = 1.0
        self._show_streamlines: bool = True

        self._streamline_actor = None
        self._marker_actor = None

    def set_result(self, result: Optional[AnalysisResult]) -> None:
        """Attach a completed analysis result and build the streamlines."""
        self.stop()
        self._clear_actors()
        self._result = result
        self._streamlines = None
        self._tube_mesh = None
        self._edt = None

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

        # Channel-thickness proxy for tube radius: distance to nearest non-metal.
        self._edt = ndimage.distance_transform_edt(metal) * self._dx

        source_pts = self._build_source_points()
        if source_pts is None or source_pts.shape[0] == 0:
            return

        try:
            self._streamlines = self._integrate_streamlines(source_pts)
        except Exception:
            self._streamlines = None

        if self._streamlines is None or self._streamlines.n_points == 0:
            return

        self._streamlines = self._trim_streamlines(self._streamlines)
        if self._streamlines is None or self._streamlines.n_points == 0:
            return

        self._compute_streamline_radius()
        self._build_tube_mesh()

        self._current_time = 0.0
        self._update_scene()

    def _build_source_points(self) -> Optional[np.ndarray]:
        """Seed up to MAX_STREAMLINES start points uniformly across the inlet."""
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
        """Trilinear interpolation of the Darcy velocity field (m/s)."""
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

    def _sample_scalar(self, pos: np.ndarray, field: np.ndarray) -> np.ndarray:
        """Trilinear interpolation of a scalar voxel field at physical positions."""
        if pos.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        inv_dx = 1.0 / self._dx
        ijk = (pos - self._origin) * inv_dx - 0.5
        ijk = np.ascontiguousarray(ijk, dtype=np.float64)
        coords = np.stack([ijk[:, 0], ijk[:, 1], ijk[:, 2]], axis=0)
        return ndimage.map_coordinates(
            field,
            coords,
            order=1,
            mode='constant',
            cval=0.0,
        )

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
        """Integrate streamlines from inlet seeds through the Darcy velocity field."""
        n = source_pts.shape[0]
        pos = source_pts.copy().astype(np.float64)
        active = np.ones(n, dtype=bool)
        t = np.zeros(n, dtype=np.float64)

        line_points: List[List[np.ndarray]] = [[] for _ in range(n)]
        line_vel: List[List[float]] = [[] for _ in range(n)]
        line_time: List[List[float]] = [[] for _ in range(n)]

        for i in range(n):
            line_points[i].append(pos[i].copy())
            line_vel[i].append(0.0)
            line_time[i].append(0.0)

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
        return poly

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

    def _compute_streamline_radius(self) -> None:
        """Add a per-point tube radius based on the local channel thickness."""
        if self._streamlines is None or self._edt is None:
            return

        radii = self._sample_scalar(self._streamlines.points, self._edt)
        # Smooth a little so the radius follows the channel centreline rather
        # than hugging wall bumps.
        lines = self._streamlines.lines
        out_radii = np.empty(self._streamlines.n_points, dtype=np.float64)
        idx = 0
        while idx < len(lines):
            n = int(lines[idx])
            inds = lines[idx + 1 : idx + 1 + n]
            idx += 1 + n
            seg = radii[inds]
            seg = ndimage.maximum_filter1d(seg, size=5, mode='nearest')
            seg = np.maximum(seg, self._dx)
            out_radii[inds] = seg

        self._streamlines['tube_radius'] = out_radii

    def _build_tube_mesh(self) -> None:
        """Convert the streamline network into a red tube mesh."""
        if self._streamlines is None or self._streamlines.n_points == 0:
            self._tube_mesh = None
            return

        try:
            self._tube_mesh = self._streamlines.tube(
                radius=0.1,
                scalars='tube_radius',
                absolute=True,
                n_sides=self.N_SIDES,
                capping=False,
            )
            self._tube_mesh.set_active_scalars(None)
        except Exception:
            self._tube_mesh = None

    def _marker_positions(self, t: float) -> Optional[np.ndarray]:
        """Return the current marker position on each streamline."""
        if self._streamlines is None:
            return None

        t = float(np.clip(t, 0.0, self._max_time))
        pts = self._streamlines.points
        arr = self._streamlines['arrival_time']
        lines = self._streamlines.lines

        markers = []
        idx = 0
        while idx < len(lines):
            n = int(lines[idx])
            inds = lines[idx + 1 : idx + 1 + n]
            idx += 1 + n

            p = pts[inds]
            a = arr[inds]
            if a[-1] <= a[0]:
                markers.append(p[-1])
                continue
            if t <= a[0]:
                markers.append(p[0])
            elif t >= a[-1]:
                markers.append(p[-1])
            else:
                x = np.interp(t, a, p[:, 0])
                y = np.interp(t, a, p[:, 1])
                z = np.interp(t, a, p[:, 2])
                markers.append([x, y, z])

        if not markers:
            return None
        return np.array(markers, dtype=np.float64)

    def set_speed_multiplier(self, speed: float) -> None:
        self._speed_multiplier = max(0.01, float(speed))

    def set_show_streamlines(self, show: bool) -> None:
        """Show/hide the red flow paths and moving markers."""
        self._show_streamlines = bool(show)
        self._update_scene()

    def set_current_time(self, t: float) -> None:
        self._current_time = float(np.clip(t, 0.0, self._max_time))
        self._update_scene()

    def play(self) -> None:
        if self._streamlines is None:
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
        if self._streamline_actor is not None:
            try:
                self._viewer.remove_actor(self._streamline_actor)
            except Exception:
                pass
            self._streamline_actor = None
        if self._marker_actor is not None:
            try:
                self._viewer.remove_actor(self._marker_actor)
            except Exception:
                pass
            self._marker_actor = None

    def _on_timer(self) -> None:
        if self._streamlines is None:
            return
        dt = self.FRAME_DT * self._speed_multiplier
        self._current_time = min(self._current_time + dt, self._max_time)
        self._update_scene()
        if self._current_time >= self._max_time:
            self.pause()

    def _update_scene(self) -> None:
        if not self._show_streamlines or self._tube_mesh is None:
            self._clear_actors()
            self._viewer.render()
            return

        if self._streamline_actor is None:
            self._streamline_actor = self._viewer.add_mesh(
                self._tube_mesh,
                color='red',
                opacity=1.0,
                show_scalar_bar=False,
                name="flow_streamlines",
            )
        else:
            self._streamline_actor.mapper.dataset = self._tube_mesh

        markers = self._marker_positions(self._current_time)
        if markers is not None and markers.shape[0] > 0:
            poly = pv.PolyData(markers)
            if self._marker_actor is None:
                self._marker_actor = self._viewer.add_mesh(
                    poly,
                    render_points_as_spheres=True,
                    point_size=self.MARKER_SIZE,
                    color='red',
                    show_scalar_bar=False,
                    name="flow_markers",
                )
            else:
                self._marker_actor.mapper.dataset = poly
        else:
            if self._marker_actor is not None:
                try:
                    self._viewer.remove_actor(self._marker_actor)
                except Exception:
                    pass
                self._marker_actor = None

        self._viewer.render()

    def particle_count(self) -> int:
        """Number of moving markers currently displayed."""
        markers = self._marker_positions(self._current_time)
        return markers.shape[0] if markers is not None else 0

    def line_count(self) -> int:
        """Number of flow-path lines."""
        if self._streamlines is None:
            return 0
        return int(self._streamlines.n_cells)
