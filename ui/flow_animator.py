"""Lightweight flow-animation engine for JoséCast Analyzer.

Renders metal filling and solidification as a sequence of pre-computed,
smooth isosurface frames.  The geometry of each frame comes from the Darcy
``fill_time`` field, and the surface colour comes from a simple cooling model
based on ``fill_time`` and ``solidification_time``.  No extra physics is
introduced: the timing and solidification data come directly from the
engineering solver.
"""

from typing import List, Optional

import numpy as np
import pyvista as pv
from PyQt6 import QtCore, QtWidgets
from scipy import ndimage

try:
    from matplotlib.colors import LinearSegmentedColormap
except Exception:  # pragma: no cover - fallback if matplotlib is missing
    LinearSegmentedColormap = None  # type: ignore

from core.materials import get_alloy, get_mold
from core.types import AnalysisResult, BodyType

from ui.flow_velocity_graph import FlowVelocityGraph


class FlowAnimator(QtCore.QObject):
    """Animate metal filling and solidification as a sequence of 3-D frames."""

    TIMER_INTERVAL = 0.05
    FRAME_DT = 0.02
    MAX_STREAMLINES = 20
    MAX_STEPS = 2000
    CFL_FRACTION = 0.5
    N_SIDES = 4
    MARKER_SIZE = 10
    MAX_ANIM_CELLS = 120_000
    MAX_FRAMES = 48
    SMOOTH_ITER = 6
    SMOOTH_RELAX = 0.1

    def __init__(self, viewer):
        super().__init__(parent=None)
        self._viewer = viewer
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_timer)

        self._result: Optional[AnalysisResult] = None
        self._is_running: bool = False
        self._current_time: float = 0.0
        self._speed_multiplier: float = 1.0
        self._max_time: float = 0.0
        self._show_streamlines: bool = True

        # Streamline data (optional overlay)
        self._streamlines: Optional[pv.PolyData] = None
        self._tube_mesh: Optional[pv.PolyData] = None
        self._edt: Optional[np.ndarray] = None
        self._streamline_actor = None
        self._marker_actor = None

        # Pre-computed volume frames
        self._frame_meshes: List[Optional[pv.PolyData]] = []
        self._frame_times: Optional[np.ndarray] = None
        self._frame_actor = None
        self._cmap = None
        self._t_pour: float = 1600.0
        self._t_liq: float = 1510.0
        self._t_sol: float = 1410.0
        self._t_mold: float = 25.0

        # Downsampled animation grid
        self._base_image: Optional[pv.ImageData] = None
        self._fill_time_d: Optional[np.ndarray] = None
        self._solid_time_d: Optional[np.ndarray] = None
        self._metal_d: Optional[np.ndarray] = None
        self._sentinel: float = 1e9

    def set_result(self, result: Optional[AnalysisResult]) -> None:
        """Attach a completed analysis result and build the animation frames."""
        self.stop()
        self._clear_actors()
        self._reset_data()
        self._result = result

        if result is None or result.flow_result is None:
            return

        fr = result.flow_result
        fill_time = fr.fill_time
        velocity = fr.velocity
        if (
            velocity is None
            or fill_time is None
            or fill_time.size == 0
            or result.grid is None
            or result.grid.size == 0
        ):
            return

        self._fill_time = np.asarray(fill_time, dtype=np.float64)
        self._velocity = np.asarray(velocity, dtype=np.float64)
        self._origin = np.asarray(result.origin_mm, dtype=np.float64)
        self._dx = float(result.dx_mm)
        self._shape = tuple(int(s) for s in result.grid.shape)

        metal = result.grid > 0
        if not metal.any():
            return

        # ---- animation time window ----
        max_fill = self._finite_max(self._fill_time, metal)
        solid_time = result.solidification_time
        if solid_time is not None and solid_time.size == fill_time.size:
            self._solid_time = np.asarray(solid_time, dtype=np.float64)
            max_solid = self._finite_max(self._solid_time, metal)
        else:
            self._solid_time = np.full_like(self._fill_time, np.inf)
            max_solid = -np.inf

        self._max_time = float(max_fill)
        if np.isfinite(max_solid) and max_solid > max_fill:
            self._max_time = float(max_solid)
        if self._max_time <= 0.0:
            return

        # ---- temperature bounds ----
        self._load_temperature_bounds(result)

        # ---- build downsampled animation grid and precompute frames ----
        if not self._build_animation_grid(metal):
            return

        # ---- optional red streamlines through the gating ----
        self._edt = ndimage.distance_transform_edt(metal) * self._dx
        source_pts = self._build_source_points()
        if source_pts is not None and source_pts.shape[0] > 0:
            try:
                self._streamlines = self._integrate_streamlines(source_pts)
                if self._streamlines is not None and self._streamlines.n_points > 0:
                    self._streamlines = self._trim_streamlines(self._streamlines)
                    self._compute_streamline_radius()
                    self._build_tube_mesh()
            except Exception:
                self._streamlines = None

        self._current_time = 0.0
        self._update_scene()

    def _reset_data(self) -> None:
        self._streamlines = None
        self._tube_mesh = None
        self._edt = None
        self._frame_meshes = []
        self._frame_times = None
        self._base_image = None
        self._fill_time_d = None
        self._solid_time_d = None
        self._metal_d = None
        self._frame_actor = None
        self._streamline_actor = None
        self._marker_actor = None
        self._cmap = None

    def _finite_max(self, arr: np.ndarray, mask: np.ndarray) -> float:
        finite = mask & np.isfinite(arr)
        if finite.any():
            return float(np.max(arr[finite]))
        return -np.inf

    def _load_temperature_bounds(self, result: AnalysisResult) -> None:
        cp = getattr(result, "casting_params", None)
        alloy = get_alloy(getattr(result, "alloy_key", "42CrMo4"))
        mold = get_mold(getattr(result, "mold_key", "sand"))

        if cp is not None:
            self._t_pour = float(cp.t_pour_c)
            self._t_mold = float(cp.t_mold_c)
            self._t_liq = float(cp.t_liquidus_c)
            self._t_sol = float(cp.t_solidus_c)
        else:
            self._t_pour = float(alloy.t_pour_c)
            self._t_mold = float(mold.t0_c)
            self._t_liq = float(alloy.t_liquidus_c)
            self._t_sol = float(alloy.t_solidus_c)

    def _build_animation_grid(self, metal: np.ndarray) -> bool:
        """Crop, downsample, and precompute the frame meshes."""
        # Crop to the metal bounding box with a small pad.
        idx = np.nonzero(metal)
        if len(idx[0]) == 0:
            return False

        pad = 1
        bbox = [
            max(0, int(idx[0].min()) - pad),
            min(self._shape[0], int(idx[0].max()) + 1 + pad),
            max(0, int(idx[1].min()) - pad),
            min(self._shape[1], int(idx[1].max()) + 1 + pad),
            max(0, int(idx[2].min()) - pad),
            min(self._shape[2], int(idx[2].max()) + 1 + pad),
        ]

        fill_c = self._fill_time[
            bbox[0] : bbox[1], bbox[2] : bbox[3], bbox[4] : bbox[5]
        ].copy()
        solid_c = self._solid_time[
            bbox[0] : bbox[1], bbox[2] : bbox[3], bbox[4] : bbox[5]
        ].copy()
        metal_c = metal[
            bbox[0] : bbox[1], bbox[2] : bbox[3], bbox[4] : bbox[5]
        ]

        self._sentinel = max(10.0 * self._max_time, 1e6) + 1.0
        fill_c = np.where(np.isfinite(fill_c) & metal_c, fill_c, self._sentinel)
        solid_c = np.where(np.isfinite(solid_c) & metal_c, solid_c, self._sentinel)

        crop_shape = fill_c.shape
        cells = int(np.prod(crop_shape))
        factor = 1
        if cells > self.MAX_ANIM_CELLS:
            factor = max(
                1,
                int(np.ceil((cells / self.MAX_ANIM_CELLS) ** (1.0 / 3.0))),
            )

        if factor > 1:
            target_shape = tuple(max(1, crop_shape[i] // factor) for i in range(3))
            ratios = tuple(
                target_shape[i] / crop_shape[i] for i in range(3)
            )
            fill_d = ndimage.zoom(fill_c, ratios, order=1)
            solid_d = ndimage.zoom(solid_c, ratios, order=1)
            metal_d = ndimage.zoom(metal_c.astype(np.float32), ratios, order=0) > 0.5
            fill_d = np.where(metal_d, fill_d, self._sentinel)
            solid_d = np.where(metal_d, solid_d, self._sentinel)
            spacing = tuple(self._dx * crop_shape[i] / target_shape[i] for i in range(3))
            shape = target_shape
        else:
            fill_d, solid_d, metal_d = fill_c, solid_c, metal_c
            spacing = (self._dx, self._dx, self._dx)
            shape = crop_shape

        origin_c = self._origin + np.array(
            [bbox[0], bbox[2], bbox[4]], dtype=np.float64
        ) * self._dx

        self._fill_time_d = fill_d
        self._solid_time_d = solid_d
        self._metal_d = metal_d

        img = pv.ImageData(
            dimensions=shape, spacing=spacing, origin=origin_c
        )
        img.point_data["fill_time"] = fill_d.ravel(order="F")
        img.point_data["solid_time"] = solid_d.ravel(order="F")
        self._base_image = img

        self._build_frames()
        return bool(self._frame_meshes)

    def _build_frames(self) -> None:
        """Precompute a fixed number of volume frames."""
        n_frames = max(2, self.MAX_FRAMES)
        self._frame_times = np.linspace(0.0, self._max_time, n_frames)
        self._frame_meshes = []

        # Metal-like colour scale: cold/blue-grey -> dark red -> orange -> yellow-white.
        if LinearSegmentedColormap is not None:
            self._cmap = LinearSegmentedColormap.from_list(
                "metal_flow",
                [
                    (0.10, 0.15, 0.25),  # cold solid / mould
                    (0.60, 0.10, 0.05),  # cooling metal
                    (1.00, 0.20, 0.00),  # red hot
                    (1.00, 0.90, 0.40),  # pour / white hot
                ],
            )
        else:
            self._cmap = "coolwarm"

        app = QtCore.QCoreApplication.instance()
        for i, t in enumerate(self._frame_times):
            mesh = self._build_frame(t)
            self._frame_meshes.append(mesh)
            if app is not None and i % 5 == 0:
                app.processEvents()

    def _build_frame(self, t: float) -> Optional[pv.PolyData]:
        """Return the liquid/mushy metal region at time t, coloured by temperature.

        A cell is visible when it has already filled (fill_time <= t) and is not
        yet fully solidified (solid_time > t).  The scalar used for clipping is
        solid_time inside filled cells and a large negative sentinel elsewhere.
        This makes the solidification front (solid_time = t) smooth because the
        scalar is continuous across the liquid/solid boundary, while the filling
        front is simply the filled/not-filled boundary.
        """
        if self._base_image is None:
            return None

        ft = self._fill_time_d
        st = self._solid_time_d
        metal = ft < self._sentinel
        filled = (ft <= t) & metal
        liquid = filled & (st > t)

        # Base temperature from the Darcy fill + thermal solid-time model.
        t_arr = self._compute_temperature(t)
        # Clamp already-solidified (filled but not liquid) cells to solidus so
        # the colour at the solidification front does not bleed cold during
        # interpolation.
        t_arr = np.where(filled & (~liquid), self._t_sol, t_arr)
        self._base_image.point_data["temperature"] = t_arr.ravel(order="F")

        # visible_scalar: use solid_time for filled cells, capped to just above
        # max_time for very late/unsolidified cells.  This avoids a symmetric
        # +/- sentinel scalar pair that puts the isosurface in the middle of an
        # edge (and gives midpoint colours like (1600+25)/2).  Empty/mold
        # cells get a large negative sentinel so they stay excluded.
        vis_cap = self._max_time + 1.0
        clipped_st = np.where(np.isfinite(st), st, vis_cap)
        clipped_st = np.minimum(clipped_st, vis_cap)
        visible_scalar = np.where(filled, clipped_st, -self._sentinel)
        self._base_image.point_data["visible_scalar"] = visible_scalar.ravel(order="F")

        # Keep cells whose effective solid time is >= current time; the
        # isosurface visible_scalar = t is the solidification/filling front.
        clipped = self._base_image.clip_scalar(
            value=float(t), scalars="visible_scalar", invert=False
        )
        if clipped.n_cells == 0:
            return None

        surface = clipped.extract_surface(algorithm="dataset_surface")
        if surface.n_points == 0:
            return None

        surface.set_active_scalars("temperature")
        try:
            surface = surface.compute_normals(
                auto_orient_normals=True, flip_normals=False
            )
        except Exception:
            pass

        try:
            surface = surface.smooth(
                n_iter=self.SMOOTH_ITER,
                relaxation_factor=self.SMOOTH_RELAX,
                feature_angle=60.0,
                boundary_smoothing=True,
                feature_smoothing=False,
            )
        except Exception:
            pass

        return surface

    def _compute_temperature(self, t: float) -> np.ndarray:
        """Return a temperature field for the downsampled animation grid."""
        ft = self._fill_time_d
        st = self._solid_time_d

        local = t - ft
        metal = ft < self._sentinel
        filled = (ft <= t) & metal

        # Cooling time constant chosen so that T reaches solidus at solid_time.
        local_solid = np.maximum(st - ft, 1e-9)
        local_solid = np.where(np.isfinite(local_solid), local_solid, self._max_time)

        ratio = (self._t_sol - self._t_mold) / max(
            self._t_pour - self._t_mold, 1e-9
        )
        ratio = np.clip(ratio, 1e-6, 1.0 - 1e-6)
        log_ratio = np.log(ratio)  # negative
        tau = local_solid / np.maximum(-log_ratio, 1e-9)

        T = self._t_mold + (self._t_pour - self._t_mold) * np.exp(
            -np.maximum(local, 0.0) / tau
        )
        T = np.clip(T, self._t_mold, self._t_pour)
        # Not-yet-filled metal stays at pour temperature so the advancing front
        # appears hot; true empty space is cold.
        T = np.where(filled, T, np.where(metal, self._t_pour, self._t_mold))
        return T

    # ------------------------------------------------------------------
    # Streamline helpers (kept as an optional red overlay)
    # ------------------------------------------------------------------
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
                mode="constant",
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
            mode="constant",
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
            self._result.grid.astype(np.float32),
            coords,
            order=0,
            mode="constant",
            cval=0.0,
        )
        return sampled > 0.5

    def _sample_body_type(self, pos: np.ndarray) -> np.ndarray:
        """Nearest-neighbour body-type sampling at physical positions."""
        if pos.shape[0] == 0:
            return np.empty(0, dtype=np.int16)
        inv_dx = 1.0 / self._dx
        ijk = (pos - self._origin) * inv_dx - 0.5
        coords = np.stack([ijk[:, 0], ijk[:, 1], ijk[:, 2]], axis=0)
        sampled = ndimage.map_coordinates(
            self._result.grid.astype(np.float32),
            coords,
            order=0,
            mode="constant",
            cval=0.0,
        )
        return sampled.astype(np.int16)

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
            body_type = self._sample_body_type(new_pos)

            for local, global_idx in enumerate(active_indices):
                if not inside[local]:
                    active[global_idx] = False
                elif body_type[local] == BodyType.PART:
                    pos[global_idx] = new_pos[local]
                    line_points[global_idx].append(pos[global_idx].copy())
                    line_vel[global_idx].append(float(speed[local]))
                    t[global_idx] += dt
                    line_time[global_idx].append(t[global_idx])
                    active[global_idx] = False
                else:
                    pos[global_idx] = new_pos[local]
                    line_points[global_idx].append(pos[global_idx].copy())
                    line_vel[global_idx].append(float(speed[local]))
                    t[global_idx] += dt
                    line_time[global_idx].append(t[global_idx])

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
        poly["velocity_magnitude"] = np.array(magnitudes, dtype=np.float64)
        poly["arrival_time"] = np.array(arrival, dtype=np.float64)
        return poly

    def _trim_streamlines(self, poly: pv.PolyData) -> Optional[pv.PolyData]:
        """Remove any trailing points of each line that fell just outside metal."""
        if poly is None or poly.n_points == 0:
            return poly
        pts = poly.points
        mag = poly["velocity_magnitude"]
        arr = poly["arrival_time"]
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
        trimmed["velocity_magnitude"] = np.concatenate(out_mag)
        trimmed["arrival_time"] = np.concatenate(out_arr)
        return trimmed

    def _compute_streamline_radius(self) -> None:
        """Add a per-point tube radius based on the local channel thickness."""
        if self._streamlines is None or self._edt is None:
            return

        radii = self._sample_scalar(self._streamlines.points, self._edt)
        lines = self._streamlines.lines
        out_radii = np.empty(self._streamlines.n_points, dtype=np.float64)
        idx = 0
        while idx < len(lines):
            n = int(lines[idx])
            inds = lines[idx + 1 : idx + 1 + n]
            idx += 1 + n
            seg = radii[inds]
            seg = ndimage.maximum_filter1d(seg, size=5, mode="nearest")
            seg = np.maximum(seg, self._dx)
            out_radii[inds] = seg

        self._streamlines["tube_radius"] = out_radii

    def _build_tube_mesh(self) -> None:
        """Convert the streamline network into a red tube mesh."""
        if self._streamlines is None or self._streamlines.n_points == 0:
            self._tube_mesh = None
            return

        try:
            self._tube_mesh = self._streamlines.tube(
                radius=0.1,
                scalars="tube_radius",
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
        arr = self._streamlines["arrival_time"]
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

    # ------------------------------------------------------------------
    # Public control API
    # ------------------------------------------------------------------
    def set_speed_multiplier(self, speed: float) -> None:
        self._speed_multiplier = min(20.0, max(0.01, float(speed)))

    def set_show_streamlines(self, show: bool) -> None:
        """Show/hide the red flow-path lines."""
        self._show_streamlines = bool(show)
        self._update_scene()

    def set_current_time(self, t: float) -> None:
        self._current_time = float(np.clip(t, 0.0, self._max_time))
        self._update_scene()

    def play(self) -> None:
        if not self._frame_meshes:
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
        if self._frame_actor is not None:
            try:
                self._viewer.remove_actor(self._frame_actor)
            except Exception:
                pass
            self._frame_actor = None

    def _on_timer(self) -> None:
        if not self._frame_meshes:
            return
        dt = self.FRAME_DT * self._speed_multiplier
        self._current_time = min(self._current_time + dt, self._max_time)
        self._update_scene()
        if self._current_time >= self._max_time:
            self.pause()

    def _update_scene(self) -> None:
        if not self._frame_meshes or self._frame_times is None:
            return

        idx = max(
            0,
            int(np.searchsorted(self._frame_times, self._current_time, side="right")) - 1,
        )
        idx = min(idx, len(self._frame_meshes) - 1)
        mesh = self._frame_meshes[idx]

        if mesh is None:
            # No geometry at this time step; hide the frame actor.
            if self._frame_actor is not None:
                try:
                    self._viewer.remove_actor(self._frame_actor)
                except Exception:
                    pass
                self._frame_actor = None
        else:
            if self._frame_actor is None:
                self._frame_actor = self._viewer.add_mesh(
                    mesh,
                    cmap=self._cmap,
                    clim=(self._t_sol, self._t_pour),
                    opacity=1.0,
                    scalars="temperature",
                    show_scalar_bar=True,
                    scalar_bar_args={"title": "Sıcaklık (°C)"},
                    name="flow_frame",
                )
            else:
                self._frame_actor.mapper.dataset = mesh

        # Optional red streamlines overlay.
        if self._show_streamlines and self._tube_mesh is not None:
            if self._streamline_actor is None:
                self._streamline_actor = self._viewer.add_mesh(
                    self._tube_mesh,
                    color="red",
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
                        color="red",
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
        else:
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

        self._viewer.render()

    def frame_count(self) -> int:
        return len(self._frame_meshes)

    def current_frame_index(self) -> int:
        if not self._frame_meshes or self._frame_times is None or self._max_time <= 0:
            return -1
        idx = max(
            0,
            int(np.searchsorted(self._frame_times, self._current_time, side="right")) - 1,
        )
        return min(idx, len(self._frame_meshes) - 1)

    def line_count(self) -> int:
        """Number of flow-path lines (streamlines)."""
        if self._streamlines is None:
            return 0
        return int(self._streamlines.n_cells)

    def particle_count(self) -> int:
        """Kept for API compatibility; the new animator uses surface frames."""
        return 0

    def show_velocity_graph(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        """Open a popup with the Darcy velocity vs. fill-time graph."""
        if self._result is None or self._result.flow_result is None:
            return
        dialog = FlowVelocityGraph(self._result, parent)
        dialog.show()
        # Keep a reference so the dialog isn't garbage-collected while open.
        self._velocity_graph_dialog = dialog
