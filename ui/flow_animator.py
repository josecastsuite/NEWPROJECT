"""Lightweight flow-animation engine for JoséCast Analyzer.

Renders metal filling and solidification as a sequence of pre-computed,
smooth isosurface frames.  The geometry of each frame comes from the Darcy
``fill_time`` field, and the surface colour comes from a simple cooling model
based on ``fill_time`` and ``solidification_time``.  No extra physics is
introduced: the timing and solidification data come directly from the
engineering solver.
"""

import time
from typing import List, Optional, Tuple

import numpy as np
import pyvista as pv
from PyQt6 import QtCore, QtWidgets
from scipy import ndimage

try:
    from matplotlib.colors import LinearSegmentedColormap
except Exception:  # pragma: no cover - fallback if matplotlib is missing
    LinearSegmentedColormap = None  # type: ignore

from core.config import load_animation_config
from core.materials import get_alloy, get_mold
from core.types import AnalysisResult, BodyType

from ui.flow_velocity_graph import FlowVelocityGraph


class FlowAnimator(QtCore.QObject):
    """Animate metal filling and solidification as a sequence of 3-D frames."""

    TIMER_INTERVAL = 0.10  # base interval between live frames (s) - slow cinematic
    MAX_STREAMLINES = 20
    MAX_STEPS = 2000
    CFL_FRACTION = 0.5
    N_SIDES = 4
    MARKER_SIZE = 10
    MAX_ANIM_CELLS = 120_000
    MAX_FRAMES = 1350
    MIN_FILL_FRAMES = 1200  # most frames are allocated to the filling phase
    PHI_SIGMA = 1.2  # voxels; controls how liquid surface is smoothed
    DECIMATE_TARGET = 0.5  # reduce triangle count per frame for GPU/CPU relief
    PORE_RISE_SPEED_M_S = 0.05  # buoyant pore drift against gravity

    def __init__(self, viewer, config_path=None):
        super().__init__(parent=None)
        self._viewer = viewer
        # Load user-configurable animation limits (JSON) and shadow the class
        # constants so the rest of the module keeps using self.ATTR_NAME.
        cfg = load_animation_config(config_path)
        self.MAX_ANIM_CELLS = cfg.max_anim_cells
        self.MAX_FRAMES = cfg.max_frames
        self.MIN_FILL_FRAMES = cfg.min_fill_frames
        self.PHI_SIGMA = cfg.phi_sigma
        self.DECIMATE_TARGET = cfg.decimate_target
        self.MAX_STREAMLINES = cfg.max_streamlines
        self.MAX_STEPS = cfg.max_steps
        self.CFL_FRACTION = cfg.cfl_fraction
        self.PORE_RISE_SPEED_M_S = cfg.pore_rise_speed_m_s
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timer)

        self._result: Optional[AnalysisResult] = None
        self._is_running: bool = False
        self._current_frame: int = -1
        self._current_time: float = 0.0
        self._speed_multiplier: float = 1.0
        self._max_time: float = 0.0
        self._max_fill_time: float = 0.0
        self._max_solid_time: float = 0.0
        self._show_streamlines: bool = True

        # Streamline data (optional overlay)
        self._streamlines: Optional[pv.PolyData] = None
        self._tube_mesh: Optional[pv.PolyData] = None
        self._edt: Optional[np.ndarray] = None
        self._streamline_actor = None
        self._marker_actor = None
        self._pore_actor = None

        # Two-phase pre-computed scalar matrices (NO full 3-D mesh geometry).
        # Phase 1: one phi scalar volume per fill frame.
        self._phase1_phi: List[np.ndarray] = []
        self._phase1_times: List[float] = []
        # Phase 2: fixed decimated mesh; per-frame temperature/solid-fraction.
        self._phase2_mesh: Optional[pv.PolyData] = None
        self._phase2_temps: List[np.ndarray] = []
        self._phase2_solids: List[np.ndarray] = []
        self._phase2_times: List[float] = []
        self._n_fill: int = 0
        self._n_solid: int = 0
        self._frame_times: Optional[np.ndarray] = None
        self._frame_actor = None
        self._frame_actor_scalar: str = ""

        self._t_pour: float = 1600.0
        self._t_liq: float = 1510.0
        self._t_sol: float = 1410.0
        self._t_mold: float = 25.0

        # Downsampled animation grid
        self._base_image: Optional[pv.ImageData] = None
        self._fill_time_d: Optional[np.ndarray] = None
        self._solid_time_d: Optional[np.ndarray] = None
        self._metal_d: Optional[np.ndarray] = None
        self._vmag_d: Optional[np.ndarray] = None
        self._outside_mask: Optional[np.ndarray] = None
        self._outside_idx: Optional[np.ndarray] = None
        self._sentinel: float = 1e9

        # Macro porosity data: full-res mask, downsampled mask, and the smooth
        # metal indicator used to carve live holes during solidification.
        self._pore_mask_full: Optional[np.ndarray] = None
        self._pore_mask_d: Optional[np.ndarray] = None
        self._phi_base_d: Optional[np.ndarray] = None
        self._gravity: np.ndarray = np.array([0.0, 0.0, -1.0], dtype=np.float64)

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
        self._max_fill_time = float(max_fill)
        solid_time = result.solidification_time
        if solid_time is not None and solid_time.size == fill_time.size:
            self._solid_time = np.asarray(solid_time, dtype=np.float64)
            # Extrapolate cells that did not reach solidus within the thermal
            # solver horizon using Chvorinov t_s = C * M^2 so the animation can
            # cover their actual solidification and pores can appear in time.
            metal_inf = metal & ~np.isfinite(self._solid_time)
            if (
                metal_inf.any()
                and getattr(result, "sdf", None) is not None
                and result.sdf.size == fill_time.size
                and getattr(result, "chvorinov_c", 0.0) > 0.0
            ):
                sdf = np.maximum(np.asarray(result.sdf, dtype=np.float64), 0.1)
                t_est = result.chvorinov_c * sdf * sdf
                # Never shorten an already-known solidification time.
                known = np.where(np.isfinite(self._solid_time) & metal, self._solid_time, 0.0)
                t_est = np.maximum(t_est, known + 1.0)
                self._solid_time = np.where(metal_inf, t_est, self._solid_time)
            max_solid = self._finite_max(self._solid_time, metal)
        else:
            self._solid_time = np.full_like(self._fill_time, np.inf)
            max_solid = -np.inf
        self._max_solid_time = float(max_solid) if np.isfinite(max_solid) else max_fill

        self._max_time = max(self._max_fill_time, self._max_solid_time)
        if self._max_time <= 0.0:
            return

        # ---- temperature bounds ----
        self._load_temperature_bounds(result)

        # ---- macro-pore mask from backend (full resolution; cropped/downsampled below) ----
        self._load_pore_mask(result)

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

        self._current_frame = 0
        self._current_time = 0.0
        if self._phase1_phi or self._phase2_mesh is not None:
            self._update_scene()

    def _reset_data(self) -> None:
        self._streamlines = None
        self._tube_mesh = None
        self._edt = None
        self._phase1_phi = []
        self._phase1_times = []
        self._phase2_mesh = None
        self._phase2_temps = []
        self._phase2_solids = []
        self._phase2_times = []
        self._n_fill = 0
        self._n_solid = 0
        self._frame_times = None
        self._base_image = None
        self._fill_time_d = None
        self._solid_time_d = None
        self._metal_d = None
        self._outside_mask = None
        self._outside_idx = None
        self._pore_mask_full = None
        self._pore_mask_d = None
        self._filled_d = None
        self._gravity = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self._frame_actor = None
        self._frame_actor_scalar = ""
        self._streamline_actor = None
        self._marker_actor = None
        self._pore_actor = None

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
            g = np.asarray(cp.gravity_vector, dtype=np.float64)
            norm = float(np.linalg.norm(g)) + 1e-9
            self._gravity = g / norm
        else:
            self._t_pour = float(alloy.t_pour_c)
            self._t_mold = float(mold.t0_c)
            self._t_liq = float(alloy.t_liquidus_c)
            self._t_sol = float(alloy.t_solidus_c)

    def _load_pore_mask(self, result: AnalysisResult) -> None:
        """Load or derive the full-resolution macro-pore mask from the result.

        Priority:
          1. result.pore_size_macro_mask (backend classification)
          2. result.pore_size_um >= alloy macro limit
          3. result.pore_size_shrinkage_um >= alloy macro limit
          4. empty mask if no pore data are present.
        """
        grid_shape = tuple(int(s) for s in result.grid.shape)
        mask = np.zeros(grid_shape, dtype=bool)

        macro_thr = 1000.0
        try:
            alloy = get_alloy(getattr(result, "alloy_key", "42CrMo4"))
            macro_thr = float(getattr(alloy, "macro_pore_limit_um", 1000.0))
        except Exception:
            pass

        candidate = getattr(result, "pore_size_macro_mask", None)
        if candidate is not None and candidate.shape == grid_shape:
            mask = np.asarray(candidate, dtype=bool)
        else:
            pore_um = getattr(result, "pore_size_um", None)
            if pore_um is not None and pore_um.shape == grid_shape:
                mask = np.asarray(pore_um, dtype=np.float64) >= macro_thr
            else:
                pore_shrink = getattr(result, "pore_size_shrinkage_um", None)
                if pore_shrink is not None and pore_shrink.shape == grid_shape:
                    mask = np.asarray(pore_shrink, dtype=np.float64) >= macro_thr

        # Restrict to actual metal voxels; EMPTY/void cells cannot host porosity.
        self._pore_mask_full = mask & (result.grid > 0)

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
        pore_c = self._pore_mask_full[
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
            # Nearest-neighbour for fill/solid times: a cell is either filled
            # at a known time or not (sentinel); linear interpolation would
            # invent bogus mid-times next to the sentinel.
            fill_d = ndimage.zoom(fill_c, ratios, order=0)
            solid_d = ndimage.zoom(solid_c, ratios, order=0)
            metal_d = ndimage.zoom(metal_c.astype(np.float32), ratios, order=0) > 0.5
            # Trilinear interpolation for the pore mask keeps the purple cavity
            # surfaces smooth; threshold at 0.5 preserves the binary decision.
            pore_d = ndimage.zoom(pore_c.astype(np.float32), ratios, order=1) > 0.5
            fill_d = np.where(metal_d, fill_d, self._sentinel)
            solid_d = np.where(metal_d, solid_d, self._sentinel)
            spacing = tuple(self._dx * crop_shape[i] / target_shape[i] for i in range(3))
            shape = target_shape
        else:
            fill_d, solid_d, metal_d, pore_d = fill_c, solid_c, metal_c, pore_c
            spacing = (self._dx, self._dx, self._dx)
            shape = crop_shape

        # Restrict pores to metal cells; zoom may have bled into void voxels.
        pore_d = pore_d & metal_d

        origin_c = self._origin + np.array(
            [bbox[0], bbox[2], bbox[4]], dtype=np.float64
        ) * self._dx

        self._fill_time_d = fill_d
        self._solid_time_d = solid_d
        self._metal_d = metal_d
        self._pore_mask_d = pore_d

        img = pv.ImageData(
            dimensions=shape, spacing=spacing, origin=origin_c
        )
        img.point_data["fill_time"] = fill_d.ravel(order="F")
        img.point_data["solid_time"] = solid_d.ravel(order="F")
        self._base_image = img

        self._build_frames()
        return bool(self._phase1_phi) or (self._phase2_mesh is not None)

    def _build_frames(self) -> None:
        """Precompute lightweight scalar matrices; 3-D meshes are built live."""
        n_frames = max(2, self.MAX_FRAMES)
        has_solid = self._max_solid_time > self._max_fill_time
        # Fill phase gets the lion's share of frames so the liquid rise is
        # cinematic and physically readable; remaining frames are for solidification.
        if has_solid:
            self._n_fill = min(self.MIN_FILL_FRAMES, n_frames - 2)
            self._n_solid = n_frames - self._n_fill
        else:
            self._n_fill = n_frames
            self._n_solid = 0

        fill_times = np.linspace(0.0, self._max_fill_time, self._n_fill)
        solid_times = np.linspace(self._max_fill_time, self._max_time, self._n_solid)
        self._frame_times = np.concatenate([fill_times, solid_times])

        app = QtCore.QCoreApplication.instance()

        # Precompute nearest-metal extrapolation indices and static phase-1 colour.
        inv = ~self._metal_d
        if self._metal_d.any() and inv.any():
            self._outside_idx = ndimage.distance_transform_edt(
                inv, return_indices=True, return_distances=False
            )
            self._outside_mask = inv
        else:
            self._outside_idx = None
            self._outside_mask = None

        if self._base_image is not None:
            # The base image carries a placeholder temperature array; each frame
            # overwrites it with the actual temperature field before contouring.
            self._base_image.point_data["temperature"] = np.full(
                self._base_image.n_points, self._t_pour, dtype=np.float32
            )

        # Phase 1: only store the phi scalar volume per frame.
        for i, t in enumerate(fill_times):
            phi = self._build_fill_phi(t)
            self._phase1_phi.append(phi)
            self._phase1_times.append(float(t))
            if app is not None and i % 20 == 0:
                app.processEvents()

        # Phase 2: build the decimated base mesh once, then store per-frame
        # temperature / solid-fraction surface arrays.
        if self._n_solid > 0:
            self._build_solid_base()
            for i, t in enumerate(solid_times[1:], start=1):
                surf_t, surf_sf = self._solid_scalars_for_time(t)
                self._phase2_temps.append(surf_t)
                self._phase2_solids.append(surf_sf)
                self._phase2_times.append(float(t))
                if app is not None and i % 20 == 0:
                    app.processEvents()

    def _build_fill_phi(self, t: float) -> Optional[np.ndarray]:
        """Return the raveled phi level-set for the liquid front at time t."""
        if self._base_image is None:
            return None

        ft = self._fill_time_d
        metal = ft < self._sentinel
        filled = (ft <= t) & metal
        if not filled.any():
            return None

        phi = ndimage.gaussian_filter(
            filled.astype(np.float64), sigma=self.PHI_SIGMA, mode="constant", cval=0.0
        )
        return phi.astype(np.float32).ravel(order="F")

    def _build_solid_base(self) -> None:
        """Precompute the smooth metal indicator used for live phase-2 frames.

        The solidification mesh is rebuilt every frame so that macro-pores can
        appear as true holes in the metal surface and a separate purple pore
        surface can grow with time.  This method stores the base metal level-set
        (without pores) and an empty pore-phi placeholder; per-frame updates
        subtract the active pore mask and contour both surfaces live.
        """
        if self._base_image is None:
            return

        ft = self._fill_time_d
        metal = ft < self._sentinel
        filled = (ft <= self._max_fill_time) & metal
        if not filled.any():
            self._filled_d = None
            self._phase2_mesh = None
            return

        # Boolean metal indicator for the fully-filled part.  Pore cells are
        # carved out per-frame in _update_scene by smoothing (filled & ~pore).
        self._filled_d = filled

        # Placeholders; overwritten every phase-2 frame.
        self._base_image.point_data["phi"] = np.zeros(
            self._base_image.n_points, dtype=np.float32
        )
        self._base_image.point_data["pore_phi"] = np.zeros(
            self._base_image.n_points, dtype=np.float32
        )
        # Temperature array is also overwritten per frame.
        self._base_image.point_data["temperature"] = np.full(
            self._base_image.n_points, self._t_pour, dtype=np.float32
        )

        # The fixed phase-2 mesh is no longer used; surfaces are contoured live.
        self._phase2_mesh = None

    def _finalize_surface(
        self, surface: pv.PolyData, active_scalars: str = "velocity_magnitude"
    ) -> pv.PolyData:
        """Smooth, decimate and normalise a surface mesh."""
        surface = self._windowed_sinc_smooth(surface)
        try:
            surface = surface.compute_normals(
                auto_orient_normals=True, flip_normals=False
            )
        except Exception:
            pass
        surface = self._decimate(surface)
        surface.set_active_scalars(active_scalars)
        return surface

    def _decimate(self, mesh: pv.PolyData) -> pv.PolyData:
        """Reduce polygon count to keep GPU/CPU usage low."""
        try:
            if mesh.n_cells == 0:
                return mesh
            dec = mesh.decimate(
                target_reduction=self.DECIMATE_TARGET,
                volume_preserving=True,
                attribute_error_bound=0.5,
                preserve_topology=True,
            )
            return dec if dec.n_points > 0 else mesh
        except Exception:
            return mesh

    def _solid_scalars_for_time(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return per-surface temperature and solid-fraction for a solid phase time."""
        if self._phase2_mesh is None or self._phase2_mesh.n_points == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

        ft = self._fill_time_d
        st = self._solid_time_d

        t_arr = self._compute_temperature(t)
        # Extrapolate metal values into the surrounding void so the isosurface
        # never interpolates cold mould/air values.
        t_color = self._extrapolate_metal_scalar(t_arr)

        dt = t - ft
        solid_span = st - ft
        sf = np.clip(
            np.divide(dt, solid_span, out=np.zeros_like(dt), where=solid_span > 1e-9),
            0.0,
            1.0,
        )
        sf = self._extrapolate_metal_scalar(sf)

        # Sample the volume scalars onto the fixed decimated surface.
        surf_t = self._sample_scalar_at_points(
            self._phase2_mesh.points, t_color, order=1
        )
        surf_sf = self._sample_scalar_at_points(
            self._phase2_mesh.points, sf, order=1
        )
        return surf_t, surf_sf

    def _extrapolate_metal_scalar(self, field: np.ndarray) -> np.ndarray:
        """Copy the nearest metal value into every void/off-metal cell.

        This is precomputed once via distance_transform_edt and reused for every
        solid-phase frame, so per-frame extrapolation stays cheap.
        """
        color = field.copy()
        if self._outside_idx is not None and self._outside_mask is not None:
            z, y, x = self._outside_idx
            mask = self._outside_mask
            color[mask] = field[z[mask], y[mask], x[mask]]
        return color

    def _windowed_sinc_smooth(self, mesh: pv.PolyData) -> pv.PolyData:
        """Apply a VTK windowed-sinc filter for fluid-like smooth surfaces."""
        try:
            import vtk

            smooth = vtk.vtkWindowedSincPolyDataFilter()
            smooth.SetInputData(mesh)
            smooth.SetNumberOfIterations(20)
            smooth.SetPassBand(0.1)
            smooth.SetFeatureAngle(120.0)
            smooth.BoundarySmoothingOn()
            smooth.FeatureEdgeSmoothingOff()
            smooth.NonManifoldSmoothingOff()
            smooth.NormalizeCoordinatesOn()
            smooth.Update()
            out = pv.PolyData(smooth.GetOutput())
            return out if out.n_points > 0 else mesh
        except Exception:
            return mesh

    def _sample_scalar_at_points(
        self, points: np.ndarray, field: np.ndarray, order: int = 1
    ) -> np.ndarray:
        """Sample a 3-D scalar field at physical point positions."""
        origin = np.asarray(self._base_image.origin, dtype=np.float64)
        spacing = np.asarray(self._base_image.spacing, dtype=np.float64)
        ijk = (points - origin) / spacing
        return np.asarray(
            ndimage.map_coordinates(
                field,
                ijk.T,
                order=order,
                mode="nearest",
            ),
            dtype=np.float64,
        )

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
    def _interval_ms(self) -> int:
        """Timer interval in ms for the current speed multiplier."""
        ms = int(1000.0 * self.TIMER_INTERVAL / max(0.01, self._speed_multiplier))
        return max(10, min(ms, 10000))

    def set_speed_multiplier(self, speed: float) -> None:
        self._speed_multiplier = min(20.0, max(0.01, float(speed)))

    def set_show_streamlines(self, show: bool) -> None:
        """Show/hide the red flow-path lines."""
        self._show_streamlines = bool(show)
        self._update_scene()

    def set_current_time(self, t: float) -> None:
        if self._frame_times is None or len(self._frame_times) == 0:
            return
        t = float(np.clip(t, 0.0, self._max_time))
        self._current_frame = int(
            max(0, np.searchsorted(self._frame_times, t, side="right") - 1)
        )
        self._update_scene()

    def play(self) -> None:
        if self._frame_times is None or len(self._frame_times) == 0:
            return
        if self._is_running:
            # Already playing -> toggle pause.
            self.pause()
            return
        # Restart from the beginning if already at the end.
        if self._current_frame >= len(self._frame_times) - 1:
            self._current_frame = 0
        self._is_running = True
        self._timer.start(self._interval_ms())

    def pause(self) -> None:
        if self._is_running:
            self._is_running = False
            self._timer.stop()

    def stop(self) -> None:
        self.pause()
        self._clear_actors()
        self._current_frame = 0
        self._current_time = 0.0
        if self._frame_times is not None and len(self._frame_times) > 0:
            self._update_scene()

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
        if self._pore_actor is not None:
            try:
                self._viewer.remove_actor(self._pore_actor)
            except Exception:
                pass
            self._pore_actor = None

    def _on_timer(self) -> None:
        if not self._is_running or self._frame_times is None or len(self._frame_times) == 0:
            return
        if self._current_frame >= len(self._frame_times) - 1:
            self.pause()
            return
        t0 = time.perf_counter()
        self._current_frame += 1
        self._update_scene()
        if self._is_running:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            self._timer.start(max(10, self._interval_ms() - elapsed_ms))

    def _update_scene(self) -> None:
        if self._frame_times is None or len(self._frame_times) == 0:
            return

        n_frames = len(self._frame_times)
        frame = max(0, min(self._current_frame, n_frames - 1))
        self._current_frame = frame
        self._current_time = float(self._frame_times[frame])

        if frame < self._n_fill:
            # Phase 1: build the liquid surface live from the stored phi matrix
            # and colour it by temperature (hot red -> cold blue).
            phi = self._phase1_phi[frame] if 0 <= frame < len(self._phase1_phi) else None
            if phi is None or self._base_image is None:
                if self._frame_actor is not None:
                    try:
                        self._viewer.remove_actor(self._frame_actor)
                    except Exception:
                        pass
                    self._frame_actor = None
                    self._frame_actor_scalar = ""
            else:
                t = self._frame_times[frame]
                T = self._compute_temperature(t).astype(np.float32).ravel(order="F")
                self._base_image.point_data["temperature"] = T
                self._base_image.point_data["phi"] = phi
                surface = self._base_image.contour(isosurfaces=[0.5], scalars="phi")
                if surface.n_points == 0:
                    if self._frame_actor is not None:
                        try:
                            self._viewer.remove_actor(self._frame_actor)
                        except Exception:
                            pass
                        self._frame_actor = None
                        self._frame_actor_scalar = ""
                else:
                    surface = self._finalize_surface(
                        surface, active_scalars="temperature"
                    )
                    if self._frame_actor is None or self._frame_actor_scalar != "temperature":
                        if self._frame_actor is not None:
                            try:
                                self._viewer.remove_actor(self._frame_actor)
                            except Exception:
                                pass
                        self._frame_actor = self._viewer.add_mesh(
                            surface,
                            cmap=self._metal_cmap(),
                            clim=(self._t_mold, self._t_pour),
                            opacity=1.0,
                            scalars="temperature",
                            show_scalar_bar=True,
                            scalar_bar_args={"title": "Sıcaklık (°C)"},
                            name="flow_frame",
                        )
                        self._frame_actor_scalar = "temperature"
                    else:
                        self._frame_actor.mapper.dataset = surface

            # Macro porosity is not shown during filling; remove any stale actor.
            if self._pore_actor is not None:
                try:
                    self._viewer.remove_actor(self._pore_actor)
                except Exception:
                    pass
                self._pore_actor = None
        else:
            # Phase 2: live reconstruction of the metal surface with macro-pore
            # holes carved out, plus a separate purple pore_actor that grows as
            # solid_time <= t advances.
            s_idx = frame - self._n_fill
            if (
                self._filled_d is None
                or self._pore_mask_d is None
                or s_idx >= self._n_solid
                or self._base_image is None
            ):
                return
            t = self._frame_times[frame]

            # Temperature field for this instant (cools from pour down to mold).
            T = self._compute_temperature(t)
            T = self._extrapolate_metal_scalar(T).astype(np.float32).ravel(order="F")
            self._base_image.point_data["temperature"] = T

            # Active pores: cells that are in the macro-pore mask and have already
            # solidified (solid_time <= current time).  This makes pores appear
            # one by one as the liquid path closes.
            active_pore = self._pore_mask_d & (self._solid_time_d <= t)

            # P4 prototype: gravity-driven pore rise.  Pores are buoyant in the
            # still-liquid metal and drift opposite to the gravity vector.
            # The shift is sub-voxel and grows with time since filling ended.
            dt_pore = max(0.0, t - self._max_fill_time)
            rise_m = dt_pore * self.PORE_RISE_SPEED_M_S
            if rise_m > 1e-6 and self._dx > 0.0:
                rise_voxels = rise_m * 1000.0 / self._dx
                shift = tuple(-rise_voxels * self._gravity[i] for i in range(3))
                shifted = ndimage.shift(
                    active_pore.astype(np.float64),
                    shift,
                    order=1,
                    mode="constant",
                    cval=0.0,
                )
                active_pore = shifted > 0.5

            pore_smooth = ndimage.gaussian_filter(
                active_pore.astype(np.float64), sigma=self.PHI_SIGMA, mode="constant", cval=0.0
            ).astype(np.float32)

            # Metal level-set: smooth the (filled metal minus active pore) volume.
            # Cells inside active pores become phi = 0, producing true holes.
            phi_t = ndimage.gaussian_filter(
                (self._filled_d & ~active_pore).astype(np.float64),
                sigma=self.PHI_SIGMA,
                mode="constant",
                cval=0.0,
            ).astype(np.float32)
            self._base_image.point_data["phi"] = phi_t.ravel(order="F")

            surface = self._base_image.contour(isosurfaces=[0.5], scalars="phi")
            if surface.n_points == 0:
                if self._frame_actor is not None:
                    try:
                        self._viewer.remove_actor(self._frame_actor)
                    except Exception:
                        pass
                    self._frame_actor = None
                    self._frame_actor_scalar = ""
            else:
                surface = self._finalize_surface(surface, active_scalars="temperature")
                if self._frame_actor is None or self._frame_actor_scalar != "temperature":
                    if self._frame_actor is not None:
                        try:
                            self._viewer.remove_actor(self._frame_actor)
                        except Exception:
                            pass
                    self._frame_actor = self._viewer.add_mesh(
                        surface,
                        cmap=self._metal_cmap(),
                        clim=(self._t_mold, self._t_pour),
                        opacity=1.0,
                        scalars="temperature",
                        show_scalar_bar=True,
                        scalar_bar_args={"title": "Sıcaklık (°C)"},
                        name="flow_frame",
                    )
                    self._frame_actor_scalar = "temperature"
                else:
                    self._frame_actor.mapper.dataset = surface

            # Pore surface: contour the smoothed active-pore indicator at 0.5.
            # Rendered in bright purple (#800080) with 0.9 opacity so deep cavities
            # remain visible even behind the metal surface.
            self._base_image.point_data["pore_phi"] = pore_smooth.ravel(order="F")
            pore_surface = self._base_image.contour(isosurfaces=[0.5], scalars="pore_phi")
            if pore_surface.n_points == 0:
                if self._pore_actor is not None:
                    try:
                        self._viewer.remove_actor(self._pore_actor)
                    except Exception:
                        pass
                    self._pore_actor = None
            else:
                pore_surface = self._finalize_surface(pore_surface, active_scalars="pore_phi")
                pore_surface.set_active_scalars(None)
                if self._pore_actor is None:
                    self._pore_actor = self._viewer.add_mesh(
                        pore_surface,
                        color="#800080",
                        opacity=0.9,
                        show_scalar_bar=False,
                        name="pore_actor",
                    )
                else:
                    self._pore_actor.mapper.dataset = pore_surface

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

    def _metal_cmap(self):
        # Hot metal = red (high scalar), cold/solid = blue (low scalar).
        return "coolwarm"

    def frame_count(self) -> int:
        return len(self._frame_times) if self._frame_times is not None else 0

    def current_frame_index(self) -> int:
        return self._current_frame

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
