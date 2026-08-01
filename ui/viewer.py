"""PyVistaQt 3D viewer wrapper for JoseCast Analyzer v8.x."""

import heapq
from typing import Callable, List, Optional, Tuple

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

from core.gating import (
    _characteristic_cross_section_area,
    _flow_axis,
    _section_2d_area_and_perim,
    _sprue_circular_base_and_throat,
)
from core.materials import get_alloy
from core.sdf_analyzer import _trace_path_to_riser
from core.types import BODY_TYPE_LABELS, AnalysisResult, Body, BodyType, GatingNode, HotSpot, RefinementRegion
from ui.flow_animator import FlowAnimator


BODY_COLORS = {
    BodyType.PART: "#F5F5F5",
    BodyType.RISER: "#00E676",
    BodyType.INGATE: "#2979FF",
    BodyType.RUNNER: "#FF9100",
    BodyType.SPRUE: "#D500F9",
    BodyType.CORE: "#BCAAA4",
    BodyType.COOLING_SPRUE: "#00E5FF",
    BodyType.FILTER: "#90A4AE",
    BodyType.POURING_BASIN: "#7C4DFF",
    BodyType.SPRUE_THROAT: "#FF4081",
    BodyType.DISTRIBUTOR: "#76FF03",
    BodyType.CURUFLUK: "#00BFA5",
}

BODY_LEGEND_LABELS = {
    BodyType.PART: "Parça",
    BodyType.RISER: "Besleyici",
    BodyType.INGATE: "Meme",
    BodyType.RUNNER: "Yolluk",
    BodyType.SPRUE: "Döküm Ağzı",
    BodyType.CORE: "Maça",
    BodyType.COOLING_SPRUE: "Soğ. Döküm Ağzı",
    BodyType.FILTER: "Filtre",
    BodyType.POURING_BASIN: "Döküm Havzası",
    BodyType.SPRUE_THROAT: "D.a Boğazı",
    BodyType.DISTRIBUTOR: "Dağıtıcı",
    BodyType.CURUFLUK: "Curufluk",
}

BODY_OPACITY = {
    BodyType.PART: 0.35,
    BodyType.RISER: 1.0,
    BodyType.INGATE: 1.0,
    BodyType.RUNNER: 1.0,
    BodyType.SPRUE: 1.0,
    BodyType.CORE: 1.0,
    BodyType.COOLING_SPRUE: 1.0,
    BodyType.FILTER: 1.0,
    BodyType.POURING_BASIN: 1.0,
    BodyType.SPRUE_THROAT: 1.0,
    BodyType.DISTRIBUTOR: 1.0,
    BodyType.CURUFLUK: 1.0,
}

# Post-analysis transparency: all bodies become translucent so internal
# hotspots, Niyama surfaces and flow fields are visible through the geometry.
BODY_OPACITY_POST = {
    BodyType.PART: 0.60,
    BodyType.RISER: 0.35,
    BodyType.INGATE: 0.35,
    BodyType.RUNNER: 0.35,
    BodyType.SPRUE: 0.35,
    BodyType.CORE: 0.35,
    BodyType.COOLING_SPRUE: 0.35,
    BodyType.FILTER: 0.35,
    BodyType.POURING_BASIN: 0.35,
    BodyType.SPRUE_THROAT: 0.35,
    BodyType.DISTRIBUTOR: 0.35,
    BodyType.CURUFLUK: 0.35,
}


def _nearest_body_voxel(
    body_index: np.ndarray,
    body_idx: int,
    origin_mm: np.ndarray,
    dx_mm: float,
    point_mm: np.ndarray,
) -> Optional[Tuple[int, int, int]]:
    """Return the body voxel closest to a 3-D point."""
    p = np.asarray(point_mm, dtype=np.float64)
    vox = np.round((p - origin_mm) / dx_mm - 0.5).astype(int)
    shape = body_index.shape
    if all(0 <= vox[i] < shape[i] for i in range(3)) and body_index[tuple(vox)] == body_idx:
        return tuple(int(x) for x in vox)
    # Fallback: search all body voxels for the nearest one.
    coords = np.argwhere(body_index == body_idx)
    if len(coords) == 0:
        return None
    dists = np.linalg.norm((coords + 0.5) * dx_mm + origin_mm - p, axis=1)
    best = int(np.argmin(dists))
    return tuple(int(x) for x in coords[best])


def _voxel_path_through_body(
    body_index: np.ndarray,
    body_idx: int,
    origin_mm: np.ndarray,
    dx_mm: float,
    start_mm: np.ndarray,
    end_mm: np.ndarray,
    step: int = 2,
) -> Optional[np.ndarray]:
    """Find a 26-neighbor A* path through one body's voxels.

    The returned points are voxel centres plus the exact start/end points so the
    polyline follows the real body interior instead of a straight chord.
    """
    start_v = _nearest_body_voxel(body_index, body_idx, origin_mm, dx_mm, start_mm)
    end_v = _nearest_body_voxel(body_index, body_idx, origin_mm, dx_mm, end_mm)
    if start_v is None or end_v is None:
        return None
    if start_v == end_v:
        return np.vstack([start_mm, end_mm])

    mask = body_index == body_idx
    shape = mask.shape
    # 26-neighbour offsets and costs
    neigh = []
    for iz in (-1, 0, 1):
        for iy in (-1, 0, 1):
            for ix in (-1, 0, 1):
                if iz == 0 and iy == 0 and ix == 0:
                    continue
                d = (iz, iy, ix)
                neigh.append((d, np.sqrt(iz * iz + iy * iy + ix * ix)))

    def heuristic(a, b):
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    open_set = [(heuristic(start_v, end_v), 0.0, start_v)]
    g_score = {start_v: 0.0}
    came_from = {}
    closed = set()

    while open_set:
        _, g, cur = heapq.heappop(open_set)
        if cur in closed:
            continue
        closed.add(cur)
        if cur == end_v:
            break
        for off, cost in neigh:
            nxt = (cur[0] + off[0], cur[1] + off[1], cur[2] + off[2])
            if any(nxt[i] < 0 or nxt[i] >= shape[i] for i in range(3)):
                continue
            if not mask[nxt]:
                continue
            if nxt in closed:
                continue
            ng = g + cost
            if nxt not in g_score or ng < g_score[nxt]:
                g_score[nxt] = ng
                came_from[nxt] = cur
                heapq.heappush(open_set, (ng + heuristic(nxt, end_v), ng, nxt))

    if end_v not in came_from and end_v != start_v:
        return None

    path_v = [end_v]
    cur = end_v
    while cur in came_from:
        cur = came_from[cur]
        path_v.append(cur)
    path_v.reverse()

    # Subsample to keep the polyline light but smooth.
    path_v = path_v[::step]
    if path_v[-1] != end_v:
        path_v.append(end_v)

    centres = (np.array(path_v, dtype=np.float64) + 0.5) * dx_mm + origin_mm
    return np.vstack([np.asarray(start_mm, dtype=np.float64).reshape(1, -1), centres, np.asarray(end_mm, dtype=np.float64).reshape(1, -1)])


def _scalar_bar_args(title: str, pos: Tuple[float, float], clim: Optional[Tuple[float, float]] = None) -> dict:
    """Build scalar-bar args that avoid label overlap for the value range."""
    fmt = "%.2f"
    if clim is not None:
        vmax = max(abs(float(clim[0])), abs(float(clim[1])))
        if vmax >= 1000.0:
            fmt = "%.0f"
        elif vmax >= 100.0:
            fmt = "%.1f"
        elif vmax < 1.0:
            fmt = "%.3f"
    # Renk skalasını 1.5x uzatmak için boyutları ve sağa yaslı konumu güncelle.
    width = 0.15 * 1.5
    height = 0.08 * 1.5
    # Sağ kenarı sabit tutmak için sol kenarı sola kaydır.
    pos_x = max(0.0, pos[0] + 0.15 - width)
    return {
        "color": "#00ffff",
        "title_font_size": 10,
        "label_font_size": 8,
        "fmt": fmt,
        "n_labels": 5,
        "vertical": False,
        "position_x": pos_x,
        "position_y": pos[1],
        "width": width,
        "height": height,
        "title": title,
    }


class Analyzer3DViewer(QtInteractor):
    """Extended PyVistaQt interactor for casting analysis."""

    def __init__(self, parent=None, off_screen: bool = False):
        super().__init__(parent=parent, off_screen=off_screen)
        self.set_background("#050505", top="#0a0a1a")
        self.add_axes(line_width=2, color="#00ffff")
        try:
            self.add_light(pv.Light(light_type="headlight"))
        except Exception:
            pass
        try:
            self.enable_depth_peeling(number_of_peels=32, occlusion_ratio=0.0)
            if self.ren_win is not None:
                self.ren_win.SetMultiSamples(0)
                self.ren_win.SetAlphaBitPlanes(1)
        except Exception:
            pass

        self._body_actors: List = []
        self._part_mesh_pv: Optional[pv.PolyData] = None
        self._hotspot_actors: List = []
        self._hotspot_label_actor = None
        self._risk_actor = None
        self._porosity_actor = None
        self._niyama_actors: List = []
        self._path_actors: List = []
        self._slice_actors: List = []
        self._local_actors: List = []
        self._section_actors: List = []
        self._section_picker = None
        self._flow_actor = None
        self._flow_node_actor = None
        self._flow_arrow_actor = None
        self.flow_animator = FlowAnimator(self)
        self._body_legend_actor = None
        # voxel data needed to trace flow lines through body interiors
        self._bodies: List[Body] = []
        self._body_index: Optional[np.ndarray] = None
        self._origin_mm: Optional[np.ndarray] = None
        self._dx_mm: float = 0.0

    def _update_body_legend(self, bodies: List[Body]) -> None:
        """Add a top-right PyVista legend showing only body types in the scene."""
        if self._body_legend_actor is not None:
            try:
                self.remove_actor(self._body_legend_actor)
            except Exception:
                pass
            self._body_legend_actor = None

        present = sorted(
            {body.body_type for body in bodies if len(body.faces) > 0},
            key=lambda x: int(x),
        )
        if not present:
            return

        entries = []
        for bt in present:
            color = BODY_COLORS.get(bt, "#F5F5F5")
            label = BODY_LEGEND_LABELS.get(bt, str(bt))
            entries.append([label, color, "circle"])

        n = len(entries)
        line_height = 0.035
        height = min(0.45, max(0.06, line_height * n))
        max_chars = max(len(entry[0]) for entry in entries)
        width = min(0.30, max(0.12, max_chars * 0.011))

        self._body_legend_actor = self.add_legend(
            labels=entries,
            loc="upper right",
            bcolor=(0.08, 0.08, 0.10),
            background_opacity=0.85,
            face="circle",
            size=(width, height),
            name="body_legend",
        )

        # PyVista's default loc math treats the vertical size as the right margin,
        # so override the viewport position explicitly to keep the legend inside
        # the viewer area and away from the right scrollbar/panel.
        right_margin = 0.08
        top_margin = 0.05
        x = 1.0 - width - right_margin
        y = 1.0 - height - top_margin
        self._body_legend_actor.SetPosition(x, y)
        self._body_legend_actor.SetPosition2(width, height)
        self._body_legend_actor.SetPadding(2)
        text_prop = self._body_legend_actor.GetEntryTextProperty()
        text_prop.SetFontSize(11)
        text_prop.SetBold(0)

    def set_gating_data(
        self,
        bodies: List[Body],
        body_index: np.ndarray,
        origin_mm: np.ndarray,
        dx_mm: float,
    ) -> None:
        """Store voxel data so flow lines can be traced through body interiors."""
        self._bodies = list(bodies)
        self._body_index = np.asarray(body_index)
        self._origin_mm = np.asarray(origin_mm)
        self._dx_mm = float(dx_mm)

    def clear_scene(self):
        self.flow_animator.stop()
        self.clear_actors()
        self.add_axes(line_width=2, color="#00ffff")
        self._body_actors.clear()
        self._part_mesh_pv = None
        self._hotspot_actors.clear()
        self._hotspot_label_actor = None
        self._risk_actor = None
        self._porosity_actor = None
        self._niyama_actors.clear()
        self._path_actors.clear()
        self._slice_actors.clear()
        self._local_actors.clear()
        self._flow_actor = None
        self._flow_node_actor = None
        self._flow_arrow_actor = None
        self._body_legend_actor = None
        self._bodies = []
        self._body_index = None
        self._origin_mm = None
        self._dx_mm = 0.0
        self._clear_section_actors()

    def show_bodies(
        self,
        bodies: List[Body],
        reset_camera: bool = True,
        analysis_mode: bool = False,
        selected_body: Optional[Body] = None,
    ):
        """Display original body meshes colored by type."""
        for actor in self._body_actors:
            self.remove_actor(actor)
        self._body_actors.clear()
        self._part_mesh_pv = None

        opacity_map = BODY_OPACITY_POST if analysis_mode else BODY_OPACITY
        part_vertices: List[np.ndarray] = []
        part_faces: List[np.ndarray] = []
        offset = 0
        for body in bodies:
            if len(body.faces) == 0:
                continue
            faces = np.c_[np.full(len(body.faces), 3, dtype=np.int64), body.faces].ravel()
            mesh = pv.PolyData(body.vertices, faces)
            is_selected = selected_body is not None and body is selected_body
            color = "#ff0000" if is_selected else BODY_COLORS.get(body.body_type, "#F5F5F5")
            opacity = 1.0 if is_selected else opacity_map.get(body.body_type, 1.0)
            if selected_body is not None and not is_selected and not analysis_mode:
                opacity = 0.25
            actor = self.add_mesh(
                mesh,
                color=color,
                opacity=opacity,
                show_edges=False,
                smooth_shading=True,
                pickable=False,
            )
            self._body_actors.append(actor)
            if body.body_type == BodyType.PART:
                part_vertices.append(np.asarray(body.vertices, dtype=np.float64))
                part_faces.append(np.asarray(body.faces, dtype=np.int64) + offset)
                offset += len(body.vertices)
        if part_vertices:
            merged_v = np.vstack(part_vertices)
            merged_f = np.vstack(part_faces)
            tri = np.c_[np.full(len(merged_f), 3, dtype=np.int64), merged_f].ravel()
            self._part_mesh_pv = pv.PolyData(merged_v, tri)
        self._update_body_legend(bodies)
        if reset_camera:
            self.reset_camera()

    def _remove_scalar_bar(self, title: str):
        if title in self.scalar_bars:
            try:
                self.remove_scalar_bar(title)
            except Exception:
                pass

    def _make_grid(self, result: AnalysisResult, scalars: np.ndarray, name: str) -> pv.ImageData:
        """Build a PyVista ImageData (voxel grid) with point-centered scalars and a metal mask."""
        grid = pv.ImageData()
        grid.dimensions = np.array(result.grid.shape) + 1
        grid.origin = result.origin_mm
        grid.spacing = (result.dx_mm, result.dx_mm, result.dx_mm)
        grid.cell_data[name] = np.asarray(scalars).ravel(order="F")
        grid.cell_data["is_metal"] = result.is_metal.ravel(order="F").astype(np.float64)
        grid.cell_data["part"] = (result.grid == BodyType.PART).ravel(order="F").astype(np.float64)
        # Contour / slice filters require point data; convert and keep both scalars.
        return grid.cell_data_to_point_data()

    def _metal_only(self, grid: pv.ImageData) -> pv.UnstructuredGrid:
        """Return only fully-metal cells from a grid."""
        return grid.threshold([1.0, 1.0], scalars="is_metal", all_scalars=True)

    def _part_only(self, grid: pv.ImageData) -> pv.UnstructuredGrid:
        """Return only part cells; porosity/Niyama belong to the casting, not risers/gating."""
        return grid.threshold([1.0, 1.0], scalars="part", all_scalars=True)

    def _smooth_surface(self, grid: pv.DataSet) -> pv.PolyData:
        try:
            surf = grid.extract_surface(algorithm="dataset_surface")
            return surf.smooth(n_iter=10, feature_angle=45.0, boundary_smoothing=False)
        except Exception:
            try:
                return grid.extract_surface(algorithm="dataset_surface")
            except Exception:
                return pv.PolyData()

    def show_hotspots(self, result: Optional[AnalysisResult]):
        for actor in self._hotspot_actors:
            self.remove_actor(actor)
        self._hotspot_actors.clear()
        if self._hotspot_label_actor is not None:
            self.remove_actor(self._hotspot_label_actor)
            self._hotspot_label_actor = None
        if result is None:
            return

        # Part hotspots + riser/feeder hotspots (the latter are thermal reservoirs,
        # not part defects, and must be shown as solved).
        all_hotspots: List[Tuple[HotSpot, bool]] = []
        for hs in result.hotspots:
            all_hotspots.append((hs, False))
        for fhs in result.feeder_hotspots:
            all_hotspots.append((fhs, True))
        if not all_hotspots:
            return

        bbox_min = np.min(result.bbox_size_mm) if result.bbox_size_mm.any() else 100.0
        centers = []
        labels = []
        for hs, is_feeder in all_hotspots:
            # Radius proportional to local wall thickness, clamped to a sensible fraction of the part size.
            radius = max(1.2, min(hs.t_section_mm * 0.15, bbox_min * 0.02, 6.0))
            sphere = pv.Sphere(
                radius=radius,
                center=hs.position_mm,
                theta_resolution=24,
                phi_resolution=24,
            )
            if is_feeder or hs.solved:
                # Solved / feeder-backed hot spot: blue/green sphere.
                color = "#00aaff" if is_feeder else "#00ff88"
                status = "Besleyici Tarafından Çözüldü (Safe)" if is_feeder else "Çözüldü (Safe)"
            else:
                # Unfed or hydraulic/thermal feeding failed -> dangerous.
                color = "#ff0000"
                status = "TEHLİKE: ÇÖZÜLMEDİ!"
            actor = self.add_mesh(
                sphere,
                color=color,
                opacity=0.9,
                ambient=1.0,
                diffuse=0.2,
                lighting=False,
                show_edges=False,
            )
            self._hotspot_actors.append(actor)

            label_pos = hs.position_mm + np.array([0.0, 0.0, radius * 1.4])
            centers.append(label_pos)
            d_str = f"{hs.dist_to_riser_mm:.0f}mm" if np.isfinite(hs.dist_to_riser_mm) else "inf"
            labels.append(
                f"{status}\nM={hs.m_value_mm:.1f}mm | D={d_str} | N={hs.niyama_ensemble:.2f}"
            )

        if centers:
            try:
                self._hotspot_label_actor = self.add_point_labels(
                    np.array(centers),
                    labels,
                    text_color="#ffffff",
                    font_size=11,
                    shape="rounded_rect",
                    background_color="black",
                    background_opacity=1.0,
                    show_points=False,
                    always_visible=True,
                )
            except Exception:
                # Off-screen / OpenGL label rendering can be fragile; spheres alone are enough.
                self._hotspot_label_actor = None

    def show_risk(self, result: Optional[AnalysisResult]):
        """Show risk isosurfaces (0.70 and 0.85) colored by risk value."""
        if self._risk_actor is not None:
            self.remove_actor(self._risk_actor)
            self._risk_actor = None
        self._remove_scalar_bar("Risk")
        if result is None:
            return

        grid = self._make_grid(result, result.risk, "risk")
        part = self._part_only(grid)
        if part.n_cells == 0:
            return
        iso = part.contour([0.70, 0.85], scalars="risk")
        if iso.n_points == 0:
            return
        self._risk_actor = self.add_mesh(
            iso,
            scalars="risk",
            cmap="hot",
            opacity=0.65,
            clim=[0.0, 1.0],
            show_scalar_bar=True,
            scalar_bar_args=_scalar_bar_args("Risk", (0.82, 0.02), clim=[0.0, 1.0]),
            smooth_shading=True,
        )

    def show_porosity_cloud(
        self,
        result: Optional[AnalysisResult],
        noise_percent: float = 100.0,
        max_points: int = 5000,
        pore_size_filter: Optional[str] = None,
    ):
        """Porosity point cloud colored by estimated pore size.

        ``pore_size_filter`` restricts the cloud to ``macro``, ``micro`` or
        ``fine`` size classes; when ``all`` or empty the whole pore-size field
        is shown.  ``noise_percent`` selects the fraction of the highest-risk
        voxels inside the chosen class that are displayed (100 = whole class,
        0 = only the very highest risk).  Class limits come from the alloy.
        """
        if self._porosity_actor is not None:
            self.remove_actor(self._porosity_actor)
            self._porosity_actor = None
        self._remove_scalar_bar("Pore size (µm)")
        if result is None:
            return

        part_mask = result.grid == BodyType.PART
        if not part_mask.any():
            return

        pore_size_filter = (pore_size_filter or "").lower()
        # v10.0: use the final physical pore size (shrinkage + gas/entrainment)
        # for color and class masks; class limits come from the alloy.
        pore_size_um = np.asarray(result.pore_size_um) if result.pore_size_um is not None else np.array([])
        has_pore_size = pore_size_um.size and pore_size_um.shape == part_mask.shape

        class_mask = np.zeros_like(part_mask, dtype=bool)
        use_pore_size = False
        if has_pore_size and pore_size_filter in ("macro", "micro", "fine"):
            class_mask = np.asarray(getattr(result, f"pore_size_{pore_size_filter}_mask", class_mask))
            if class_mask is None or class_mask.size == 0:
                class_mask = np.zeros_like(part_mask, dtype=bool)
            use_pore_size = class_mask.any()
            if not use_pore_size:
                return
        elif has_pore_size and pore_size_filter in ("", "all"):
            class_mask = part_mask & (pore_size_um > 0.0)
            use_pore_size = class_mask.any()

        if use_pore_size:
            field = pore_size_um
            scalar_name = "pore_size_um"
        else:
            # Fallback to the old behaviour if no pore-size field or no filter.
            solid_time = np.asarray(result.solidification_time) if result.solidification_time is not None else np.array([])
            risk = np.asarray(result.risk) if result.risk is not None else np.array([])
            if solid_time.size and np.isfinite(solid_time[part_mask]).any():
                field = solid_time
                scalar_name = "t_solid"
            elif risk.size:
                field = risk
                scalar_name = "risk"
            else:
                return
            class_mask = part_mask

        # Use the risk field to thin very low-probability voxels while still
        # honouring the selected size class.  noise_percent=100 means the whole
        # class is shown; lower values keep only the highest-risk portion.
        risk = np.asarray(result.risk) if result.risk is not None else np.array([])
        has_risk = risk.size and risk.shape == part_mask.shape
        if use_pore_size and has_risk and 0.0 <= noise_percent < 100.0:
            risk_values = risk[class_mask & (risk > 0.0)]
            if risk_values.size == 0:
                return
            p = max(0.0, 100.0 - noise_percent)
            risk_threshold = float(np.percentile(risk_values, p))
            class_mask = class_mask & (risk >= risk_threshold)

        field = np.asarray(field, dtype=np.float64)
        values = field[class_mask]
        finite = np.isfinite(values) & (values > 0.0)
        if not finite.any():
            return
        finite_max = float(np.max(values[finite]))
        finite_min = float(np.min(values[finite]))

        # Keep all shrinkage selected by the risk filter; color by size.
        # Hide zero-value cells so only real positive porosity is shown.
        lo = max(finite_min, 1e-12)
        hi = finite_max
        if hi <= lo:
            return

        # Mask the field to the selected class so threshold only picks from there.
        display_field = np.where(class_mask, field, 0.0)
        clean_field = np.where(np.isfinite(display_field), display_field, hi * 1.5)

        grid = self._make_grid(result, clean_field, scalar_name)
        part = self._part_only(grid)
        if part.n_cells == 0:
            return
        high = part.threshold([lo, hi], scalars=scalar_name)
        if high.n_cells == 0:
            return

        # cell_centers() drops arrays; convert point->cell data first and attach it.
        try:
            high_cells = high.point_data_to_cell_data(pass_point_data=False)
            cloud = high_cells.cell_centers()
            if scalar_name in high_cells.cell_data:
                cloud.point_data[scalar_name] = high_cells.cell_data[scalar_name]
        except Exception:
            cloud = high.cell_centers()

        if cloud.n_points > max_points:
            idx = np.random.choice(cloud.n_points, max_points, replace=False)
            points = cloud.points[idx]
            if scalar_name in cloud.point_data:
                vals = np.asarray(cloud.point_data[scalar_name])[idx]
                cloud = pv.PolyData(points)
                cloud.point_data[scalar_name] = vals
            else:
                cloud = pv.PolyData(points)

        # Porozite noktalarını parça dışına taşanları sil: sadece parça yüzeyi
        # içinde kalan noktaları tut.
        if self._part_mesh_pv is not None and self._part_mesh_pv.n_cells > 0 and cloud.n_points > 0:
            try:
                selected = cloud.select_enclosed_points(
                    self._part_mesh_pv, tolerance=0.001, check_surface=True
                )
                inside = selected["SelectedPoints"].astype(bool)
                if inside.any() and not inside.all():
                    kept_points = cloud.points[inside]
                    kept_cloud = pv.PolyData(kept_points)
                    for name in cloud.array_names:
                        arr = np.asarray(cloud.point_data[name])
                        if arr.shape[0] == cloud.n_points:
                            kept_cloud.point_data[name] = arr[inside]
                    cloud = kept_cloud
                elif not inside.any():
                    # Hiçbir nokta içeride değilse gösterme.
                    return
            except Exception:
                pass

        title = "Pore size (µm)" if scalar_name == "pore_size_um" else ("Solidification time" if scalar_name == "t_solid" else scalar_name)
        self._porosity_actor = self.add_mesh(
            cloud,
            scalars=scalar_name if scalar_name in cloud.array_names else None,
            color="#ff0000" if scalar_name not in cloud.array_names else None,
            cmap="plasma",
            clim=[0.0, max(hi, 1.0)],
            style="points",
            point_size=8,
            render_points_as_spheres=True,
            opacity=1.0,
            lighting=False,
            show_scalar_bar=True,
            scalar_bar_args=_scalar_bar_args(title, (0.82, 0.02), clim=[0.0, max(hi, 1.0)]),
        )

    def show_niyama_isosurfaces(self, result: Optional[AnalysisResult]):
        """Show Niyama isosurfaces (real surfaces) colored by Niyama value inside the part."""
        for actor in self._niyama_actors:
            self.remove_actor(actor)
        self._niyama_actors.clear()
        self._remove_scalar_bar("Niyama")
        if result is None:
            return

        alloy = get_alloy(result.alloy_key)
        grid = self._make_grid(result, result.niyama, "niyama")
        part = self._part_only(grid)
        if part.n_cells == 0:
            return

        iso = part.contour(
            [alloy.niyama_macro, alloy.niyama_shrinkage],
            scalars="niyama",
        )
        if iso.n_points == 0:
            return

        actor = self.add_mesh(
            iso,
            scalars="niyama",
            cmap="jet",
            opacity=0.8,
            clim=[0.0, alloy.niyama_shrinkage * 2.0],
            show_scalar_bar=True,
            scalar_bar_args=_scalar_bar_args("Niyama", (0.82, 0.16), clim=[0.0, alloy.niyama_shrinkage * 2.0]),
            smooth_shading=True,
        )
        self._niyama_actors.append(actor)

    def show_flow_velocity(self, result: Optional[AnalysisResult]):
        """Overlay the 3-D Darcy flow-velocity magnitude on the metal surfaces."""
        if self._flow_actor is not None:
            self.remove_actor(self._flow_actor)
            self._flow_actor = None
        self._remove_scalar_bar("Akış hızı (m/s)")
        if result is None or result.flow_result is None:
            return
        fr = result.flow_result
        vmag = fr.velocity_magnitude
        if vmag is None or vmag.size == 0:
            return
        grid = self._make_grid(result, vmag, "velocity_magnitude")
        metal = self._metal_only(grid)
        if metal.n_cells == 0:
            return
        surf = self._smooth_surface(metal)
        if surf.n_points == 0:
            return
        vmax = float(np.nanmax(vmag)) if np.isfinite(vmag).any() else 1.0
        self._flow_actor = self.add_mesh(
            surf,
            scalars="velocity_magnitude",
            cmap="turbo",
            opacity=1.0,
            clim=[0.0, max(vmax, 1e-3)],
            show_scalar_bar=True,
            scalar_bar_args=_scalar_bar_args("Akış hızı (m/s)", (0.64, 0.02), clim=[0.0, max(vmax, 1e-3)]),
            smooth_shading=True,
        )

    def show_flow_node_labels(self, result: Optional[AnalysisResult]):
        """Add gating arrows and only ingate-inlet velocity labels."""
        if self._flow_node_actor is not None:
            self.remove_actor(self._flow_node_actor)
            self._flow_node_actor = None
        if self._flow_arrow_actor is not None:
            self.remove_actor(self._flow_arrow_actor)
            self._flow_arrow_actor = None
        if result is None or result.flow_result is None:
            return
        nodes = result.flow_result.gating_nodes
        if not nodes:
            return

        # Voxel data needed to trace flow lines through body interiors.
        bodies = getattr(self, "_bodies", [])
        body_index = getattr(self, "_body_index", None)
        origin_mm = getattr(self, "_origin_mm", None)
        dx_mm = float(getattr(self, "_dx_mm", 0.0) or 0.0)
        body_idx_by_name = {
            b.name: int(b.index)
            for b in bodies
            if getattr(b, "index", None) is not None
        }

        # ---- one continuous colored line per ingate from inlet to ingate-part entry ----
        node_by_downstream: Dict[str, GatingNode] = {}
        for node in nodes:
            if "→" not in node.name:
                continue
            down_name = node.name.split("→")[1].strip()
            if down_name not in node_by_downstream:
                node_by_downstream[down_name] = node

        points: List[np.ndarray] = []
        lines: List[int] = []
        cell_velocities: List[float] = []
        label_points: List[Tuple[float, float, float]] = []
        label_texts: List[str] = []

        def _node_velocity(node: GatingNode) -> float:
            return node.max_velocity_m_s if node.max_velocity_m_s > 1e-12 else node.velocity_m_s

        for node in nodes:
            if "→" not in node.name or "→" not in node.body_type:
                continue
            down_type = node.body_type.split("→")[1].strip()
            if down_type != "PART":
                continue
            # Trace from this ingate->part node back to the source.
            path: List[GatingNode] = [node]
            current_up = node.name.split("→")[0].strip()
            while current_up in node_by_downstream:
                pred = node_by_downstream[current_up]
                path.append(pred)
                current_up = pred.name.split("→")[0].strip()
                if current_up == "Kaynak" or pred.body_type.split("→")[0].strip() == "SOURCE":
                    break
            path.reverse()

            if len(path) < 2:
                continue

            # Build segment points through each body so the line follows the
            # real gate interior instead of a straight chord cutting the part.
            path_points: List[np.ndarray] = []
            for i in range(len(path) - 1):
                n0 = path[i]
                n1 = path[i + 1]
                body_name = n0.name.split("→")[1].strip()
                segment = [np.asarray(n0.centroid_mm, dtype=np.float64)]
                if (
                    body_index is not None
                    and body_name in body_idx_by_name
                    and dx_mm > 0.0
                    and origin_mm is not None
                ):
                    try:
                        pts = _voxel_path_through_body(
                            body_index,
                            body_idx_by_name[body_name],
                            origin_mm,
                            dx_mm,
                            np.asarray(n0.centroid_mm, dtype=np.float64),
                            np.asarray(n1.centroid_mm, dtype=np.float64),
                            step=2,
                        )
                        if pts is not None and len(pts) >= 2:
                            segment = list(pts)
                    except Exception:
                        pass
                if i == 0:
                    path_points.extend(segment)
                else:
                    path_points.extend(segment[1:])

            start_idx = len(points)
            n_pts = len(path_points)
            points.extend(path_points)
            lines.extend([n_pts] + list(range(start_idx, start_idx + n_pts)))

            end_v = _node_velocity(node)
            cell_velocities.append(end_v)

            # label only at the ingate->part entry
            if end_v > 1e-12:
                label_points.append(node.centroid_mm)
                label_texts.append(f"{end_v:.2f} m/s")

        if points:
            poly = pv.PolyData(
                np.asarray(points, dtype=np.float64),
                lines=np.asarray(lines, dtype=np.int64),
            )
            poly.cell_data["velocity_m_s"] = np.asarray(cell_velocities, dtype=float)
            self._flow_arrow_actor = self.add_mesh(
                poly,
                scalars="velocity_m_s",
                cmap="turbo",
                line_width=5,
                opacity=0.95,
                show_scalar_bar=True,
                scalar_bar_args=_scalar_bar_args("Akış hızı (m/s)", (0.02, 0.02)),
                lighting=False,
            )

        if label_points:
            label_points = np.asarray(label_points, dtype=np.float64)
            self._flow_node_actor = self.add_point_labels(
                label_points,
                label_texts,
                font_size=10,
                text_color="white",
                point_color="red",
                point_size=12,
                shape=None,
                always_visible=True,
                shadow=False,
                name="flow_node_labels",
            )

    def show_feeding_paths(self, result: Optional[AnalysisResult]):
        for actor in self._path_actors:
            self.remove_actor(actor)
        self._path_actors.clear()
        if result is None or result.dist_to_riser.size == 0:
            return

        part_mask = result.grid == BodyType.PART
        for hs in result.hotspots:
            if hs.solved:
                continue
            vox = np.round((hs.position_mm - result.origin_mm) / result.dx_mm).astype(int)
            if not (
                0 <= vox[0] < part_mask.shape[0]
                and 0 <= vox[1] < part_mask.shape[1]
                and 0 <= vox[2] < part_mask.shape[2]
            ):
                continue
            if not part_mask[vox[0], vox[1], vox[2]]:
                continue
            path = _trace_path_to_riser(result.dist_to_riser, part_mask, vox)
            if len(path) < 2:
                continue
            pts = np.array(path) * result.dx_mm + result.origin_mm
            poly = pv.PolyData()
            poly.points = pts
            poly.lines = np.hstack([[len(pts)], np.arange(len(pts))]).astype(np.int64)
            radius = max(2.0, result.dx_mm * 2.0)
            try:
                tube = poly.tube(radius=radius)
            except Exception:
                tube = poly
            # Cyan tubes are visible against red/yellow risk surfaces.
            color = "#00ff88" if hs.feed_ok else "#00ffff"
            actor = self.add_mesh(
                tube,
                color=color,
                opacity=0.9,
                smooth_shading=True,
                lighting=False,
                show_scalar_bar=False,
            )
            self._path_actors.append(actor)

    def show_slices(self, result: Optional[AnalysisResult], field: str = "sdf"):
        """Add three orthogonal slices through the part for the selected scalar field."""
        for actor in self._slice_actors:
            self.remove_actor(actor)
        self._slice_actors.clear()
        if result is None:
            return

        field_map = {
            "sdf": (result.sdf, "SDF (mm)", "viridis"),
            "risk": (result.risk, "Risk", "hot"),
            "niyama": (result.niyama, "Niyama", "plasma"),
            "mat_id": (result.grid.astype(np.float64), "Mat ID", "tab10"),
            "temperature": (
                result.temperature if result.temperature.size > 0 else result.sdf,
                "T (°C)",
                "coolwarm",
            ),
        }
        if field not in field_map:
            return
        data, title, cmap = field_map[field]

        finite_data = data[np.isfinite(data)]
        if finite_data.size == 0:
            return
        dmin = float(finite_data.min())
        dmax = float(finite_data.max())
        if field == "risk":
            slice_clim = (0.0, 1.0)
        elif field == "mat_id":
            slice_clim = (dmin, dmax)
        elif field == "niyama":
            slice_clim = (max(0.0, dmin), dmax)
        else:
            slice_clim = (dmin, dmax)

        grid = self._make_grid(result, data, field)
        domain = self._part_only(grid) if field in ("risk", "niyama") else self._metal_only(grid)
        if domain.n_cells == 0:
            return

        self._remove_scalar_bar(title)
        origin = np.array(domain.center)
        first_bar = True
        for normal in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            slc = domain.slice(normal=normal, origin=origin)
            if slc.n_points == 0:
                continue
            actor = self.add_mesh(
                slc,
                scalars=field,
                cmap=cmap,
                opacity=0.95,
                show_scalar_bar=first_bar,
                scalar_bar_args=_scalar_bar_args(title, (0.82, 0.18), clim=slice_clim),
                clim=slice_clim,
            )
            self._slice_actors.append(actor)
            first_bar = False

    def show_local_regions(self, result: Optional[AnalysisResult], field: str = "risk"):
        for actor in self._local_actors:
            self.remove_actor(actor)
        self._local_actors.clear()
        if result is None:
            return

        for region in result.local_regions:
            if region.grid.size == 0:
                continue
            grid = pv.ImageData()
            grid.dimensions = np.array(region.grid.shape) + 1
            grid.origin = region.origin_mm
            grid.spacing = (region.dx_mm, region.dx_mm, region.dx_mm)
            data = {
                "sdf": region.sdf,
                "risk": region.risk,
                "niyama": region.niyama,
            }.get(field, region.risk)
            grid.cell_data[field] = data.flatten(order="F")
            actor = self.add_mesh(
                grid,
                scalars=field,
                cmap="hot" if field == "risk" else "viridis",
                opacity=0.35,
                show_scalar_bar=False,
            )
            self._local_actors.append(actor)

    # ---------------- toggles ----------------
    def toggle_risk(self, result: AnalysisResult, checked: bool):
        if checked:
            self.show_risk(result)
        else:
            if self._risk_actor is not None:
                self.remove_actor(self._risk_actor)
                self._risk_actor = None
            self._remove_scalar_bar("Risk")

    def toggle_hotspots(self, result: AnalysisResult, checked: bool):
        if checked:
            self.show_hotspots(result)
        else:
            for actor in self._hotspot_actors:
                self.remove_actor(actor)
            self._hotspot_actors.clear()
            if self._hotspot_label_actor is not None:
                self.remove_actor(self._hotspot_label_actor)
                self._hotspot_label_actor = None

    def toggle_porosity(self, result: AnalysisResult, checked: bool, noise_percent: float = 3.0, max_points: int = 5000, pore_size_filter: Optional[str] = None):
        if checked:
            self.show_porosity_cloud(result, noise_percent=noise_percent, max_points=max_points, pore_size_filter=pore_size_filter)
        else:
            if self._porosity_actor is not None:
                self.remove_actor(self._porosity_actor)
                self._porosity_actor = None
            self._remove_scalar_bar("Pore size (µm)")

    def toggle_niyama(self, result: AnalysisResult, checked: bool):
        if checked:
            self.show_niyama_isosurfaces(result)
        else:
            for actor in self._niyama_actors:
                self.remove_actor(actor)
            self._niyama_actors.clear()
            self._remove_scalar_bar("Niyama")

    def toggle_flow_velocity(self, result: AnalysisResult, checked: bool):
        if checked:
            self.show_flow_velocity(result)
        else:
            if self._flow_actor is not None:
                self.remove_actor(self._flow_actor)
                self._flow_actor = None
            self._remove_scalar_bar("Akış hızı (m/s)")

    def toggle_flow_node_labels(self, result: AnalysisResult, checked: bool):
        if checked:
            self.show_flow_node_labels(result)
        else:
            if self._flow_node_actor is not None:
                self.remove_actor(self._flow_node_actor)
                self._flow_node_actor = None
            if self._flow_arrow_actor is not None:
                self.remove_actor(self._flow_arrow_actor)
                self._flow_arrow_actor = None

    def toggle_flow_animation(self, result: AnalysisResult, checked: bool):
        if checked:
            self.flow_animator.set_result(result)
        else:
            self.flow_animator.stop()

    def toggle_feeding_paths(self, result: AnalysisResult, checked: bool):
        if checked:
            self.show_feeding_paths(result)
        else:
            for actor in self._path_actors:
                self.remove_actor(actor)
            self._path_actors.clear()

    def toggle_slices(self, result: AnalysisResult, checked: bool, field: str = "sdf"):
        if checked:
            self.show_slices(result, field)
        else:
            for actor in self._slice_actors:
                self.remove_actor(actor)
            self._slice_actors.clear()
            # Remove any slice scalar bar to avoid overlap when switching fields.
            for title in ["SDF (mm)", "Risk", "Niyama", "Mat ID", "T (°C)"]:
                self._remove_scalar_bar(title)

    # ---------------- gating cross-section picker ----------------
    def start_section_picker(
        self,
        section_key: str,
        bodies: List[Body],
        callback: Callable[[str, float, str], None],
    ):
        """Let the user click on a gating body to measure its cross-sectional area.

        The cut plane is perpendicular to the principal axis that best aligns
        with the clicked face normal, so the measured area corresponds to the
        flow cross-section when the user clicks on an end face.
        """
        self._clear_section_actors()
        self._section_callback = callback
        self._section_bodies = bodies
        self._section_key = section_key

        def _on_pick(point):
            self._handle_section_pick(np.asarray(point, dtype=np.float64))

        # Make sure a previous point-picking session is fully disabled before
        # starting a new one, otherwise PyVista raises "Picking is already enabled".
        try:
            self.disable_picking()
        except Exception:
            pass
        self._section_picker = None

        print(f"[section picker] '{section_key}' kesiti için 3D görünümde ilgili yüzeye tıklayın.")
        self._section_picker = self.enable_point_picking(
            _on_pick,
            left_clicking=True,
            picker="cell",
            show_message=False,
            color="#ff00ff",
            point_size=12,
        )

    def _handle_section_pick(self, point: np.ndarray):
        try:
            body = self._find_body_at_point(point)
            if body is None:
                print("[section picker] Tıklanan nokta herhangi bir body yüzeyine yakın değil.")
                return

            # Use the body's natural flow axis and the robust cross-section
            # estimator so a single click on any face gives the characteristic
            # runner/ingate/sprue area, not a one-off slice.
            axis = _flow_axis(body.mesh)

            if self._section_key in ("SPRUE_BASE", "SPRUE_THROAT"):
                base_mm2, throat_mm2 = _sprue_circular_base_and_throat(
                    body.mesh, axis
                )
                area_mm2 = base_mm2 if self._section_key == "SPRUE_BASE" else throat_mm2
            else:
                area_mm2 = _characteristic_cross_section_area(body.mesh, axis)

            area_cm2 = area_mm2 / 100.0

            # Visualise a representative section through the body centroid.
            centroid = np.asarray(body.mesh.centroid, dtype=np.float64)
            section = body.mesh.section(plane_origin=centroid, plane_normal=axis)
            if section is not None and len(section.vertices) >= 3:
                self._show_section_outline(centroid, axis, body, section)
            print(f"[section picker] {body.name} ({self._section_key}): A = {area_cm2:.3f} cm²")

            if self._section_callback is not None:
                self._section_callback(self._section_key, area_cm2, body.name)
        except Exception as e:
            print(f"[section picker] Kesit ölçüm hatası: {e}")
        finally:
            self.disable_picking()
            self._section_picker = None

    def _find_body_at_point(self, point: np.ndarray) -> Optional[Body]:
        import trimesh

        best_body = None
        best_dist = float("inf")
        for body in self._section_bodies:
            if len(body.faces) == 0:
                continue
            try:
                closest, dist, _ = trimesh.proximity.closest_point(
                    body.mesh, np.array([point])
                )
                dist = float(dist[0])
                if dist < best_dist:
                    best_dist = dist
                    best_body = body
            except Exception:
                continue
        return best_body

    def _show_section_outline(
        self,
        point: np.ndarray,
        axis: np.ndarray,
        body: Body,
        section,
    ):
        """Visualise the picked point, cutting plane and section outline."""
        # Picked point marker
        marker = pv.PolyData(point)
        actor = self.add_mesh(
            marker,
            color="#ff00ff",
            style="points",
            point_size=14,
            render_points_as_spheres=True,
            pickable=False,
        )
        self._section_actors.append(actor)

        # Section vertices as a point cloud / outline
        pts = np.asarray(section.vertices, dtype=np.float64)
        if len(pts) >= 3:
            poly = pv.PolyData(pts)
            actor = self.add_mesh(
                poly,
                color="#ffff00",
                style="points",
                point_size=8,
                render_points_as_spheres=True,
                pickable=False,
            )
            self._section_actors.append(actor)

        # Transparent cutting plane sized to the body bounds
        bounds = body.mesh.bounds
        diag = float(np.linalg.norm(bounds[1] - bounds[0]))
        if diag <= 0:
            diag = 50.0
        plane = pv.Plane(
            center=point,
            direction=axis,
            i_size=diag,
            j_size=diag,
        )
        actor = self.add_mesh(
            plane,
            color="#00ffff",
            opacity=0.15,
            pickable=False,
        )
        self._section_actors.append(actor)
        self.render()

    def _clear_section_actors(self):
        for actor in getattr(self, "_section_actors", []):
            try:
                self.remove_actor(actor)
            except Exception:
                pass
        self._section_actors = []

    def save_screenshot(self, path: str) -> str:
        """Save a PNG screenshot of the current 3D view."""
        super().screenshot(path, transparent_background=False)
        return path
