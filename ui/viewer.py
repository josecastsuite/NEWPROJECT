"""PyVistaQt 3D viewer wrapper for JoseCast Analyzer v8.x."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
from PyQt6 import QtCore, QtWidgets
from pyvistaqt import QtInteractor
from scipy import ndimage
from scipy.spatial import cKDTree

from core.gating import (
    _characteristic_cross_section_area,
    _flow_axis,
    _section_2d_area_and_perim,
    _sprue_circular_base_and_throat,
)
from core.materials import get_alloy
from core.sdf_analyzer import _trace_path_to_riser
from core.types import BODY_TYPE_LABELS, AnalysisResult, Body, BodyType, HotSpot, RefinementRegion
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
    # Estimate text lengths so the bar is wide enough for the title and the
    # numeric labels without overlap.
    label_chars = 0
    if clim is not None:
        spec = fmt[1:]  # e.g. ".2f"
        label_chars = max(len(f"{clim[0]:{spec}}"), len(f"{clim[1]:{spec}}"))
    title_chars = len(title)
    # Keep height fixed, make width dynamic based on title and desired labels.
    height = 0.08 * 1.5
    n_labels = 2
    for n in (5, 4, 3, 2):
        needed = title_chars * 0.014 + 0.05 + n * (max(label_chars, 1) * 0.013 + 0.010)
        if needed <= 0.85:
            n_labels = n
            break
    width = min(0.85, max(0.25, needed))
    # Center the bar horizontally near the requested bottom position.
    pos_x = max(0.0, 0.5 - width / 2.0)
    args = {
        "color": "#334155",
        "title_font_size": 10,
        "label_font_size": 8,
        "fmt": fmt,
        "n_labels": n_labels,
        "vertical": False,
        "position_x": pos_x,
        "position_y": pos[1],
        "width": width,
        "height": height,
        "title": title,
    }
    return args


class Analyzer3DViewer(QtInteractor):
    """Extended PyVistaQt interactor for casting analysis."""

    def __init__(self, parent=None, off_screen: bool = False):
        super().__init__(parent=parent, off_screen=off_screen)
        self.set_background("#F8FAFC", top="#E2E8F0")
        self.add_axes(line_width=2, color="#64748B")
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
        self._flow_colorbar_actor = None
        self._mold_wall_actor = None
        self._cold_shot_actor = None
        self._cold_shot_scalar_bar_actor = None
        self._last_fill_actor = None
        self._erosion_actor = None
        self._air_entrapment_actor = None
        self._air_entrapment_marker_actor = None
        self.flow_animator = FlowAnimator(self)
        self._body_legend_actors: List[Any] = []
        self._body_legend_frame = QtWidgets.QFrame(self)
        self._body_legend_frame.setObjectName("bodyLegend")
        self._body_legend_frame.setStyleSheet(
            "QFrame#bodyLegend {"
            "  background-color: rgba(248, 250, 252, 0.95);"
            "  border: 1px solid #94A3B8;"
            "  border-radius: 6px;"
            "  padding: 6px;"
            "}"
            "QLabel { background: transparent; border: none; }"
        )
        self._body_legend_frame.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        legend_layout = QtWidgets.QVBoxLayout(self._body_legend_frame)
        legend_layout.setContentsMargins(6, 6, 6, 6)
        legend_layout.setSpacing(4)
        self._body_legend_frame.setLayout(legend_layout)
        self._body_legend_frame.hide()
        self._bodies: List[Body] = []
        self._body_index: Optional[np.ndarray] = None
        self._origin_mm: Optional[np.ndarray] = None
        self._dx_mm: float = 0.0

    def _remove_body_legend(self) -> None:
        """Clear the Qt-based body legend overlay."""
        layout = self._body_legend_frame.layout()
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._body_legend_frame.hide()

    def _position_body_legend(self) -> None:
        """Keep the legend frame anchored to the top-right corner."""
        margin = 10
        self._body_legend_frame.adjustSize()
        self._body_legend_frame.move(
            max(margin, self.width() - self._body_legend_frame.width() - margin),
            margin,
        )

    def _update_body_legend(self, bodies: List[Body]) -> None:
        """Add a top-right Qt legend with fixed bold text and aligned bullets."""
        self._remove_body_legend()

        present = sorted(
            {body.body_type for body in bodies if len(body.faces) > 0},
            key=lambda x: int(x),
        )
        if not present:
            return

        layout = self._body_legend_frame.layout()
        for bt in present:
            color = BODY_COLORS.get(bt, "#F5F5F5")
            label = BODY_LEGEND_LABELS.get(bt, str(bt))

            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)

            bullet = QtWidgets.QLabel()
            bullet.setFixedSize(10, 10)
            bullet.setStyleSheet(
                f"background-color: {color}; border-radius: 5px; border: 1.5px solid #000000;"
            )
            row_layout.addWidget(bullet)

            text = QtWidgets.QLabel(label)
            text.setStyleSheet(
                "font-size: 10pt; font-weight: bold; color: #334155;"
                "background: transparent; border: none;"
            )
            row_layout.addWidget(text, alignment=QtCore.Qt.AlignmentFlag.AlignVCenter)
            row_layout.addStretch()

            layout.addWidget(row)

        self._body_legend_frame.show()
        self._position_body_legend()

    def resizeEvent(self, event: QtCore.QEvent) -> None:
        """Keep the body legend in the top-right corner on resize."""
        super().resizeEvent(event)
        if self._body_legend_frame.isVisible():
            self._position_body_legend()

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
        self._remove_all_scalar_bars()
        self.clear_actors()
        self.add_axes(line_width=2, color="#64748B")
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
        self._flow_colorbar_actor = None
        self._mold_wall_actor = None
        self._cold_shot_actor = None
        self._cold_shot_scalar_bar_actor = None
        self._last_fill_actor = None
        self._erosion_actor = None
        self._air_entrapment_actor = None
        self._air_entrapment_marker_actor = None
        self._body_legend_actors = []
        self._remove_body_legend()
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
            is_selected = selected_body is not None and (
                body is selected_body or body.name == selected_body.name
            )
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
                ambient=0.55,
                diffuse=0.45,
                specular=0.05,
                specular_power=1,
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

    def _remove_all_scalar_bars(self):
        for title in list(self.scalar_bars.keys()):
            self._remove_scalar_bar(title)

    def _arrange_scalar_bars(self):
        """Stack active scalar bars vertically at the bottom-left to avoid overlap."""
        try:
            y = 0.02
            for _, actor in self.scalar_bars.items():
                w, h = actor.GetPosition2()
                actor.SetPosition(0.02, y)
                y += h + 0.02
        except Exception:
            pass

    def _make_grid(self, result: AnalysisResult, scalars: np.ndarray, name: str) -> pv.ImageData:
        """Build a PyVista ImageData (voxel grid) with point-centered scalars and masks."""
        grid = pv.ImageData()
        grid.dimensions = np.array(result.grid.shape) + 1
        grid.origin = result.origin_mm
        grid.spacing = (result.dx_mm, result.dx_mm, result.dx_mm)
        grid.cell_data[name] = np.asarray(scalars).ravel(order="F")
        grid.cell_data["is_metal"] = result.is_metal.ravel(order="F").astype(np.float64)
        grid.cell_data["part"] = (result.grid == BodyType.PART).ravel(order="F").astype(np.float64)
        gate_mask = np.isin(
            result.grid,
            [
                BodyType.SPRUE_THROAT,
                BodyType.SPRUE,
                BodyType.RUNNER,
                BodyType.DISTRIBUTOR,
                BodyType.INGATE,
                BodyType.POURING_BASIN,
                BodyType.COOLING_SPRUE,
                BodyType.FILTER,
            ],
        )
        grid.cell_data["is_gate"] = gate_mask.ravel(order="F").astype(np.float64)
        # Contour / slice filters require point data; convert and keep both scalars.
        return grid.cell_data_to_point_data()

    def _metal_only(self, grid: pv.ImageData) -> pv.UnstructuredGrid:
        """Return only fully-metal cells from a grid."""
        return grid.threshold([1.0, 1.0], scalars="is_metal", all_scalars=True)

    def _gate_only(self, grid: pv.ImageData) -> pv.UnstructuredGrid:
        """Return only gating-system cells (sprue/runner/distributor/ingate)."""
        return grid.threshold([1.0, 1.0], scalars="is_gate", all_scalars=True)

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
                # Solved / feeder-backed hot spot: blue sphere.
                color = "#2563EB" if is_feeder else "#3B82F6"
                status = "Besleyici Tarafından Çözüldü (Safe)" if is_feeder else "Çözüldü (Safe)"
            else:
                # Unfed or hydraulic/thermal feeding failed -> dangerous.
                color = "#EF4444"
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
                    text_color="#334155",
                    font_size=11,
                    shape="rounded_rect",
                    background_color="#F1F5F9",
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
            scalar_bar_args=_scalar_bar_args("Risk", (0.02, 0.02), clim=[0.0, 1.0]),
            smooth_shading=True,
        )
        self._arrange_scalar_bars()

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
            scalar_bar_args=_scalar_bar_args(title, (0.02, 0.02), clim=[0.0, max(hi, 1.0)]),
        )
        self._arrange_scalar_bars()

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
            scalar_bar_args=_scalar_bar_args("Niyama", (0.02, 0.02), clim=[0.0, alloy.niyama_shrinkage * 2.0]),
            smooth_shading=True,
        )
        self._niyama_actors.append(actor)
        self._arrange_scalar_bars()

    @staticmethod
    def _build_gating_branches(nodes: List) -> List[List]:
        """Return every source -> ingate path as a list of gating nodes."""
        if not nodes:
            return []
        down_to_node: Dict[str, Any] = {}
        source_node: Optional[Any] = None
        for n in nodes:
            name = getattr(n, "name", "")
            if " → " not in name:
                continue
            up, down = name.split(" → ", 1)
            if up == "Kaynak":
                source_node = n
            down_to_node[down] = n
        if source_node is None:
            source_node = nodes[0]
        up_names = {n.name.split(" → ", 1)[0] for n in nodes if " → " in getattr(n, "name", "")}
        leaves = [
            n
            for n in nodes
            if n is not source_node and " → " in getattr(n, "name", "") and n.name.split(" → ", 1)[1] not in up_names
        ]
        branches: List[List] = []
        for leaf in leaves:
            path = [leaf]
            while True:
                up = path[0].name.split(" → ", 1)[0]
                if up == "Kaynak" or up not in down_to_node:
                    break
                path.insert(0, down_to_node[up])
            branches.append(path)
        if not branches:
            branches = [[source_node] + [n for n in nodes if n is not source_node]]
        return branches

    def _flow_velocity_from_nodes(
        self, result: AnalysisResult, gate_mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """Fallback: assign each gate voxel the throat velocity v=Q/A of the
        nearest gating-node edge.  This works even when the raw Darcy field is
        uniform, so section/branch changes still show up as colour changes.
        """
        fr = result.flow_result
        nodes = getattr(fr, "gating_nodes", None) or []
        if not nodes:
            return None
        branches = self._build_gating_branches(nodes)
        if not branches:
            return None

        edges: List[Tuple[Any, Any]] = []
        seen: set = set()
        for branch in branches:
            for up, down in zip(branch[:-1], branch[1:]):
                key = (up.name, down.name)
                if key in seen:
                    continue
                seen.add(key)
                edges.append((up, down))
        if not edges:
            return None

        dx = float(result.dx_mm)
        origin = np.asarray(result.origin_mm, dtype=np.float64)

        ref_points: List[np.ndarray] = []
        ref_vel: List[float] = []
        for up, down in edges:
            p0 = np.asarray(up.centroid_mm, dtype=np.float64)
            p1 = np.asarray(down.centroid_mm, dtype=np.float64)
            dist = float(np.linalg.norm(p1 - p0))
            n_pts = max(2, int(np.ceil(dist / (dx * 0.5))) + 1)
            pts = np.linspace(p0, p1, n_pts)
            v = float(down.velocity_m_s) if down.velocity_m_s > 1e-12 else float(up.velocity_m_s)
            if v <= 1e-12:
                continue
            for p in pts:
                ref_points.append(p)
                ref_vel.append(v)

        if not ref_points:
            return None
        ref_points_arr = np.asarray(ref_points, dtype=np.float64)
        ref_vel_arr = np.asarray(ref_vel, dtype=np.float64)

        tree = cKDTree(ref_points_arr)
        indices = np.argwhere(gate_mask)
        if indices.size == 0:
            return None
        centers = (indices + 0.5) * dx + origin
        try:
            _, nearest = tree.query(centers, k=1, workers=-1)
        except TypeError:
            _, nearest = tree.query(centers, k=1)
        nearest = np.asarray(nearest, dtype=np.int64)

        vel_arr = np.zeros(result.grid.shape, dtype=np.float64)
        vel_arr[tuple(indices.T)] = ref_vel_arr[nearest]
        return vel_arr

    def show_flow_node_labels(self, result: Optional[AnalysisResult]):
        """Add fixed numeric velocity labels at the sprue and ingate nodes.

        Velocity is the simple hydraulic estimate v = Q / A already stored on
        each gating node; no 3-D colour field is rendered.
        """
        if self._flow_node_actor is not None:
            self.remove_actor(self._flow_node_actor)
            self._flow_node_actor = None
        if result is None or result.flow_result is None:
            return

        nodes = result.flow_result.gating_nodes
        label_points: List[Tuple[float, float, float]] = []
        label_texts: List[str] = []
        gate_type_names = {
            BodyType.INGATE.name,
            BodyType.SPRUE_THROAT.name,
            BodyType.SPRUE.name,
            BodyType.RUNNER.name,
            BodyType.DISTRIBUTOR.name,
            BodyType.CURUFLUK.name,
            BodyType.POURING_BASIN.name,
            BodyType.COOLING_SPRUE.name,
            BodyType.FILTER.name,
        }

        for node in nodes:
            body_type = getattr(node, "body_type", "")
            node_name = getattr(node, "name", "")
            if not body_type or "→" not in body_type or not node_name or "→" not in node_name:
                continue
            up_type, down_type = [s.strip() for s in body_type.split("→", 1)]
            _, down_name = [s.strip() for s in node_name.split("→", 1)]
            # Label the downstream gate element (meme, sprue, runner, etc.).
            if down_type not in gate_type_names and up_type not in gate_type_names:
                continue

            area_cm2 = float(getattr(node, "section_area_cm2", 0.0) or 0.0)
            q_m3_s = float(getattr(node, "flow_rate_m3_s", 0.0) or 0.0)
            if area_cm2 > 1e-9 and q_m3_s > 1e-12:
                v = q_m3_s / (area_cm2 * 1e-4)
            else:
                v = float(getattr(node, "velocity_m_s", 0.0) or 0.0)
            if v <= 1e-12:
                continue

            label_points.append(getattr(node, "centroid_mm", (0.0, 0.0, 0.0)))
            label_texts.append(f"{v:.2f} m/s")

        if label_points:
            label_points = np.asarray(label_points, dtype=np.float64)
            self._flow_node_actor = self.add_point_labels(
                label_points,
                label_texts,
                font_size=10,
                text_color="#334155",
                point_color="#EF4444",
                point_size=12,
                shape="rounded_rect",
                background_color="#FFFFFF",
                background_opacity=0.9,
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
            # Blue/gray tubes are visible against risk surfaces.
            color = "#3B82F6" if hs.feed_ok else "#64748B"
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
                scalar_bar_args=_scalar_bar_args(title, (0.02, 0.02), clim=slice_clim),
                clim=slice_clim,
            )
            self._slice_actors.append(actor)
            first_bar = False
        self._arrange_scalar_bars()

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

    def show_mold_wall_movement(self, result: Optional[AnalysisResult]):
        """Heatmap of regions where graphite expansion pushes the mould wall.

        The movement percentage is positive where the graphite expansion force is
        not fully absorbed by the mould rigidity (i.e. soft green-sand moulds).
        """
        if self._mold_wall_actor is not None:
            self.remove_actor(self._mold_wall_actor)
            self._mold_wall_actor = None
        self._remove_scalar_bar("Kalıp şişmesi (%)")
        if result is None or result.mold_wall_movement is None or result.mold_wall_movement.size == 0:
            return

        grid = self._make_grid(result, result.mold_wall_movement, "mold_wall_movement")
        part = self._part_only(grid)
        if part.n_cells == 0:
            return
        cells = part.threshold(0.05, scalars="mold_wall_movement", all_scalars=True)
        if cells.n_cells == 0:
            return
        vmax = float(np.percentile(cells["mold_wall_movement"], 98))
        if vmax <= 0.05:
            vmax = 0.1
        clim = [0.0, vmax]
        self._mold_wall_actor = self.add_mesh(
            cells,
            scalars="mold_wall_movement",
            cmap="hot",
            opacity=0.75,
            clim=clim,
            show_scalar_bar=True,
            scalar_bar_args=_scalar_bar_args("Kalıp şişmesi (%)", (0.02, 0.02), clim=clim),
            smooth_shading=True,
        )
        self._arrange_scalar_bars()

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

    def toggle_flow_animation(self, result: AnalysisResult, checked: bool):
        if checked:
            self.flow_animator.set_result(result)
        else:
            self.flow_animator.stop()

    def toggle_mold_wall_movement(self, result: AnalysisResult, checked: bool):
        if checked:
            self.show_mold_wall_movement(result)
        else:
            if self._mold_wall_actor is not None:
                self.remove_actor(self._mold_wall_actor)
                self._mold_wall_actor = None
            self._remove_scalar_bar("Kalıp şişmesi (%)")

    def show_cold_shot_risk(self, result: Optional[AnalysisResult]):
        """Heatmap of cold-shut (soğuk birleşme) risk on the part surface.

        Risk values below 0.3 are made transparent.  Values between 0.3 and 1.0
        are shown with a yellow-orange-red heatmap.  The latest-filled voxel is
        marked with a red sphere so the operator sees where air/oxide is most
        likely trapped.
        """
        if self._cold_shot_actor is not None:
            self.remove_actor(self._cold_shot_actor)
            self._cold_shot_actor = None
        if self._last_fill_actor is not None:
            self.remove_actor(self._last_fill_actor)
            self._last_fill_actor = None
        self._remove_scalar_bar("Soğuk birleşme riski")

        if result is None or result.cold_shot_risk is None or result.cold_shot_risk.size == 0:
            return

        grid = self._make_grid(result, result.cold_shot_risk, "cold_shot_risk")
        part = self._part_only(grid)
        if part.n_cells == 0:
            return

        # Threshold: only show risk >= 0.3, per protocol.
        cells = part.threshold(0.3, scalars="cold_shot_risk", all_scalars=True)
        if cells.n_cells == 0:
            return

        vmax = float(np.percentile(cells["cold_shot_risk"], 99))
        if vmax <= 0.3:
            vmax = 1.0
        clim = [0.3, vmax]

        self._cold_shot_actor = self.add_mesh(
            cells,
            scalars="cold_shot_risk",
            cmap="YlOrRd",
            opacity=0.85,
            clim=clim,
            show_scalar_bar=True,
            scalar_bar_args=_scalar_bar_args("Soğuk birleşme riski", (0.02, 0.02), clim=clim),
            smooth_shading=True,
        )

        # Last-fill point marker: red sphere at the latest-filled voxel.
        if result.last_fill_point_mm is not None and result.last_fill_point_mm.size == 3:
            radius = max(float(result.dx_mm) * 2.0, 2.0)
            sphere = pv.Sphere(radius=radius, center=result.last_fill_point_mm)
            self._last_fill_actor = self.add_mesh(
                sphere,
                color="red",
                opacity=0.9,
                show_scalar_bar=False,
            )

        self._arrange_scalar_bars()

    def toggle_cold_shot_risk(self, result: AnalysisResult, checked: bool):
        if checked:
            self.show_cold_shot_risk(result)
        else:
            if self._cold_shot_actor is not None:
                self.remove_actor(self._cold_shot_actor)
                self._cold_shot_actor = None
            if self._last_fill_actor is not None:
                self.remove_actor(self._last_fill_actor)
                self._last_fill_actor = None
            self._remove_scalar_bar("Soğuk birleşme riski")

    def show_erosion_risk(self, result: Optional[AnalysisResult]):
        """Heatmap of mold-sand erosion risk driven by local metal velocity."""
        if self._erosion_actor is not None:
            self.remove_actor(self._erosion_actor)
            self._erosion_actor = None
        self._remove_scalar_bar("Kalıp erozyonu riski")

        if result is None or result.erosion_risk is None or result.erosion_risk.size == 0:
            return

        grid = self._make_grid(result, result.erosion_risk, "erosion_risk")
        metal = self._metal_only(grid)
        if metal.n_cells == 0:
            return

        cells = metal.threshold(1e-6, scalars="erosion_risk", all_scalars=True)
        if cells.n_cells == 0:
            return

        vmax = float(np.percentile(cells["erosion_risk"], 99))
        vmax = max(vmax, 0.3)
        clim = [0.0, vmax]

        self._erosion_actor = self.add_mesh(
            cells,
            scalars="erosion_risk",
            cmap="YlOrRd",
            opacity=0.85,
            clim=clim,
            show_scalar_bar=True,
            scalar_bar_args=_scalar_bar_args("Kalıp erozyonu riski", (0.02, 0.02), clim=clim),
            smooth_shading=True,
        )
        self._arrange_scalar_bars()

    def toggle_erosion_risk(self, result: AnalysisResult, checked: bool):
        if checked:
            self.show_erosion_risk(result)
        else:
            if self._erosion_actor is not None:
                self.remove_actor(self._erosion_actor)
                self._erosion_actor = None
            self._remove_scalar_bar("Kalıp erozyonu riski")

    def show_air_entrapment(self, result: Optional[AnalysisResult]):
        """Heatmap of trapped-air pockets detected by the LBM/VOF free-surface solver."""
        if self._air_entrapment_actor is not None:
            self.remove_actor(self._air_entrapment_actor)
            self._air_entrapment_actor = None
        if self._air_entrapment_marker_actor is not None:
            self.remove_actor(self._air_entrapment_marker_actor)
            self._air_entrapment_marker_actor = None
        self._remove_scalar_bar("Hava sıkışması")

        if result is None or result.air_entrapment is None or result.air_entrapment.size == 0:
            return

        grid = self._make_grid(result, result.air_entrapment, "air_entrapment")
        part = self._part_only(grid)
        if part.n_cells == 0:
            part = self._metal_only(grid)
        if part.n_cells == 0:
            return

        cells = part.threshold(0.3, scalars="air_entrapment", all_scalars=True)
        if cells.n_cells == 0:
            return

        vmax = max(float(np.percentile(cells["air_entrapment"], 99)), 0.5)
        if vmax <= 0.3:
            vmax = 1.0
        clim = [0.3, vmax]

        self._air_entrapment_actor = self.add_mesh(
            cells,
            scalars="air_entrapment",
            cmap="cool",
            opacity=0.85,
            clim=clim,
            show_scalar_bar=True,
            scalar_bar_args=_scalar_bar_args("Hava sıkışması", (0.02, 0.02), clim=clim),
            smooth_shading=True,
        )

        if result.air_entrapment_centroid_mm is not None and result.air_entrapment_centroid_mm.size == 3:
            radius = max(float(result.dx_mm) * 2.0, 2.0)
            sphere = pv.Sphere(radius=radius, center=result.air_entrapment_centroid_mm)
            self._air_entrapment_marker_actor = self.add_mesh(
                sphere,
                color="cyan",
                opacity=0.9,
                show_scalar_bar=False,
            )

        self._arrange_scalar_bars()

    def toggle_air_entrapment(self, result: AnalysisResult, checked: bool):
        if checked:
            self.show_air_entrapment(result)
        else:
            if self._air_entrapment_actor is not None:
                self.remove_actor(self._air_entrapment_actor)
                self._air_entrapment_actor = None
            if self._air_entrapment_marker_actor is not None:
                self.remove_actor(self._air_entrapment_marker_actor)
                self._air_entrapment_marker_actor = None
            self._remove_scalar_bar("Hava sıkışması")

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

        # Transparent cutting plane sized to the body bounds.
        # Sanitise the plane normal: a zero/NaN normal makes vtkPlaneSource
        # raise "Bad plane coordinate system". Clamp it to a finite unit vector.
        axis = np.asarray(axis, dtype=np.float64)
        axis = np.nan_to_num(axis)
        n = float(np.linalg.norm(axis))
        if n <= 1e-12:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            axis = axis / n
        point = np.asarray(point, dtype=np.float64)
        point = np.nan_to_num(point)

        bounds = body.mesh.bounds
        diag = float(np.linalg.norm(bounds[1] - bounds[0]))
        if diag <= 0 or not np.isfinite(diag):
            diag = 50.0
        try:
            plane = pv.Plane(
                center=point,
                direction=axis,
                i_size=diag,
                j_size=diag,
            )
        except Exception as exc:
            print(f"[section] Plane creation failed: {exc}")
            return
        actor = self.add_mesh(
            plane,
            color="#3B82F6",
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
