"""PyVistaQt 3D viewer wrapper for JoseCast Analyzer v8.x."""

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
from core.types import AnalysisResult, Body, BodyType, RefinementRegion


BODY_COLORS = {
    BodyType.PART: "#E0E0E0",
    BodyType.RISER: "#4CAF50",
    BodyType.INGATE: "#2196F3",
    BodyType.RUNNER: "#FF9800",
    BodyType.SPRUE: "#9C27B0",
    BodyType.CORE: "#795548",
    BodyType.COOLING_SPRUE: "#00BCD4",
    BodyType.FILTER: "#607D8B",
    BodyType.POURING_BASIN: "#3F51B5",
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
}


def _scalar_bar_args(title: str, pos: Tuple[float, float]) -> dict:
    return {
        "color": "#00ffff",
        "title_font_size": 12,
        "label_font_size": 10,
        "fmt": "%.2f",
        "vertical": False,
        "position_x": pos[0],
        "position_y": pos[1],
        "width": 0.15,
        "height": 0.08,
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

    def clear_scene(self):
        self.clear_actors()
        self.add_axes(line_width=2, color="#00ffff")
        self._body_actors.clear()
        self._hotspot_actors.clear()
        self._hotspot_label_actor = None
        self._risk_actor = None
        self._porosity_actor = None
        self._niyama_actors.clear()
        self._path_actors.clear()
        self._slice_actors.clear()
        self._local_actors.clear()
        self._clear_section_actors()

    def show_bodies(self, bodies: List[Body], reset_camera: bool = True):
        """Display original body meshes colored by type; part is semi-transparent."""
        for actor in self._body_actors:
            self.remove_actor(actor)
        self._body_actors.clear()

        for body in bodies:
            if len(body.faces) == 0:
                continue
            faces = np.c_[np.full(len(body.faces), 3, dtype=np.int64), body.faces].ravel()
            mesh = pv.PolyData(body.vertices, faces)
            color = BODY_COLORS.get(body.body_type, "#E0E0E0")
            opacity = BODY_OPACITY.get(body.body_type, 1.0)
            actor = self.add_mesh(
                mesh,
                color=color,
                opacity=opacity,
                show_edges=False,
                smooth_shading=True,
                pickable=False,
            )
            self._body_actors.append(actor)
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
        if result is None or not result.hotspots:
            return

        alloy = get_alloy(result.alloy_key)
        bbox_min = np.min(result.bbox_size_mm) if result.bbox_size_mm.any() else 100.0
        centers = []
        labels = []
        for hs in result.hotspots:
            # Radius proportional to local wall thickness, clamped to a sensible fraction of the part size.
            radius = max(1.2, min(hs.t_section_mm * 0.15, bbox_min * 0.02, 6.0))
            sphere = pv.Sphere(
                radius=radius,
                center=hs.position_mm,
                theta_resolution=24,
                phi_resolution=24,
            )
            if not hs.feed_ok or hs.niyama_ensemble < alloy.niyama_macro:
                color = "#ff3333"
            elif hs.niyama_ensemble < alloy.niyama_shrinkage:
                color = "#ffaa00"
            else:
                color = "#00ff88"
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
            labels.append(
                f"M={hs.m_value_mm:.1f}mm | D={hs.dist_to_riser_mm:.0f}mm | N={hs.niyama_ensemble:.2f}"
            )

        if centers:
            try:
                self._hotspot_label_actor = self.add_point_labels(
                    np.array(centers),
                    labels,
                    text_color="#00ffff",
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
            scalar_bar_args=_scalar_bar_args("Risk", (0.82, 0.02)),
            smooth_shading=True,
        )

    def show_porosity_cloud(
        self,
        result: Optional[AnalysisResult],
        noise_percent: float = 3.0,
        max_points: int = 5000,
        pore_size_filter: Optional[str] = None,
    ):
        """Porosity point cloud colored by estimated pore size.

        Only the top ``noise_percent``% of the displayed scalar is rendered so
        that numerical / physical noise is suppressed.  ``pore_size_filter``
        restricts the cloud to ``macro``, ``micro`` or ``fine`` classes.
        Falls back to the slowest-solidifying regions only when pore-size data
        are not available or the user has not requested a specific class.
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
        # v9.2: visualize the shrinkage-only pore-size field; the gas/oxide
        # baseline would otherwise make every voxel positive and dominate the cloud.
        shrinkage_um = getattr(result, "pore_size_shrinkage_um", None)
        if shrinkage_um is not None and shrinkage_um.size == part_mask.size:
            pore_size_um = np.asarray(shrinkage_um)
        else:
            pore_size_um = np.asarray(result.pore_size_um) if result.pore_size_um is not None else np.array([])
        has_pore_size = pore_size_um.size and pore_size_um.shape == part_mask.shape

        class_mask = np.zeros_like(part_mask, dtype=bool)
        use_pore_size = False
        if has_pore_size and pore_size_filter in ("macro", "micro", "fine"):
            class_mask = np.asarray(getattr(result, f"pore_size_{pore_size_filter}_mask", class_mask))
            if class_mask is None or class_mask.size == 0:
                class_mask = np.zeros_like(part_mask, dtype=bool)
            use_pore_size = class_mask.any()
            # v9.1: if the user explicitly selected a class, do not fall back to
            # solidification-time/risk clouds; show nothing for that class.
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

        field = np.asarray(field, dtype=np.float64)
        values = field[class_mask]
        finite = np.isfinite(values) & (values > 0.0)
        if not finite.any():
            return
        finite_values = values[finite]
        finite_max = float(np.max(finite_values))

        # v9.2: per-class top percentages (macro 60%, micro 40%, fine 20%).
        # The slider scales these percentages for class filters; for "all" it is
        # the top percentage directly.
        if use_pore_size and pore_size_filter in ("macro", "micro", "fine"):
            target_percent = float(
                getattr(result, f"pore_size_{pore_size_filter}_percent", 0.0)
            )
            if target_percent <= 0.0:
                target_percent = {"macro": 60.0, "micro": 40.0, "fine": 20.0}[pore_size_filter]
            effective_percent = min(100.0, target_percent * (noise_percent / 3.0))
            p = max(0.0, 100.0 - effective_percent)
            lo = float(np.percentile(finite_values, p))
        else:
            p = max(0.0, min(100.0 - noise_percent, 100.0))
            lo = float(np.percentile(finite_values, p))

        # Physical floor: only positive shrinkage values.
        lo = max(lo, 0.0)
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
            scalar_bar_args=_scalar_bar_args(title, (0.82, 0.02)),
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
            scalar_bar_args=_scalar_bar_args("Niyama", (0.82, 0.16)),
            smooth_shading=True,
        )
        self._niyama_actors.append(actor)

    def show_feeding_paths(self, result: Optional[AnalysisResult]):
        for actor in self._path_actors:
            self.remove_actor(actor)
        self._path_actors.clear()
        if result is None or result.dist_to_riser.size == 0:
            return

        part_mask = result.grid == BodyType.PART
        for hs in result.hotspots:
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
                scalar_bar_args=_scalar_bar_args(title, (0.82, 0.18)),
                clim=[0.0, 1.0] if field == "risk" else None,
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
