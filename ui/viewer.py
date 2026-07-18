"""PyVistaQt 3D viewer wrapper for JoseCast Analyzer v8.2."""

from typing import List, Optional

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

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
}

# PART is transparent so internal porosity/hot-spots are visible;
# feeders / sprue / runner / riser / core remain opaque.
BODY_OPACITY = {
    BodyType.PART: 0.35,
    BodyType.RISER: 1.0,
    BodyType.INGATE: 1.0,
    BodyType.RUNNER: 1.0,
    BodyType.SPRUE: 1.0,
    BodyType.CORE: 1.0,
}


SCALAR_BAR_ARGS = {
    "color": "#00ffff",
    "title_font_size": 12,
    "label_font_size": 10,
    "fmt": "%.2f",
    "vertical": False,
    "position_x": 0.82,
    "position_y": 0.02,
    "width": 0.15,
    "height": 0.08,
}


class Analyzer3DViewer(QtInteractor):
    """Extended PyVistaQt interactor for casting analysis."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.set_background("#050505", top="#0a0a1a")
        self.add_axes(line_width=2, color="#00ffff")
        # Add a bright fill light so colored bodies remain visible in the dark scene.
        try:
            self.add_light(pv.Light(light_type='headlight'))
        except Exception:
            pass
        try:
            self.enable_depth_peeling(number_of_peels=32, occlusion_ratio=0.0)
            if self.ren_win is not None:
                self.ren_win.SetMultiSamples(0)
                self.ren_win.SetAlphaBitPlanes(1)
        except Exception:
            pass

        self._body_actors = []
        self._hotspot_actors = []
        self._label_actor = None
        self._risk_actor = None
        self._niyama_actor = None
        self._porosity_actor = None
        self._path_actors = []
        self._slice_actors = []
        self._local_actors = []

    def clear_scene(self):
        self.clear_actors()
        self.add_axes(line_width=2, color="#00ffff")
        self._body_actors.clear()
        self._hotspot_actors.clear()
        self._label_actor = None
        self._risk_actor = None
        self._niyama_actor = None
        self._porosity_actor = None
        self._path_actors.clear()
        self._slice_actors.clear()
        self._local_actors.clear()

    def show_bodies(self, bodies: List[Body], reset_camera: bool = True):
        """Display the original body meshes colored by type with transparency for the part."""
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

    def _make_image_data(self, result: AnalysisResult, scalar_field: np.ndarray, scalar_name: str):
        grid = pv.ImageData()
        grid.dimensions = np.array(result.grid.shape) + 1
        grid.origin = result.origin_mm
        grid.spacing = (result.dx_mm, result.dx_mm, result.dx_mm)
        grid.cell_data[scalar_name] = scalar_field.flatten(order="F")
        return grid

    def show_hotspots(self, result: Optional[AnalysisResult]):
        for actor in self._hotspot_actors:
            self.remove_actor(actor)
        self._hotspot_actors.clear()
        if self._label_actor is not None:
            self.remove_actor(self._label_actor)
            self._label_actor = None
        if result is None or not result.hotspots:
            return

        centers = []
        labels = []
        alloy = get_alloy(result.alloy_key)
        for hs in result.hotspots:
            radius = max(3.0, hs.m_value_mm * 2.0)
            sphere = pv.Sphere(radius=radius, center=hs.position_mm, theta_resolution=24, phi_resolution=24)
            if not hs.feed_ok:
                color = "#ff3333"
            elif hs.niyama_ensemble < alloy.niyama_macro:
                color = "#ff3333"
            elif hs.niyama_ensemble < alloy.niyama_shrinkage:
                color = "#ffaa00"
            else:
                color = "#00ff88"
            actor = self.add_mesh(sphere, color=color, opacity=1.0, lighting=False, ambient=1.0)
            self._hotspot_actors.append(actor)

            centers.append(hs.position_mm + np.array([0, 0, radius + 1.0]))
            labels.append(
                f"M={hs.m_value_mm:.1f}mm\nD={hs.dist_to_riser_mm:.0f}mm\nN={hs.niyama_ensemble:.2f}"
            )

        if centers:
            self._label_actor = self.add_point_labels(
                np.array(centers),
                labels,
                text_color="#00ffff",
                font_size=10,
                shape=None,
                show_points=False,
                always_visible=True,
            )

    def show_risk(self, result: Optional[AnalysisResult], threshold: float = 0.7, max_points: int = 3000):
        if self._risk_actor is not None:
            self.remove_actor(self._risk_actor)
            self._risk_actor = None
        if result is None:
            return

        grid = self._make_image_data(result, result.risk, "risk")
        thresh = grid.threshold([threshold, 1.0])
        if thresh.n_cells == 0:
            return

        cloud = thresh.cell_centers()
        if cloud.n_points > max_points:
            idx = np.random.choice(cloud.n_points, max_points, replace=False)
            points = cloud.points[idx]
            scalars = cloud["risk"][idx]
            cloud = pv.PolyData(points)
            cloud["risk"] = scalars

        self._risk_actor = self.add_mesh(
            cloud,
            scalars="risk",
            cmap="hot",
            style="points",
            point_size=1,
            render_points_as_spheres=False,
            opacity=0.15,
            show_scalar_bar=True,
            scalar_bar_args={**SCALAR_BAR_ARGS, "title": "Shrinkage Risk"},
        )
    def show_porosity_cloud(self, result: Optional[AnalysisResult], threshold: float = 0.85, max_points: int = 5000):
        """High-risk porosity point cloud (sampled so bodies stay visible)."""
        if self._porosity_actor is not None:
            self.remove_actor(self._porosity_actor)
            self._porosity_actor = None
        if result is None:
            return

        grid = self._make_image_data(result, result.risk, "risk")
        thresh = grid.threshold([threshold, 1.0])
        if thresh.n_cells == 0:
            return

        cloud = thresh.cell_centers()
        if cloud.n_points > max_points:
            idx = np.random.choice(cloud.n_points, max_points, replace=False)
            points = cloud.points[idx]
            scalars = cloud["risk"][idx]
            cloud = pv.PolyData(points)
            cloud["risk"] = scalars

        self._porosity_actor = self.add_mesh(
            cloud,
            scalars="risk",
            cmap="hot",
            style="points",
            point_size=5,
            render_points_as_spheres=True,
            opacity=0.85,
            emissive=True,
            show_scalar_bar=True,
            scalar_bar_args={**SCALAR_BAR_ARGS, "title": "Porozite Riski"},
        )

    def show_niyama_isosurfaces(self, result: Optional[AnalysisResult]):
        """Niyama point cloud for the macro-shrinkage / micro-porosity band."""
        if self._niyama_actor is not None:
            self.remove_actor(self._niyama_actor)
            self._niyama_actor = None
        if result is None:
            return

        alloy = get_alloy(result.alloy_key)
        grid = self._make_image_data(result, result.niyama, "niyama")
        band = grid.threshold([0.0, alloy.niyama_shrinkage])
        if band.n_cells == 0:
            return

        cloud = band.cell_centers()
        if cloud.n_points > 4000:
            idx = np.random.choice(cloud.n_points, 4000, replace=False)
            points = cloud.points[idx]
            scalars = cloud["niyama"][idx]
            cloud = pv.PolyData(points)
            cloud["niyama"] = scalars

        self._niyama_actor = self.add_mesh(
            cloud,
            scalars="niyama",
            cmap="plasma",
            style="points",
            point_size=3,
            render_points_as_spheres=False,
            opacity=0.35,
            show_scalar_bar=True,
            scalar_bar_args={**SCALAR_BAR_ARGS, "title": "Niyama"},
        )

    def show_feeding_paths(self, result: Optional[AnalysisResult]):
        for actor in self._path_actors:
            self.remove_actor(actor)
        self._path_actors.clear()
        if result is None or result.dist_to_riser.size == 0:
            return

        part_mask = result.grid == BodyType.PART
        for hs in result.hotspots:
            vox = np.round((hs.position_mm - result.origin_mm) / result.dx_mm).astype(int)
            if not (0 <= vox[0] < part_mask.shape[0] and 0 <= vox[1] < part_mask.shape[1] and 0 <= vox[2] < part_mask.shape[2]):
                continue
            if not part_mask[vox[0], vox[1], vox[2]]:
                continue
            path = _trace_path_to_riser(result.dist_to_riser, part_mask, vox)
            if len(path) < 2:
                continue
            pts = np.array(path) * result.dx_mm + result.origin_mm
            if len(pts) < 2:
                continue
            poly = pv.PolyData()
            poly.points = pts
            poly.lines = np.hstack([[len(pts)], np.arange(len(pts))]).astype(np.int64)
            radius = max(1.0, result.dx_mm * 0.8)
            try:
                tube = poly.tube(radius=radius)
            except Exception:
                tube = poly
            color = "#00ff88" if hs.feed_ok else "#ff3333"
            actor = self.add_mesh(tube, color=color, opacity=0.9, smooth_shading=True, lighting=False)
            self._path_actors.append(actor)

    def show_slices(self, result: Optional[AnalysisResult], field: str = "sdf"):
        """Add three orthogonal slices through the center for the selected scalar field."""
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
            "temperature": (result.temperature if result.temperature.size > 0 else result.sdf, "T (°C)", "coolwarm"),
        }
        if field not in field_map:
            return
        data, title, cmap = field_map[field]
        grid = self._make_image_data(result, data, field)
        origin = grid.center
        for normal in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            slice_mesh = grid.slice(normal=normal, origin=origin)
            actor = self.add_mesh(
                slice_mesh,
                scalars=field,
                cmap=cmap,
                opacity=0.9,
                show_scalar_bar=True,
                scalar_bar_args={**SCALAR_BAR_ARGS, "title": title},
            )
            self._slice_actors.append(actor)

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

    def toggle_risk(self, result: AnalysisResult, checked: bool):
        if checked:
            self.show_risk(result)
        elif self._risk_actor is not None:
            self.remove_actor(self._risk_actor)
            self._risk_actor = None

    def toggle_hotspots(self, result: AnalysisResult, checked: bool):
        if checked:
            self.show_hotspots(result)
        else:
            for actor in self._hotspot_actors:
                self.remove_actor(actor)
            self._hotspot_actors.clear()
            if self._label_actor is not None:
                self.remove_actor(self._label_actor)
                self._label_actor = None

    def toggle_porosity(self, result: AnalysisResult, checked: bool):
        if checked:
            self.show_porosity_cloud(result)
        elif self._porosity_actor is not None:
            self.remove_actor(self._porosity_actor)
            self._porosity_actor = None

    def toggle_niyama(self, result: AnalysisResult, checked: bool):
        if checked:
            self.show_niyama_isosurfaces(result)
        elif self._niyama_actor is not None:
            self.remove_actor(self._niyama_actor)
            self._niyama_actor = None

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

    def save_screenshot(self, path: str) -> str:
        """Save a PNG screenshot of the current 3D view."""
        super().screenshot(path, transparent_background=False)
        return path
