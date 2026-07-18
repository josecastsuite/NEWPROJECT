"""PyVistaQt 3D viewer wrapper for JoseCast Analyzer v7."""

from typing import List, Optional

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

from core.types import AnalysisResult, Body, BodyType, RefinementRegion


BODY_COLORS = {
    BodyType.PART: "#E0E0E0",
    BodyType.RISER: "#4CAF50",
    BodyType.INGATE: "#2196F3",
    BodyType.RUNNER: "#FF9800",
    BodyType.SPRUE: "#9C27B0",
    BodyType.CORE: "#795548",
}


class Analyzer3DViewer(QtInteractor):
    """Extended PyVistaQt interactor for casting analysis."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.set_background("#050505")
        self.add_axes(line_width=2)
        self._body_actors = []
        self._hotspot_actors = []
        self._risk_actor = None
        self._slice_actors = []
        self._local_actors = []

    def clear_scene(self):
        self.clear_actors()
        self.add_axes(line_width=2)
        self._body_actors.clear()
        self._hotspot_actors.clear()
        self._risk_actor = None
        self._slice_actors.clear()
        self._local_actors.clear()

    def show_bodies(self, bodies: List[Body]):
        """Display the original body meshes colored by type."""
        for actor in self._body_actors:
            self.remove_actor(actor)
        self._body_actors.clear()

        for body in bodies:
            if len(body.faces) == 0:
                continue
            faces = np.c_[np.full(len(body.faces), 3), body.faces].ravel()
            mesh = pv.PolyData(body.vertices, faces)
            color = BODY_COLORS.get(body.body_type, "#E0E0E0")
            actor = self.add_mesh(
                mesh,
                color=color,
                opacity=0.8,
                show_edges=False,
                smooth_shading=True,
                pickable=False,
            )
            self._body_actors.append(actor)
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
        if result is None or not result.hotspots:
            return

        for hs in result.hotspots:
            sphere = pv.Sphere(
                radius=max(0.5, hs.m_value_mm * 0.15), center=hs.position_mm
            )
            color = "red" if not hs.feed_ok else ("orange" if not hs.resistance_ok else "#00ff88")
            actor = self.add_mesh(sphere, color=color, opacity=0.85)
            self._hotspot_actors.append(actor)

    def show_risk(self, result: Optional[AnalysisResult], threshold: float = 0.7):
        if self._risk_actor is not None:
            self.remove_actor(self._risk_actor)
            self._risk_actor = None
        if result is None:
            return

        grid = self._make_image_data(result, result.risk, "risk")
        thresh = grid.threshold([threshold, 1.0])
        if thresh.n_cells == 0:
            return

        self._risk_actor = self.add_mesh(
            thresh,
            scalars="risk",
            cmap="hot",
            opacity=0.55,
            show_scalar_bar=True,
            scalar_bar_args={"title": "Shrinkage Risk"},
        )

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
                opacity=0.85,
                show_scalar_bar=True,
                scalar_bar_args={"title": title},
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
