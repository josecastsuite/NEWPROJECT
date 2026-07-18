"""PyVistaQt 3D viewer wrapper for JoseCast Analyzer."""

from typing import List, Optional

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

from core.types import AnalysisResult, Body, BodyType


BODY_COLORS = {
    BodyType.PART: "#E0E0E0",
    BodyType.RISER: "#4CAF50",
    BodyType.INGATE: "#2196F3",
    BodyType.RUNNER: "#FF9800",
    BodyType.SPRUE: "#9C27B0",
    BodyType.CORE: "#795548",
}

RISK_COLORMAP = ["#000000", "#0000FF", "#00FF00", "#FFFF00", "#FF0000"]


class Analyzer3DViewer(QtInteractor):
    """Extended PyVistaQt interactor for casting analysis."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.set_background("#1e1e1e")
        self.add_axes(line_width=2)
        self._body_actors = []
        self._hotspot_actors = []
        self._risk_actor = None
        self._feed_path_actor = None

    def clear_scene(self):
        self.clear_actors()
        self.add_axes(line_width=2)
        self._body_actors.clear()
        self._hotspot_actors.clear()
        self._risk_actor = None
        self._feed_path_actor = None

    def show_bodies(self, bodies: List[Body]):
        """Display the original body meshes colored by type."""
        for actor in self._body_actors:
            self.remove_actor(actor)
        self._body_actors.clear()

        for body in bodies:
            if len(body.faces) == 0:
                continue
            mesh = pv.PolyData(body.vertices, np.c_[np.full(len(body.faces), 3), body.faces].ravel())
            color = BODY_COLORS.get(body.body_type, "#E0E0E0")
            actor = self.add_mesh(
                mesh,
                color=color,
                opacity=0.85,
                show_edges=False,
                smooth_shading=True,
                pickable=False,
            )
            self._body_actors.append(actor)
        self.reset_camera()

    def show_hotspots(self, result: Optional[AnalysisResult]):
        for actor in self._hotspot_actors:
            self.remove_actor(actor)
        self._hotspot_actors.clear()
        if result is None or not result.hotspots:
            return

        for hs in result.hotspots:
            sphere = pv.Sphere(radius=max(0.5, hs.m_value_mm * 0.15), center=hs.position_mm)
            actor = self.add_mesh(
                sphere,
                color="red" if not hs.feed_ok else "orange",
                opacity=0.8,
            )
            self._hotspot_actors.append(actor)

    def show_risk(self, result: Optional[AnalysisResult], threshold: float = 0.7):
        if self._risk_actor is not None:
            self.remove_actor(self._risk_actor)
            self._risk_actor = None
        if result is None:
            return

        grid = pv.ImageData()
        grid.dimensions = np.array(result.grid.shape) + 1
        grid.origin = result.origin_mm
        grid.spacing = (result.dx_mm, result.dx_mm, result.dx_mm)
        grid.cell_data["risk"] = result.risk.flatten(order="F")

        thresh = grid.threshold([threshold, 1.0])
        if thresh.n_cells == 0:
            return

        self._risk_actor = self.add_mesh(
            thresh,
            scalars="risk",
            cmap="hot",
            opacity=0.6,
            show_scalar_bar=True,
            scalar_bar_args={"title": "Poro. Riski"},
        )

    def show_sdf_slice(self, result: Optional[AnalysisResult], normal: str = "z", index: Optional[int] = None):
        if result is None:
            return
        grid = pv.ImageData()
        grid.dimensions = np.array(result.grid.shape) + 1
        grid.origin = result.origin_mm
        grid.spacing = (result.dx_mm, result.dx_mm, result.dx_mm)
        grid.cell_data["sdf"] = result.sdf.flatten(order="F")

        if index is None:
            index = result.grid.shape[2] // 2

        slice_mesh = grid.slice(normal=normal, origin=grid.center)
        self.add_mesh(slice_mesh, scalars="sdf", cmap="viridis", opacity=0.9)

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

    def save_screenshot(self, path: str) -> str:
        """Save a PNG screenshot of the current 3D view."""
        super().screenshot(path, transparent_background=False)
        return path
