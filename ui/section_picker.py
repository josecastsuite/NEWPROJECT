"""Modal 3D section picker dialog for gating cross-section area correction."""
from typing import List, Optional

from PyQt6 import QtWidgets

from core.types import Body
from ui.viewer import Analyzer3DViewer


class SectionPickerDialog(QtWidgets.QDialog):
    """A separate viewer window where the user clicks a gating body to measure its
    characteristic cross-sectional area.  The measured value is returned to the
    caller through ``area_cm2`` and ``body_name``.
    """

    def __init__(
        self,
        bodies: List[Body],
        section_key: str,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Kesit Seç - {section_key}")
        self.resize(1000, 750)
        self.bodies = bodies
        self.section_key = section_key
        self.area_cm2: Optional[float] = None
        self.body_name: str = ""
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel(
            f"<b>{self.section_key}</b> kesiti için 3D görünümde ilgili body'e "
            "tıklayın. Kesit alanı otomatik hesaplanacak."
        )
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

        self.viewer = Analyzer3DViewer(parent=self)
        layout.addWidget(self.viewer, stretch=1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def showEvent(self, event):
        super().showEvent(event)
        self.viewer.show_bodies(self.bodies)
        self.viewer.start_section_picker(
            self.section_key,
            self.bodies,
            callback=self._on_picked,
        )

    def _on_picked(self, section_key: str, area_cm2: float, body_name: str):
        self.area_cm2 = area_cm2
        self.body_name = body_name
        self.accept()

    def closeEvent(self, event):
        try:
            self.viewer.disable_picking()
        except Exception:
            pass
        try:
            self.viewer.clear_scene()
        except Exception:
            pass
        super().closeEvent(event)
