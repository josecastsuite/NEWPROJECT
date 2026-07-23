"""Dialog to set riser/feeder type and optional modulus."""
from typing import Optional

from PyQt6 import QtCore, QtWidgets

from core.types import Body


FEEDER_TYPE_NAMES = {
    "conventional": "Konvansiyonel kum kalıp besleyici",
    "exothermic": "Ekzotermik gömleklı besleyici",
    "insulated": "İzole gömleklı besleyici",
    "sleeve": "Seramik / manyetik gömlek besleyici",
    "chilled": "Soğutucu çelik / chill besleyici",
}


class FeederDialog(QtWidgets.QDialog):
    """Let the user choose the feeder (riser) type and an optional modulus.

    Returns ``feeder_type`` (key string), ``feeder_m_mm`` (0.0 = auto) and
    ``feeder_note`` through the dialog attributes.
    """

    def __init__(self, body: Body, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(f"{body.name} - Besleyici Tipi")
        self.resize(380, 260)
        self.setWindowFlags(
            self.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint
        )
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

        self.feeder_type = body.feeder_type or "conventional"
        # The user enters modulus in cm (industry standard); internal storage is mm.
        self.feeder_m_mm = float(body.feeder_m_mm)
        self.feeder_note = body.feeder_note or ""

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        info = QtWidgets.QLabel(
            "Besleyici tipi ve isteğe bağlı modül girin.<br>"
            "M = 0 ise program otomatik hesaplar."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.type_combo = QtWidgets.QComboBox()
        for key, name in FEEDER_TYPE_NAMES.items():
            self.type_combo.addItem(name, key)
        idx = self.type_combo.findData(self.feeder_type)
        self.type_combo.setCurrentIndex(idx if idx >= 0 else 0)
        layout.addWidget(QtWidgets.QLabel("Besleyici tipi:"))
        layout.addWidget(self.type_combo)

        self.m_spin = QtWidgets.QDoubleSpinBox()
        self.m_spin.setRange(0.0, 999.0)
        self.m_spin.setDecimals(2)
        self.m_spin.setValue(self.feeder_m_mm / 10.0)
        self.m_spin.setSuffix(" cm")
        layout.addWidget(QtWidgets.QLabel("Opsiyonel besleyici modülü (M):"))
        layout.addWidget(self.m_spin)

        self.note_edit = QtWidgets.QLineEdit(self.feeder_note)
        self.note_edit.setPlaceholderText("Not (örn: exotermik %40)")
        layout.addWidget(QtWidgets.QLabel("Not:"))
        layout.addWidget(self.note_edit)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _on_accept(self):
        self.feeder_type = self.type_combo.currentData() or "conventional"
        # Convert user input from cm to mm for internal calculations.
        self.feeder_m_mm = float(self.m_spin.value()) * 10.0
        self.feeder_note = self.note_edit.text().strip()
        self.accept()
