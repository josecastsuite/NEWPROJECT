"""Compact per-body row widget for the JoséCast main window body list.

Each body occupies one horizontal row:
- body name
- body type selector
- optional feeder controls (visible only when body type == RISER)
    - feeder type
    - modulus (cm)
    - note

This keeps the body list compact and avoids large stacked panels.
"""
from typing import Dict, Optional

from PyQt6 import QtCore, QtWidgets

from core.types import Body, BodyType


FEEDER_TYPE_NAMES = {
    "conventional": "Konvansiyonel",
    "exothermic": "Ekzotermik",
    "insulated": "İzole",
    "sleeve": "Seramik/Manyetik",
    "chilled": "Chill",
    "side": "Yan",
    "blind": "Kör",
}


class BodyRowWidget(QtWidgets.QWidget):
    """Single-line body row with inline, type-conditional feeder controls."""

    body_type_changed = QtCore.pyqtSignal(Body, int)
    feeder_type_changed = QtCore.pyqtSignal(Body, str)
    feeder_m_changed = QtCore.pyqtSignal(Body, float)
    feeder_note_changed = QtCore.pyqtSignal(Body, str)

    def __init__(
        self,
        body: Body,
        body_type_names: Dict[BodyType, str],
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._body = body
        self._body_type_names = body_type_names
        self._block_updates = False

        self._build_ui()
        self._sync_from_body()

    def _build_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(4)

        name_label = QtWidgets.QLabel(self._body.name)
        name_label.setToolTip(
            f"Hacim: {self._body.volume_cm3:.2f} cm³\nMerkez: {self._body.center}"
        )
        name_label.setStyleSheet("font-size: 11px;")
        name_label.setMaximumWidth(80)
        name_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        layout.addWidget(name_label)

        self._type_combo = QtWidgets.QComboBox()
        self._type_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._type_combo.setMaximumWidth(110)
        for bt, label in self._body_type_names.items():
            self._type_combo.addItem(label, int(bt))
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        layout.addWidget(self._type_combo)

        # --- Feeder controls: compact, hidden unless RISER ---
        self._feeder_type_combo = QtWidgets.QComboBox()
        self._feeder_type_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._feeder_type_combo.setMaximumWidth(120)
        self._feeder_type_combo.setToolTip("Besleyici tipi")
        for key, name in FEEDER_TYPE_NAMES.items():
            self._feeder_type_combo.addItem(name, key)
        self._feeder_type_combo.currentIndexChanged.connect(self._on_feeder_type_changed)
        layout.addWidget(self._feeder_type_combo)

        self._feeder_m_spin = QtWidgets.QDoubleSpinBox()
        self._feeder_m_spin.setRange(0.0, 999.0)
        self._feeder_m_spin.setDecimals(2)
        self._feeder_m_spin.setSuffix(" cm")
        self._feeder_m_spin.setSpecialValueText("Auto")
        self._feeder_m_spin.setToolTip("Opsiyonel besleyici modülü (cm); 0 = otomatik")
        self._feeder_m_spin.setMaximumWidth(80)
        self._feeder_m_spin.valueChanged.connect(self._on_feeder_m_changed)
        layout.addWidget(self._feeder_m_spin)

        self._feeder_note_edit = QtWidgets.QLineEdit()
        self._feeder_note_edit.setPlaceholderText("Not")
        self._feeder_note_edit.setToolTip("Besleyici notu (örn: exotermik %40)")
        self._feeder_note_edit.setMaximumWidth(100)
        self._feeder_note_edit.textChanged.connect(self._on_feeder_note_changed)
        layout.addWidget(self._feeder_note_edit)

        layout.addStretch()

    def body(self) -> Body:
        return self._body

    def set_body_type(self, body_type: BodyType) -> None:
        idx = self._type_combo.findData(int(body_type))
        if idx >= 0 and idx != self._type_combo.currentIndex():
            self._type_combo.setCurrentIndex(idx)
        self._update_visibility(body_type)

    def _sync_from_body(self) -> None:
        self._block_updates = True
        try:
            self.set_body_type(self._body.body_type)
            idx = self._feeder_type_combo.findData(self._body.feeder_type or "")
            if idx >= 0:
                self._feeder_type_combo.setCurrentIndex(idx)
            else:
                self._feeder_type_combo.setCurrentIndex(0)
            # stored in mm, shown in cm
            self._feeder_m_spin.setValue((self._body.feeder_m_mm or 0.0) / 10.0)
            self._feeder_note_edit.setText(self._body.feeder_note or "")
        finally:
            self._block_updates = False

    def _update_visibility(self, body_type: BodyType) -> None:
        is_riser = body_type == BodyType.RISER
        self._feeder_type_combo.setVisible(is_riser)
        self._feeder_m_spin.setVisible(is_riser)
        self._feeder_note_edit.setVisible(is_riser)

    def _on_type_changed(self, index: int) -> None:
        if self._block_updates:
            return
        data = self._type_combo.itemData(index)
        try:
            new_type = BodyType(data)
        except Exception:
            new_type = BodyType.PART
        self._body.body_type = new_type
        if new_type != BodyType.RISER:
            self._body.feeder_type = ""
            self._body.feeder_m_mm = 0.0
            self._body.feeder_note = ""
            self._sync_from_body()
        self._update_visibility(new_type)
        self.body_type_changed.emit(self._body, int(new_type))

    def _on_feeder_type_changed(self, index: int) -> None:
        if self._block_updates:
            return
        key = self._feeder_type_combo.itemData(index) or "conventional"
        self._body.feeder_type = key
        self.feeder_type_changed.emit(self._body, key)

    def _on_feeder_m_changed(self, value: float) -> None:
        if self._block_updates:
            return
        self._body.feeder_m_mm = float(value) * 10.0
        self.feeder_m_changed.emit(self._body, self._body.feeder_m_mm)

    def _on_feeder_note_changed(self, text: str) -> None:
        if self._block_updates:
            return
        self._body.feeder_note = text.strip()
        self.feeder_note_changed.emit(self._body, self._body.feeder_note)
