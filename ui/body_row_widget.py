"""Compact per-body row widget for the JoséCast main window body list.

Each body occupies one horizontal row:
- body name
- body type selector
- optional feeder controls (visible only when body type == RISER)
    - feeder type
    - modulus (cm)
- optional mould-sand controls (visible only when body type == CORE)
    - sand preset
    - "..." button opening AFS / moisture / binder / compactability dialog

This keeps the body list compact and avoids large stacked panels.
"""
from typing import Dict, Optional

from PyQt6 import QtCore, QtWidgets

from core.materials import MOLDS
from core.types import Body, BodyType
from ui.mold_properties_dialog import MoldPropertiesDialog


FEEDER_TYPE_NAMES = {
    "conventional": "Konvansiyonel",
    "exothermic": "Ekzotermik",
    "insulated": "İzole",
    "sleeve": "Seramik/Manyetik",
    "side": "Yan",
    "blind": "Kör",
}

SAND_PRESET_NAMES = {
    "green_sand": "Yeşil Kum",
    "silica_sand": "Silis Kum",
    "chromite_sand": "Kromit Kum",
    "zircon_sand": "Zirkon Kum",
}


class BodyTypeComboBox(QtWidgets.QComboBox):
    """QComboBox that reports when its popup is shown/hidden."""

    popup_shown = QtCore.pyqtSignal()
    popup_hidden = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._show_emitted = False

    def showPopup(self):
        super().showPopup()
        if not self._show_emitted:
            self._show_emitted = True
            # Defer the signal so showPopup returns and the popup is fully
            # rendered before the viewer is rebuilt.
            QtCore.QTimer.singleShot(0, self.popup_shown.emit)

    def hidePopup(self):
        super().hidePopup()
        if self._show_emitted:
            self._show_emitted = False
            QtCore.QTimer.singleShot(0, self.popup_hidden.emit)


class BodyRowWidget(QtWidgets.QWidget):
    """Single-line body row with inline, type-conditional feeder / mould controls."""

    body_type_changed = QtCore.pyqtSignal(Body, int)
    feeder_type_changed = QtCore.pyqtSignal(Body, str)
    feeder_m_changed = QtCore.pyqtSignal(Body, float)
    mold_settings_changed = QtCore.pyqtSignal(Body)
    body_focused = QtCore.pyqtSignal(Body)
    body_unfocused = QtCore.pyqtSignal()

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

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        self._build_ui()
        self._sync_from_body()

    def _build_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(4)
        self.setMinimumHeight(32)

        name_label = QtWidgets.QLabel(self._body.name)
        name_label.setToolTip(
            f"Hacim: {self._body.volume_cm3:.2f} cm³\nMerkez: {self._body.center}"
        )
        name_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        name_label.setStyleSheet(
            "background-color: #F1F5F9; color: #334155; border: 1px solid #CBD5E1; "
            "border-radius: 4px; padding: 3px; font-size: 11px; font-weight: 600; "
            "min-height: 24px;"
        )
        name_label.setMaximumWidth(70)
        name_label.setMinimumHeight(24)
        name_label.setMaximumHeight(32)
        name_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        layout.addWidget(name_label)

        self._type_combo = BodyTypeComboBox()
        self._type_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._type_combo.setMinimumWidth(80)
        self._type_combo.setMinimumHeight(24)
        self._type_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        for bt, label in self._body_type_names.items():
            self._type_combo.addItem(label, int(bt))
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        self._type_combo.popup_shown.connect(lambda: self.body_focused.emit(self._body))
        self._type_combo.popup_hidden.connect(self.body_unfocused.emit)
        layout.addWidget(self._type_combo)

        # --- Feeder controls: compact, hidden unless RISER ---
        self._feeder_type_combo = QtWidgets.QComboBox()
        self._feeder_type_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._feeder_type_combo.setMinimumWidth(80)
        self._feeder_type_combo.setMinimumHeight(24)
        self._feeder_type_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
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
        self._feeder_m_spin.setMinimumWidth(60)
        self._feeder_m_spin.setMaximumWidth(95)
        self._feeder_m_spin.setMinimumHeight(24)
        self._feeder_m_spin.valueChanged.connect(self._on_feeder_m_changed)
        layout.addWidget(self._feeder_m_spin)

        # --- Mould-sand controls: compact, hidden unless CORE ---
        self._sand_type_combo = QtWidgets.QComboBox()
        self._sand_type_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._sand_type_combo.setMinimumWidth(80)
        self._sand_type_combo.setMinimumHeight(24)
        self._sand_type_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self._sand_type_combo.setToolTip("Kum tipi")
        for key, name in SAND_PRESET_NAMES.items():
            self._sand_type_combo.addItem(name, key)
        self._sand_type_combo.currentIndexChanged.connect(self._on_sand_type_changed)
        layout.addWidget(self._sand_type_combo)

        self._sand_prop_btn = QtWidgets.QPushButton("Parametreler")
        self._sand_prop_btn.setToolTip(
            "Kum parametrelerini düzenle (AFS tane, nem %, bağlayıcı %, compactability %)"
        )
        self._sand_prop_btn.setStyleSheet(
            "background-color: #3B82F6; color: #FFFFFF; border: none; "
            "border-radius: 4px; padding: 2px; font-size: 11px; font-weight: 600; "
            "text-align: center;"
        )
        self._sand_prop_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self._sand_prop_btn.setMinimumWidth(80)
        self._sand_prop_btn.setMaximumWidth(95)
        self._sand_prop_btn.setMinimumHeight(24)
        self._sand_prop_btn.setMaximumHeight(32)
        self._sand_prop_btn.clicked.connect(self._on_sand_properties)
        layout.addWidget(self._sand_prop_btn)

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
            self._feeder_type_combo.setCurrentIndex(idx if idx >= 0 else 0)
            # stored in mm, shown in cm
            self._feeder_m_spin.setValue((self._body.feeder_m_mm or 0.0) / 10.0)

            sidx = self._sand_type_combo.findData(self._body.mold_preset or "")
            self._sand_type_combo.setCurrentIndex(sidx if sidx >= 0 else 0)
        finally:
            self._block_updates = False

    def _update_visibility(self, body_type: BodyType) -> None:
        is_riser = body_type == BodyType.RISER
        is_core = body_type == BodyType.CORE
        self._feeder_type_combo.setVisible(is_riser)
        self._feeder_m_spin.setVisible(is_riser)
        self._sand_type_combo.setVisible(is_core)
        self._sand_prop_btn.setVisible(is_core)

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
            self._sync_from_body()
        if new_type != BodyType.CORE:
            # Do not erase stored sand overrides; just hide the widgets.
            pass
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

    def _on_sand_type_changed(self, index: int) -> None:
        if self._block_updates:
            return
        key = self._sand_type_combo.itemData(index) or "green_sand"
        self._body.mold_preset = key
        self.mold_settings_changed.emit(self._body)

    def _on_sand_properties(self) -> None:
        dialog = MoldPropertiesDialog(
            parent=self,
            body=self._body,
            preset_key=self._body.mold_preset or "green_sand",
            title=f"Kum Parametreleri – {self._body.name}",
        )
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.mold_settings_changed.emit(self._body)
