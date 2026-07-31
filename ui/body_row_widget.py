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


FEEDER_TYPE_NAMES = {
    "conventional": "Konvansiyonel",
    "exothermic": "Ekzotermik",
    "insulated": "İzole",
    "sleeve": "Seramik/Manyetik",
    "chilled": "Chill",
    "side": "Yan",
    "blind": "Kör",
}

SAND_PRESET_NAMES = {
    "green_sand": "Yeşil Kum",
    "silica_sand": "Silis Kum",
    "chromite_sand": "Kromit Kum",
    "zircon_sand": "Zirkon Kum",
}


class MoldPropertiesDialog(QtWidgets.QDialog):
    """Per-CORE sand property override dialog with app-styled dark UI."""

    def __init__(self, body: Body, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._body = body
        self.setWindowTitle(f"Kum Özellikleri – {body.name}")
        self.setMinimumWidth(340)

        self.setStyleSheet(
            """
            QDialog { background-color: #18181b; }
            QLabel { color: #00ffff; font-weight: 800; font-size: 13px; }
            QGroupBox {
                color: #00ff88;
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #3f3f46;
                border-radius: 8px;
                margin-top: 14px;
                padding-top: 12px;
                padding-left: 10px;
                padding-right: 10px;
                padding-bottom: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                color: #00ff88;
                font-weight: bold;
            }
            QDoubleSpinBox {
                background: #27272a;
                color: #00ffff;
                border: 1px solid #52525b;
                border-radius: 5px;
                padding: 5px;
                min-height: 22px;
                font-weight: bold;
            }
            QPushButton {
                background: #00ff88;
                color: #000000;
                border: none;
                border-radius: 6px;
                padding: 8px 14px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background: #00cc6a; }
            """
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        header = QtWidgets.QLabel(
            f"<b>{body.name}</b> için kum parametrelerini ayarlayın."
        )
        header.setStyleSheet("color: #00ffff; font-size: 13px; font-weight: bold;")
        layout.addWidget(header)

        info = QtWidgets.QLabel(
            "Varsayılan değerler seçili kum presetinden gelir; isteğe göre değiştirebilirsiniz."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #00ffff; font-size: 11px; font-weight: normal;")
        layout.addWidget(info)

        group = QtWidgets.QGroupBox("Kum Parametreleri")
        form = QtWidgets.QFormLayout(group)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        base = MOLDS.get(body.mold_preset, MOLDS["green_sand"])

        self._afs_spin = QtWidgets.QDoubleSpinBox()
        self._afs_spin.setRange(0.0, 200.0)
        self._afs_spin.setDecimals(1)
        self._afs_spin.setSuffix(" AFS")
        self._afs_spin.setValue(body.mold_afs_grain_size or base.afs_grain_size)
        form.addRow("AFS Tane İnceliği:", self._afs_spin)

        self._moisture_spin = QtWidgets.QDoubleSpinBox()
        self._moisture_spin.setRange(0.0, 30.0)
        self._moisture_spin.setDecimals(1)
        self._moisture_spin.setSuffix(" %")
        self._moisture_spin.setValue(body.mold_moisture_percent or base.moisture_percent)
        form.addRow("Nem Oranı:", self._moisture_spin)

        self._binder_spin = QtWidgets.QDoubleSpinBox()
        self._binder_spin.setRange(0.0, 20.0)
        self._binder_spin.setDecimals(1)
        self._binder_spin.setSuffix(" %")
        self._binder_spin.setValue(body.mold_binder_percent or base.binder_percent)
        form.addRow("Bağlayıcı Oranı:", self._binder_spin)

        self._compact_spin = QtWidgets.QDoubleSpinBox()
        self._compact_spin.setRange(0.0, 100.0)
        self._compact_spin.setDecimals(1)
        self._compact_spin.setSuffix(" %")
        self._compact_spin.setValue(body.mold_compactability_percent or base.compactability_percent)
        form.addRow("Compactability Oranı:", self._compact_spin)

        layout.addWidget(group)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        self._body.mold_afs_grain_size = float(self._afs_spin.value())
        self._body.mold_moisture_percent = float(self._moisture_spin.value())
        self._body.mold_binder_percent = float(self._binder_spin.value())
        self._body.mold_compactability_percent = float(self._compact_spin.value())
        self.accept()


class BodyRowWidget(QtWidgets.QWidget):
    """Single-line body row with inline, type-conditional feeder / mould controls."""

    body_type_changed = QtCore.pyqtSignal(Body, int)
    feeder_type_changed = QtCore.pyqtSignal(Body, str)
    feeder_m_changed = QtCore.pyqtSignal(Body, float)
    mold_settings_changed = QtCore.pyqtSignal(Body)

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
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self._build_ui()
        self._sync_from_body()

    def _build_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(3)

        name_label = QtWidgets.QLabel(self._body.name)
        name_label.setToolTip(
            f"Hacim: {self._body.volume_cm3:.2f} cm³\nMerkez: {self._body.center}"
        )
        name_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        name_label.setStyleSheet(
            "background-color: #27272a; color: #00ffff; border: 1px solid #52525b; "
            "border-radius: 4px; padding: 1px; font-size: 11px; font-weight: bold;"
        )
        name_label.setMaximumWidth(70)
        name_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        layout.addWidget(name_label)

        self._type_combo = QtWidgets.QComboBox()
        self._type_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._type_combo.setMinimumWidth(80)
        self._type_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        for bt, label in self._body_type_names.items():
            self._type_combo.addItem(label, int(bt))
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        layout.addWidget(self._type_combo)

        # --- Feeder controls: compact, hidden unless RISER ---
        self._feeder_type_combo = QtWidgets.QComboBox()
        self._feeder_type_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._feeder_type_combo.setMinimumWidth(80)
        self._feeder_type_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
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
        self._feeder_m_spin.setMaximumWidth(70)
        self._feeder_m_spin.valueChanged.connect(self._on_feeder_m_changed)
        layout.addWidget(self._feeder_m_spin)

        # --- Mould-sand controls: compact, hidden unless CORE ---
        self._sand_type_combo = QtWidgets.QComboBox()
        self._sand_type_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._sand_type_combo.setMinimumWidth(80)
        self._sand_type_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self._sand_type_combo.setToolTip("Kum tipi")
        for key, name in SAND_PRESET_NAMES.items():
            self._sand_type_combo.addItem(name, key)
        self._sand_type_combo.currentIndexChanged.connect(self._on_sand_type_changed)
        layout.addWidget(self._sand_type_combo)

        self._sand_prop_btn = QtWidgets.QPushButton("...")
        self._sand_prop_btn.setMaximumWidth(28)
        self._sand_prop_btn.setToolTip(
            "Kum parametreleri (AFS tane, nem %, bağlayıcı %, compactability %)"
        )
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
        dialog = MoldPropertiesDialog(self._body, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.mold_settings_changed.emit(self._body)
