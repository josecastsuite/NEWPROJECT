"""Reusable mould-sand property dialog.

Used from the main window (global mould) and from the per-CORE body row.
Values are loaded from ``MOLDS`` presets; the user can edit and save the
preset back to ``core/materials_data/molds.json``.
"""
from typing import Optional

from PyQt6 import QtCore, QtWidgets

from core.materials import MOLDS, MoldMaterial, save_molds
from core.types import Body


class MoldPropertiesDialog(QtWidgets.QDialog):
    """Edit AFS, moisture, binder, compactability and rigidity for a sand preset.

    Parameters can be saved back to ``molds.json`` via the *Kaydet* button,
    or applied without persisting via *Tamam*.  *İptal* discards changes.
    """

    saved = QtCore.pyqtSignal(str)

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        body: Optional[Body] = None,
        preset_key: Optional[str] = None,
        title: str = "Kum Parametreleri",
    ):
        super().__init__(parent)
        self._body = body
        self._initial_key = preset_key or (body.mold_preset if body else "green_sand")
        self.setWindowTitle(title)
        self.setMinimumWidth(340)

        self.setStyleSheet(
            """
            QDialog { background-color: #18181b; }
            QLabel { color: #00ffff; font-weight: 800; font-size: 13px; }
            QGroupBox {
                color: #00ffff;
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
                color: #00ffff;
                font-weight: bold;
            }
            QDoubleSpinBox, QComboBox {
                background: #27272a;
                color: #00ffff;
                border: 1px solid #52525b;
                border-radius: 5px;
                padding: 5px;
                min-height: 22px;
                font-weight: bold;
            }
            QPushButton {
                background: #27272a;
                color: #00ffff;
                border: 1px solid #00ffff;
                border-radius: 6px;
                padding: 8px 14px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background: #3f3f46; }
            """
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        header = QtWidgets.QLabel(
            body.name if body else "Kalıp kumu parametrelerini seçin ve düzenleyin."
        )
        header.setStyleSheet("color: #00ffff; font-size: 13px; font-weight: bold;")
        layout.addWidget(header)

        info = QtWidgets.QLabel(
            "Kayıtlı kum tipinden seçim yapın, isteğe göre değiştirin; "
            "Kaydet butonu değişiklikleri JSON kütüphanesine yazar."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #00ffff; font-size: 11px; font-weight: normal;")
        layout.addWidget(info)

        preset_layout = QtWidgets.QHBoxLayout()
        preset_label = QtWidgets.QLabel("Kum tipi:")
        preset_layout.addWidget(preset_label)
        self._preset_combo = QtWidgets.QComboBox()
        for key, mold in MOLDS.items():
            if getattr(mold, "is_sand", True):
                self._preset_combo.addItem(mold.name, key)
        idx = self._preset_combo.findData(self._initial_key)
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)
        self._preset_combo.currentIndexChanged.connect(self._load_preset)
        preset_layout.addWidget(self._preset_combo, 1)
        layout.addLayout(preset_layout)

        group = QtWidgets.QGroupBox("Kum Parametreleri")
        form = QtWidgets.QFormLayout(group)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        self._afs_spin = QtWidgets.QDoubleSpinBox()
        self._afs_spin.setRange(0.0, 200.0)
        self._afs_spin.setDecimals(1)
        self._afs_spin.setSuffix(" AFS")
        form.addRow("AFS Tane İnceliği:", self._afs_spin)

        self._moisture_spin = QtWidgets.QDoubleSpinBox()
        self._moisture_spin.setRange(0.0, 30.0)
        self._moisture_spin.setDecimals(1)
        self._moisture_spin.setSuffix(" %")
        form.addRow("Nem Oranı:", self._moisture_spin)

        self._binder_spin = QtWidgets.QDoubleSpinBox()
        self._binder_spin.setRange(0.0, 20.0)
        self._binder_spin.setDecimals(1)
        self._binder_spin.setSuffix(" %")
        form.addRow("Bağlayıcı Oranı:", self._binder_spin)

        self._compact_spin = QtWidgets.QDoubleSpinBox()
        self._compact_spin.setRange(0.0, 100.0)
        self._compact_spin.setDecimals(1)
        self._compact_spin.setSuffix(" %")
        form.addRow("Compactability Oranı:", self._compact_spin)

        self._rigidity_spin = QtWidgets.QDoubleSpinBox()
        self._rigidity_spin.setRange(0.0, 1.0)
        self._rigidity_spin.setDecimals(2)
        self._rigidity_spin.setSuffix(" (0=yumuşak, 1=rijit)")
        form.addRow("Kalıp Rijitliği:", self._rigidity_spin)

        layout.addWidget(group)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()
        self._cancel_btn = QtWidgets.QPushButton("İptal")
        self._cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self._cancel_btn)
        self._ok_btn = QtWidgets.QPushButton("Tamam")
        self._ok_btn.clicked.connect(self._on_ok)
        btn_layout.addWidget(self._ok_btn)
        self._save_btn = QtWidgets.QPushButton("Kaydet")
        self._save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(self._save_btn)
        layout.addLayout(btn_layout)

        self._load_preset()

    def _current_preset(self) -> str:
        return self._preset_combo.currentData() or "green_sand"

    def _load_preset(self, _index: int = -1) -> None:
        key = self._current_preset()
        mold = MOLDS.get(key) or MOLDS.get("green_sand") or next(iter(MOLDS.values()))
        base = mold  # type: MoldMaterial

        # If the dialog is tied to a body and the body is using this preset,
        # prefer the body's stored overrides.
        if self._body and self._body.mold_preset == key:
            afs = self._body.mold_afs_grain_size or base.afs_grain_size
            moisture = self._body.mold_moisture_percent or base.moisture_percent
            binder = self._body.mold_binder_percent or base.binder_percent
            compact = self._body.mold_compactability_percent or base.compactability_percent
            rigidity = (
                self._body.mold_rigidity_factor
                if self._body.mold_rigidity_factor > 0.0
                else base.mold_rigidity_factor
            )
        else:
            afs = base.afs_grain_size
            moisture = base.moisture_percent
            binder = base.binder_percent
            compact = base.compactability_percent
            rigidity = base.mold_rigidity_factor

        self._afs_spin.setValue(afs)
        self._moisture_spin.setValue(moisture)
        self._binder_spin.setValue(binder)
        self._compact_spin.setValue(compact)
        self._rigidity_spin.setValue(rigidity)

    def _read_values(self) -> dict:
        return {
            "afs_grain_size": float(self._afs_spin.value()),
            "moisture_percent": float(self._moisture_spin.value()),
            "binder_percent": float(self._binder_spin.value()),
            "compactability_percent": float(self._compact_spin.value()),
            "mold_rigidity_factor": float(self._rigidity_spin.value()),
        }

    def _apply_to_body(self) -> None:
        if self._body is None:
            return
        vals = self._read_values()
        self._body.mold_preset = self._current_preset()
        self._body.mold_afs_grain_size = vals["afs_grain_size"]
        self._body.mold_moisture_percent = vals["moisture_percent"]
        self._body.mold_binder_percent = vals["binder_percent"]
        self._body.mold_compactability_percent = vals["compactability_percent"]
        self._body.mold_rigidity_factor = vals["mold_rigidity_factor"]

    def _on_ok(self) -> None:
        self._apply_to_body()
        self.accept()

    def _on_save(self) -> None:
        key = self._current_preset()
        vals = self._read_values()
        if key in MOLDS:
            from dataclasses import replace

            MOLDS[key] = replace(MOLDS[key], **vals)
        save_molds()
        self._apply_to_body()
        self.saved.emit(key)
        self.accept()
