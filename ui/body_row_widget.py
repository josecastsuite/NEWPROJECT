"""Per-body row widget for the JoséCast main window body list.

A body row shows the body name, a body-type selector, and an optional
properties stack that appears when the body type needs extra parameters.
Currently only ``RISER`` exposes feeder-type / modulus / note fields;
other body types keep the stack empty so the list stays compact.
"""
from typing import Dict, Optional

from PyQt6 import QtCore, QtWidgets

from core.types import Body, BodyType


FEEDER_TYPE_NAMES = {
    "conventional": "Konvansiyonel kum kalıp besleyici",
    "exothermic": "Ekzotermik gömleklı besleyici",
    "insulated": "İzole gömleklı besleyici",
    "sleeve": "Seramik / manyetik gömlek besleyici",
    "chilled": "Soğutucu çelik / chill besleyici",
    "side": "Yan besleyici (side riser)",
    "blind": "Kör besleyici (blind riser)",
}


class BodyRowWidget(QtWidgets.QWidget):
    """One row in the body list with dynamic, body-type aware controls."""

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
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(2, 1, 2, 1)
        main_layout.setSpacing(2)

        # ---- top row: name + body type ----
        top_row = QtWidgets.QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(4)

        name_label = QtWidgets.QLabel(self._body.name)
        name_label.setToolTip(
            f"Hacim: {self._body.volume_cm3:.2f} cm³\nMerkez: {self._body.center}"
        )
        name_label.setStyleSheet("font-size: 11px;")
        name_label.setMaximumWidth(90)
        name_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        top_row.addWidget(name_label)

        self._type_combo = QtWidgets.QComboBox()
        self._type_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._type_combo.setMaximumWidth(130)
        for bt, label in self._body_type_names.items():
            self._type_combo.addItem(label, int(bt))
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        top_row.addWidget(self._type_combo)

        self._feeder_summary = QtWidgets.QLabel("")
        self._feeder_summary.setStyleSheet("font-size: 10px; color: #00ff88;")
        self._feeder_summary.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        top_row.addWidget(self._feeder_summary)

        main_layout.addLayout(top_row)

        # ---- dynamic properties stack ----
        self._stack = QtWidgets.QStackedWidget()
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._stack.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )

        # page 0: empty (most body types)
        self._empty_page = QtWidgets.QWidget()
        self._stack.addWidget(self._empty_page)

        # page 1: riser properties
        self._riser_page = QtWidgets.QWidget()
        riser_layout = QtWidgets.QFormLayout(self._riser_page)
        riser_layout.setContentsMargins(8, 2, 2, 2)
        riser_layout.setSpacing(2)

        self._feeder_type_combo = QtWidgets.QComboBox()
        self._feeder_type_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._feeder_type_combo.setMinimumWidth(160)
        for key, name in FEEDER_TYPE_NAMES.items():
            self._feeder_type_combo.addItem(name, key)
        self._feeder_type_combo.currentIndexChanged.connect(self._on_feeder_type_changed)
        riser_layout.addRow("Besleyici tipi:", self._feeder_type_combo)

        self._feeder_m_spin = QtWidgets.QDoubleSpinBox()
        self._feeder_m_spin.setRange(0.0, 999.0)
        self._feeder_m_spin.setDecimals(2)
        self._feeder_m_spin.setSuffix(" cm")
        self._feeder_m_spin.setSpecialValueText("Otomatik")
        self._feeder_m_spin.valueChanged.connect(self._on_feeder_m_changed)
        riser_layout.addRow("Modül (M):", self._feeder_m_spin)

        self._feeder_note_edit = QtWidgets.QLineEdit()
        self._feeder_note_edit.setPlaceholderText("Not (örn: exotermik %40)")
        self._feeder_note_edit.textChanged.connect(self._on_feeder_note_changed)
        riser_layout.addRow("Not:", self._feeder_note_edit)

        self._stack.addWidget(self._riser_page)

        main_layout.addWidget(self._stack)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def body(self) -> Body:
        return self._body

    def set_body_type(self, body_type: BodyType) -> None:
        idx = self._type_combo.findData(int(body_type))
        if idx >= 0 and idx != self._type_combo.currentIndex():
            self._type_combo.setCurrentIndex(idx)
        self._update_stack(body_type)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
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
            self._update_summary()
        finally:
            self._block_updates = False

    def _update_stack(self, body_type: BodyType) -> None:
        if body_type == BodyType.RISER:
            self._stack.setCurrentWidget(self._riser_page)
            self._stack.setVisible(True)
        else:
            self._stack.setCurrentWidget(self._empty_page)
            self._stack.setVisible(False)

    def _update_summary(self) -> None:
        if self._body.body_type != BodyType.RISER or not self._body.feeder_type:
            self._feeder_summary.setText("")
            return
        short = {
            "conventional": "konv",
            "exothermic": "ekzo",
            "insulated": "izol",
            "sleeve": "göm",
            "chilled": "chill",
            "side": "yan",
            "blind": "kör",
        }.get(self._body.feeder_type, self._body.feeder_type[:4])
        m_text = (
            f" M={self._body.feeder_m_mm / 10.0:.1f}cm"
            if self._body.feeder_m_mm > 0
            else " auto"
        )
        self._feeder_summary.setText(f"[{short}{m_text}]")

    # ------------------------------------------------------------------
    # slots
    # ------------------------------------------------------------------
    def _on_type_changed(self, index: int) -> None:
        if self._block_updates:
            return
        data = self._type_combo.itemData(index)
        try:
            new_type = BodyType(data)
        except Exception:
            new_type = BodyType.PART
        self._body.body_type = new_type
        self._update_stack(new_type)
        if new_type != BodyType.RISER:
            self._body.feeder_type = ""
            self._body.feeder_m_mm = 0.0
            self._body.feeder_note = ""
            self._sync_from_body()
        self._update_summary()
        self.body_type_changed.emit(self._body, int(new_type))

    def _on_feeder_type_changed(self, index: int) -> None:
        if self._block_updates:
            return
        key = self._feeder_type_combo.itemData(index) or "conventional"
        self._body.feeder_type = key
        self._update_summary()
        self.feeder_type_changed.emit(self._body, key)

    def _on_feeder_m_changed(self, value: float) -> None:
        if self._block_updates:
            return
        self._body.feeder_m_mm = float(value) * 10.0
        self._update_summary()
        self.feeder_m_changed.emit(self._body, self._body.feeder_m_mm)

    def _on_feeder_note_changed(self, text: str) -> None:
        if self._block_updates:
            return
        self._body.feeder_note = text.strip()
        self.feeder_note_changed.emit(self._body, self._body.feeder_note)
