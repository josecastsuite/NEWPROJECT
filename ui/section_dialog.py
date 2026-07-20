"""Simple dialog to pick or override the cross-sectional area of a gating body."""
from typing import Optional, Tuple

import numpy as np
from PyQt6 import QtCore, QtWidgets

from core.gating import (
    _characteristic_cross_section_area,
    _flow_axis,
    _section_2d_area_and_perim,
    _sprue_circular_base_and_throat,
)
from core.types import Body, BodyType


def _axis_name(axis: Tuple[float, float, float]) -> str:
    a = np.round(axis, 3)
    labels = []
    for v, name in zip(a, ["X", "Y", "Z"]):
        if abs(v) > 0.5:
            labels.append(f"{name}{'+' if v > 0 else '-'}")
    return "".join(labels) if labels else str(tuple(a.tolist()))


def _area_along_axis_cm2(body: Body, axis) -> Optional[float]:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm == 0:
        return None
    axis = axis / norm
    centroid = np.asarray(body.mesh.centroid, dtype=np.float64)
    try:
        section = body.mesh.section(plane_origin=centroid, plane_normal=axis)
    except Exception:
        return None
    if section is None or len(section.vertices) < 3:
        return None
    try:
        area_mm2, _ = _section_2d_area_and_perim(section, axis, centroid)
    except Exception:
        # Fallback: compute via 2D convex hull of projected vertices
        pts = np.asarray(section.vertices, dtype=np.float64)
        # Build orthonormal basis for plane perpendicular to axis
        u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(axis[0]) > 0.9:
            u = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = u - np.dot(u, axis) * axis
        u = u / float(np.linalg.norm(u))
        v = np.cross(axis, u)
        coords = np.column_stack((pts @ u, pts @ v))
        from scipy.spatial import ConvexHull
        area_mm2 = float(ConvexHull(coords).volume)
    return area_mm2 / 100.0


class SectionDialog(QtWidgets.QDialog):
    """Let the user choose the cross-sectional area of one gating body.

    The dialog computes several candidate areas:
      * characteristic (recommended) flow cross-section,
      * sections perpendicular to the world X, Y, Z axes,
      * a manually entered value.

    The selected value (cm²) is returned through ``area_cm2`` and ``section_key``.
    """

    def __init__(
        self,
        body: Body,
        section_key: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"{body.name} - Kesit Alanı Seçimi")
        self.resize(420, 320)
        self.setWindowFlags(
            self.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint
        )
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self.body = body
        self.section_key = section_key or body.section_key or self._default_key_for(body)
        self.area_cm2: Optional[float] = None
        self.initial_area_cm2: Optional[float] = body.section_area_cm2 if body.section_area_cm2 and body.section_area_cm2 > 0 else None

        self._build_ui()
        self._compute_candidates()

    @staticmethod
    def _default_key_for(body: Body) -> str:
        try:
            bt = BodyType(body.body_type) if isinstance(body.body_type, int) else body.body_type
        except Exception:
            return "INGATE"
        if bt == BodyType.SPRUE:
            return "SPRUE_BASE"
        if bt == BodyType.SPRUE_THROAT:
            return "SPRUE_THROAT"
        if bt == BodyType.RUNNER:
            return "RUNNER"
        if bt == BodyType.INGATE:
            return "INGATE"
        return "INGATE"

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        try:
            bt = BodyType(self.body.body_type) if isinstance(self.body.body_type, int) else self.body.body_type
            type_name = bt.name if isinstance(bt, BodyType) else str(bt)
        except Exception:
            type_name = str(self.body.body_type)

        info = QtWidgets.QLabel(
            f"<b>{self.body.name}</b> ({type_name})<br>"
            f"Hacim: {self.body.volume_cm3:.2f} cm³<br>"
            "Lütfen kesit alanını seçin veya elle girin:"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.key_combo = QtWidgets.QComboBox()
        for label, key in [
            ("Meme (INGATE)", "INGATE"),
            ("Yolluk (RUNNER)", "RUNNER"),
            ("Döküm ağzı tabanı (SPRUE_BASE)", "SPRUE_BASE"),
            ("Döküm ağzı boğazı (SPRUE_THROAT)", "SPRUE_THROAT"),
        ]:
            self.key_combo.addItem(label, key)
        idx = self.key_combo.findData(self.section_key)
        if idx >= 0:
            self.key_combo.setCurrentIndex(idx)
        layout.addWidget(self.key_combo)

        self.candidate_group = QtWidgets.QButtonGroup(self)
        self.candidate_widgets: list[Tuple[QtWidgets.QRadioButton, float]] = []

        self.char_radio = QtWidgets.QRadioButton("Karakteristik (önerilen): hesaplanıyor...")
        self.char_radio.setChecked(True)
        self.candidate_group.addButton(self.char_radio)
        layout.addWidget(self.char_radio)

        self.x_radio = QtWidgets.QRadioButton("X düzlemine dik kesit: ---")
        self.candidate_group.addButton(self.x_radio)
        layout.addWidget(self.x_radio)

        self.y_radio = QtWidgets.QRadioButton("Y düzlemine dik kesit: ---")
        self.candidate_group.addButton(self.y_radio)
        layout.addWidget(self.y_radio)

        self.z_radio = QtWidgets.QRadioButton("Z düzlemine dik kesit: ---")
        self.candidate_group.addButton(self.z_radio)
        layout.addWidget(self.z_radio)

        manual_layout = QtWidgets.QHBoxLayout()
        self.manual_radio = QtWidgets.QRadioButton("Manuel:")
        self.candidate_group.addButton(self.manual_radio)
        self.manual_spin = QtWidgets.QDoubleSpinBox()
        self.manual_spin.setRange(0.0, 1e6)
        self.manual_spin.setDecimals(4)
        self.manual_spin.setValue(0.0)
        self.manual_spin.setSuffix(" cm²")
        manual_layout.addWidget(self.manual_radio)
        manual_layout.addWidget(self.manual_spin, stretch=1)
        layout.addLayout(manual_layout)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _compute_candidates(self):
        try:
            axis = _flow_axis(self.body.mesh)

            # Characteristic flow area
            try:
                if self.section_key in ("SPRUE_BASE", "SPRUE_THROAT"):
                    try:
                        bt = BodyType(self.body.body_type) if isinstance(self.body.body_type, int) else self.body.body_type
                    except Exception:
                        bt = BodyType.INGATE
                    if bt == BodyType.SPRUE_THROAT:
                        # A dedicated throat body: use its characteristic cross-section directly.
                        char_mm2 = _characteristic_cross_section_area(self.body.mesh, axis)
                    else:
                        base_mm2, throat_mm2 = _sprue_circular_base_and_throat(
                            self.body.mesh, axis
                        )
                        char_mm2 = base_mm2 if self.section_key == "SPRUE_BASE" else throat_mm2
                else:
                    char_mm2 = _characteristic_cross_section_area(self.body.mesh, axis)
                self.char_area = char_mm2 / 100.0
            except Exception:
                self.char_area = None

            self.x_area = _area_along_axis_cm2(self.body, (1.0, 0.0, 0.0))
            self.y_area = _area_along_axis_cm2(self.body, (0.0, 1.0, 0.0))
            self.z_area = _area_along_axis_cm2(self.body, (0.0, 0.0, 1.0))

            self._set_radio_text(self.char_radio, "Karakteristik (önerilen)", self.char_area)
            self._set_radio_text(self.x_radio, "X düzlemine dik kesit", self.x_area)
            self._set_radio_text(self.y_radio, "Y düzlemine dik kesit", self.y_area)
            self._set_radio_text(self.z_radio, "Z düzlemine dik kesit", self.z_area)

            # Pre-fill manual spin with the recommended value for convenience
            if self.char_area is not None and self.char_area > 0:
                self.manual_spin.setValue(self.char_area)
        except Exception:
            # Never let a calculation failure block the dialog; manual entry is always available.
            self.char_area = None
            self.x_area = None
            self.y_area = None
            self.z_area = None
            for radio in (self.char_radio, self.x_radio, self.y_radio, self.z_radio):
                radio.setText(radio.text().split(":")[0] + ": hesaplanamadı")
                radio.setEnabled(False)

        # Manual is always available; if no computed candidate, select it.
        self.manual_radio.setEnabled(True)
        self.manual_spin.setEnabled(True)
        if self.char_area is None:
            self.manual_radio.setChecked(True)

        self._restore_selection()

    def _set_radio_text(self, radio: QtWidgets.QRadioButton, label: str, value: Optional[float]):
        if value is not None and value > 0:
            d_mm = np.sqrt(4.0 * value * 100.0 / np.pi)
            radio.setText(f"{label}: A = {value:.4f} cm² (≈ Ø {d_mm:.2f} mm)")
            radio.setEnabled(True)
            radio.setToolTip("Bu değeri seçmek için tıklayın")
        else:
            radio.setText(f"{label}: hesaplanamadı")
            radio.setEnabled(False)

    def _restore_selection(self):
        """Pre-select the radio that matches the previously stored area, if any."""
        if self.initial_area_cm2 is None:
            if self.char_area is not None and self.char_area > 0:
                self.char_radio.setChecked(True)
            else:
                self.manual_radio.setChecked(True)
            return

        stored = float(self.initial_area_cm2)
        candidates = [
            (self.char_radio, self.char_area),
            (self.x_radio, self.x_area),
            (self.y_radio, self.y_area),
            (self.z_radio, self.z_area),
        ]
        for radio, value in candidates:
            if value is not None and value > 0 and abs(value - stored) <= 1e-4:
                radio.setChecked(True)
                return
        # No candidate matched: it must have been a manual value.
        self.manual_radio.setChecked(True)
        self.manual_spin.setValue(stored)

    def _on_ok(self):
        if self.char_radio.isChecked():
            self.area_cm2 = self.char_area
        elif self.x_radio.isChecked():
            self.area_cm2 = self.x_area
        elif self.y_radio.isChecked():
            self.area_cm2 = self.y_area
        elif self.z_radio.isChecked():
            self.area_cm2 = self.z_area
        elif self.manual_radio.isChecked():
            self.area_cm2 = self.manual_spin.value()
        else:
            self.area_cm2 = None

        self.section_key = self.key_combo.currentData()
        if self.area_cm2 is None or self.area_cm2 <= 0:
            QtWidgets.QMessageBox.warning(self, "Hata", "Lütfen geçerli bir kesit alanı seçin.")
            return
        self.accept()
