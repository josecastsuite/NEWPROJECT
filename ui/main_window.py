"""Main PyQt6 application window for JoseCast Analyzer v7.0 Titan."""

import os
import sys
import time
from typing import List, Optional

import pyvista as pv
from PyQt6 import QtCore, QtGui, QtWidgets

from core import (
    MAX_RES,
    analyze,
    analyze_gating,
    apply_unit_scale,
    build_voxel_grid,
    detect_unit_suggestion,
    generate_report,
    get_material,
    load_step,
)
from core.materials import MATERIALS
from core.types import Body, BodyType
from ui.viewer import Analyzer3DViewer


BODY_TYPE_NAMES = {
    BodyType.PART: "PARÇA",
    BodyType.RISER: "BESLEYİCİ",
    BodyType.INGATE: "MEME",
    BodyType.RUNNER: "YOLLUK",
    BodyType.SPRUE: "DÖKÜM AĞZI",
    BodyType.CORE: "MAÇA",
}


class CheckListItem(QtWidgets.QWidget):
    """Row in the checklist panel."""

    def __init__(self, text: str, ok: bool, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        icon = "[OK]" if ok else "[X]"
        color = "#00ff88" if ok else "#ff4444"
        self.label = QtWidgets.QLabel(
            f"<span style='color:{color};font-weight:bold'>{icon}</span> {_escape_html(text)}"
        )
        layout.addWidget(self.label)


def _escape_html(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JoseCast Analyzer v7.0 Titan")
        self.resize(1600, 1000)

        self._bodies: List[Body] = []
        self._analysis = None
        self._grid = None
        self._origin = None
        self._dx = None
        self._unit_scale = 1.0

        self._build_ui()
        self._apply_dark_theme()
        self.aiLog(
            "JOSECAST TITAN ENGINE v7.0 BOOTING... [2040-READY]",
            "info",
        )
        self.aiLog("Siyah AI terminal hazır. Gelecekte LLM bağlantı noktası.", "ok")

    def _apply_dark_theme(self):
        self.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#1e1e1e"))
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#eeeeee"))
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#121212"))
        palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#2d2d30"))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor("#eeeeee"))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor("#eeeeee"))
        palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#eeeeee"))
        palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#3c3f41"))
        palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#eeeeee"))
        palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor("#ff4444"))
        palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#2d72d9"))
        palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#ffffff"))
        self.setPalette(palette)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Left panel
        left_panel = QtWidgets.QWidget()
        left_panel.setFixedWidth(340)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        self.load_btn = QtWidgets.QPushButton("STEP Yükle")
        self.load_btn.clicked.connect(self.on_load_step)
        left_layout.addWidget(self.load_btn)

        left_layout.addWidget(QtWidgets.QLabel("Body Listesi (Tip Ata):"))
        self.body_list = QtWidgets.QListWidget()
        self.body_list.setMinimumHeight(180)
        left_layout.addWidget(self.body_list)

        unit_layout = QtWidgets.QHBoxLayout()
        unit_layout.addWidget(QtWidgets.QLabel("Birim:"))
        self.unit_combo = QtWidgets.QComboBox()
        for unit, label in [("mm", "mm"), ("cm", "cm"), ("m", "m"), ("inch", "inch")]:
            self.unit_combo.addItem(label, unit)
        self.unit_combo.currentIndexChanged.connect(self.on_unit_changed)
        unit_layout.addWidget(self.unit_combo)
        left_layout.addLayout(unit_layout)

        res_layout = QtWidgets.QHBoxLayout()
        res_layout.addWidget(QtWidgets.QLabel("Max çözünürlük:"))
        self.res_spin = QtWidgets.QSpinBox()
        self.res_spin.setRange(160, MAX_RES)
        self.res_spin.setValue(160)
        self.res_spin.setSingleStep(80)
        res_layout.addWidget(self.res_spin)
        left_layout.addLayout(res_layout)

        refine_layout = QtWidgets.QHBoxLayout()
        self.refine_check = QtWidgets.QCheckBox("Yerel adaptive refine (hotspot çevresi)")
        self.refine_check.setChecked(True)
        refine_layout.addWidget(self.refine_check)
        left_layout.addLayout(refine_layout)

        mat_layout = QtWidgets.QHBoxLayout()
        mat_layout.addWidget(QtWidgets.QLabel("Malzeme:"))
        self.mat_combo = QtWidgets.QComboBox()
        for key, mat in MATERIALS.items():
            self.mat_combo.addItem(mat.display_name, key)
        mat_layout.addWidget(self.mat_combo)
        left_layout.addLayout(mat_layout)

        self.voxelize_btn = QtWidgets.QPushButton("Mesh Ata (Voxelize)")
        self.voxelize_btn.setEnabled(False)
        self.voxelize_btn.clicked.connect(self.on_voxelize)
        left_layout.addWidget(self.voxelize_btn)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        left_layout.addWidget(self.progress)

        self.analyze_btn = QtWidgets.QPushButton("Geometrik Analiz Et")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.on_analyze)
        left_layout.addWidget(self.analyze_btn)

        self.status_label = QtWidgets.QLabel("Hazır. STEP dosyası yükleyin.")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)

        # Black AI Terminal
        left_layout.addWidget(QtWidgets.QLabel("AI Terminal:"))
        self.ai_terminal = QtWidgets.QTextEdit()
        self.ai_terminal.setReadOnly(True)
        self.ai_terminal.setMinimumHeight(220)
        self.ai_terminal.setStyleSheet(
            "QTextEdit { background-color: #050505; color: #00ff88; "
            "font-family: Consolas, monospace; font-size: 11px; "
            "border: 1px solid #00ff88; border-radius: 8px; padding: 6px; }"
        )
        left_layout.addWidget(self.ai_terminal)

        self.ai_input = QtWidgets.QLineEdit()
        self.ai_input.setPlaceholderText("> Gelecekte AI komut girişi (şu an devre dışı)")
        self.ai_input.setEnabled(False)
        self.ai_input.setStyleSheet(
            "QLineEdit { background-color: #050505; color: #aaffcc; border: 1px solid #00ff88; }")
        left_layout.addWidget(self.ai_input)

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # Center 3D viewer
        self.viewer = Analyzer3DViewer()
        main_layout.addWidget(self.viewer, stretch=1)

        # Right panel
        right_panel = QtWidgets.QWidget()
        right_panel.setFixedWidth(380)
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        right_layout.addWidget(QtWidgets.QLabel("Kontrol Listesi:"))
        self.checklist_layout = QtWidgets.QVBoxLayout()
        right_layout.addLayout(self.checklist_layout)

        right_layout.addWidget(QtWidgets.QLabel("Öneriler:"))
        self.rec_text = QtWidgets.QTextEdit()
        self.rec_text.setReadOnly(True)
        self.rec_text.setMinimumHeight(160)
        right_layout.addWidget(self.rec_text)

        self.export_btn = QtWidgets.QPushButton("PDF Raporu Export Et")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.on_export_pdf)
        right_layout.addWidget(self.export_btn)

        self.hotspot_toggle = QtWidgets.QCheckBox("Hot Spot'ları Göster")
        self.hotspot_toggle.setChecked(True)
        self.hotspot_toggle.toggled.connect(self.on_toggle_hotspots)
        right_layout.addWidget(self.hotspot_toggle)

        self.risk_toggle = QtWidgets.QCheckBox("Risk Bulutunu Göster")
        self.risk_toggle.setChecked(True)
        self.risk_toggle.toggled.connect(self.on_toggle_risk)
        right_layout.addWidget(self.risk_toggle)

        self.local_toggle = QtWidgets.QCheckBox("Yerel Refine Bölgelerini Göster")
        self.local_toggle.setChecked(False)
        self.local_toggle.toggled.connect(self.on_toggle_local)
        right_layout.addWidget(self.local_toggle)

        slice_layout = QtWidgets.QHBoxLayout()
        self.slice_toggle = QtWidgets.QCheckBox("Kesit Düzlemleri")
        self.slice_toggle.setChecked(False)
        self.slice_toggle.toggled.connect(self.on_toggle_slices)
        slice_layout.addWidget(self.slice_toggle)
        self.slice_field = QtWidgets.QComboBox()
        for field, label in [
            ("sdf", "SDF"),
            ("risk", "Risk"),
            ("niyama", "Niyama"),
            ("mat_id", "Mat ID"),
        ]:
            self.slice_field.addItem(label, field)
        self.slice_field.currentIndexChanged.connect(self.on_slice_field_changed)
        slice_layout.addWidget(self.slice_field)
        right_layout.addLayout(slice_layout)

        right_layout.addStretch()
        main_layout.addWidget(right_panel)

    def aiLog(self, msg: str, type_: str = "info"):
        """Print a line to the black AI terminal."""
        color = {
            "crit": "#ff4444",
            "ok": "#00ff88",
            "info": "#aaffcc",
            "warn": "#ffaa00",
        }.get(type_, "#aaffcc")
        line = f'<span style="color:{color};margin:2px 0">&gt; {_escape_html(msg)}</span>'
        self.ai_terminal.append(line)
        # Auto-scroll
        scrollbar = self.ai_terminal.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _set_progress(self, value: int):
        self.progress.setValue(value)
        QtCore.QCoreApplication.processEvents()

    def _add_body_row(self, body: Body):
        item = QtWidgets.QListWidgetItem()
        widget = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(widget)
        row.setContentsMargins(4, 2, 4, 2)

        label = QtWidgets.QLabel(f"{body.name} ({body.volume_cm3:.2f} cm³)")
        label.setToolTip(f"Merkez: {body.center}")
        row.addWidget(label, stretch=1)

        combo = QtWidgets.QComboBox()
        for bt in (
            BodyType.PART,
            BodyType.RISER,
            BodyType.INGATE,
            BodyType.RUNNER,
            BodyType.SPRUE,
            BodyType.CORE,
        ):
            combo.addItem(BODY_TYPE_NAMES[bt], bt)
        combo.setCurrentIndex(combo.findData(body.body_type))
        combo.currentIndexChanged.connect(
            lambda _, b=body, c=combo: self.on_body_type_changed(b, c)
        )
        row.addWidget(combo)

        item.setSizeHint(widget.sizeHint())
        self.body_list.addItem(item)
        self.body_list.setItemWidget(item, widget)

    def on_load_step(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "STEP Dosyası Seç", "", "STEP Files (*.step *.stp *.STEP *.STP)"
        )
        if not path:
            return
        try:
            self._bodies = load_step(path)
            self.aiLog(f"STEP yüklendi: {os.path.basename(path)}", "ok")
            self.body_list.clear()
            for body in self._bodies:
                self._add_body_row(body)

            # Unit suggestion
            suggested = detect_unit_suggestion(self._bodies)
            idx = self.unit_combo.findData(suggested)
            if idx >= 0:
                self.unit_combo.setCurrentIndex(idx)
            self.aiLog(f"Önerilen birim: {suggested}", "info")

            self.viewer.clear_scene()
            self.viewer.show_bodies(self._bodies)
            self.status_label.setText(
                f"{len(self._bodies)} body yüklendi. Tip atamalarını yapıp voxelize edin."
            )
            self.voxelize_btn.setEnabled(True)
            self.analyze_btn.setEnabled(False)
            self._analysis = None
            self._clear_checklist()
            self.rec_text.clear()
        except Exception as e:
            self.aiLog(f"Yükleme hatası: {e}", "crit")
            QtWidgets.QMessageBox.critical(self, "Yükleme Hatası", str(e))

    def on_unit_changed(self):
        if not self._bodies:
            return
        unit = self.unit_combo.currentData()
        # Reset then apply scale to original? We don't keep originals; scale relative to mm.
        # For simplicity re-load STEP not done; user must reload if unit changed after voxelize.
        if self._grid is not None:
            self.aiLog("Birim değişikliği için STEP'i yeniden yükleyin.", "warn")
            return
        self._unit_scale = apply_unit_scale(self._bodies, unit)
        self.aiLog(f"Birim ölçeği uygulandı: {unit} (x{self._unit_scale:.2f})", "ok")
        self.viewer.show_bodies(self._bodies)

    def on_body_type_changed(self, body: Body, combo: QtWidgets.QComboBox):
        body.body_type = combo.currentData()
        self.viewer.show_bodies(self._bodies)

    def on_voxelize(self):
        if not self._bodies:
            return
        try:
            self.progress.setValue(0)
            self.status_label.setText("Voxelizasyon yapılıyor...")
            self.aiLog("Voxelizasyon başlıyor...", "info")
            self._set_progress(10)
            target_dim = self.res_spin.value()
            grid, origin, dx, bodies = build_voxel_grid(
                self._bodies,
                target_dim=target_dim,
                progress_callback=self._set_progress,
            )
            self._grid = grid
            self._origin = origin
            self._dx = dx
            self._bodies = bodies
            self.progress.setValue(100)
            self.status_label.setText(
                f"Voxel grid hazır: {grid.shape} (dx={dx:.3f} mm)"
            )
            self.aiLog(
                f"Voxel grid hazır: {grid.shape} | dx={dx:.3f} mm | metal voxel={int((grid > 0).sum())}",
                "ok",
            )
            self.analyze_btn.setEnabled(True)
        except Exception as e:
            import traceback

            self.aiLog(f"Voxelizasyon hatası: {e}", "crit")
            QtWidgets.QMessageBox.critical(self, "Voxelizasyon Hatası", f"{e}\n{traceback.format_exc()}")

    def on_analyze(self):
        if self._grid is None:
            return
        try:
            self.status_label.setText("Titan motoru çalışıyor, 2-3 dk sürebilir...")
            self.aiLog("AŞAMA 1/6: SDF haritası oluşturuluyor...", "info")
            self.progress.setValue(0)
            t0 = time.time()

            material_key = self.mat_combo.currentData()
            max_res = self.res_spin.value()
            refine_local = self.refine_check.isChecked()

            self._analysis = analyze(
                self._bodies,
                self._grid,
                self._origin,
                self._dx,
                material_key=material_key,
                base_res=160,
                max_res=max_res,
                refine_local=refine_local,
                progress_callback=self._set_progress,
            )

            self.aiLog("AŞAMA 5/6: Meme / yolluk / döküm ağzı kontrolleri...", "info")
            gate_result = analyze_gating(self._analysis, fill_time_s=10.0)
            self._analysis.gate_result = gate_result
            self._analysis.recommendations.extend(self._gating_recommendations(gate_result))

            elapsed = time.time() - t0
            self.aiLog(f"AŞAMA 6/6: Analiz tamamlandı ({elapsed:.1f} sn)", "ok")

            self.progress.setValue(100)
            self.status_label.setText(
                f"Analiz tamamlandı ({elapsed:.1f} sn). {len(self._analysis.hotspots)} hot spot."
            )
            self.export_btn.setEnabled(True)
            self._update_checklist()
            self._update_recommendations()
            self.viewer.show_hotspots(self._analysis)
            self.viewer.show_risk(self._analysis)
            if self.local_toggle.isChecked():
                self.viewer.show_local_regions(self._analysis, self.slice_field.currentData())
        except Exception as e:
            import traceback

            self.aiLog(f"Analiz hatası: {e}", "crit")
            QtWidgets.QMessageBox.critical(self, "Analiz Hatası", f"{e}\n{traceback.format_exc()}")

    def _gating_recommendations(self, gr) -> List[str]:
        recs = []
        if gr is None:
            return recs
        if not gr.campbell_ok:
            recs.append(
                f"Yolluk (runner) kesiti küçük: toplam meme alanı {gr.total_ingate_contact_area_cm2:.2f} cm², "
                f"yolluk minimum kesit {gr.runner_min_area_cm2:.2f} cm². Yolluk oranını 1:1.5 - 1:2 arası yapın."
            )
        if not gr.bernoulli_ok:
            recs.append(
                f"Döküm ağzı (sprue) taban alanı {gr.sprue_base_area_cm2:.2f} cm² < gerekli {gr.required_sprue_area_cm2:.2f} cm². "
                f"Döküm ağzını büyütün."
            )
        if gr.ingate_on_thick_region:
            recs.append(
                f"Meme kalın bölgede (ortalama M={gr.ingate_avg_m_mm:.2f} mm). "
                f"İnce kesite veya sıcak nokta etrafına taşıyın."
            )
        if gr.campbell_ok and gr.bernoulli_ok and not gr.ingate_on_thick_region:
            recs.append("Meme ve yolluk geometrisi kurallara uygun görünüyor.")
        return recs

    def _clear_checklist(self):
        while self.checklist_layout.count():
            item = self.checklist_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _update_checklist(self):
        self._clear_checklist()
        if self._analysis is None:
            return

        for hs in self._analysis.hotspots:
            text = (
                f"Hot spot M={hs.m_value_mm:.1f} mm - "
                f"besleme mesafesi {hs.dist_to_riser_mm:.1f} mm | "
                f"Niyama={hs.niyama_min:.2f} | Direnç={hs.resistance:.1f}"
            )
            self.checklist_layout.addWidget(CheckListItem(text, hs.feed_ok))

        for rr in self._analysis.riser_results:
            text = f"{rr.name}: M={rr.m_value_mm:.1f} mm (gerekli {rr.effective_m_required:.1f} mm)"
            self.checklist_layout.addWidget(CheckListItem(text, rr.large_enough))

        if self._analysis.gate_result:
            gr = self._analysis.gate_result
            self.checklist_layout.addWidget(
                CheckListItem("Campbell yolluk kontrolü (Ag/Ar)", gr.campbell_ok)
            )
            self.checklist_layout.addWidget(
                CheckListItem("Bernoulli döküm ağzı kontrolü", gr.bernoulli_ok)
            )
            self.checklist_layout.addWidget(
                CheckListItem(
                    "Meme konumu (kalın bölgede olmamalı)",
                    not gr.ingate_on_thick_region,
                )
            )

    def _update_recommendations(self):
        if self._analysis and self._analysis.recommendations:
            self.rec_text.setHtml(
                "<br><br>".join(
                    f"<span style='color:#aaffcc'>&#8226; {_escape_html(r)}</span>"
                    for r in self._analysis.recommendations
                )
            )
        else:
            self.rec_text.setPlainText("Henüz öneri yok.")

    def on_toggle_hotspots(self, checked: bool):
        if self._analysis:
            self.viewer.toggle_hotspots(self._analysis, checked)

    def on_toggle_risk(self, checked: bool):
        if self._analysis:
            self.viewer.toggle_risk(self._analysis, checked)

    def on_toggle_local(self, checked: bool):
        if self._analysis:
            if checked:
                self.viewer.show_local_regions(
                    self._analysis, self.slice_field.currentData()
                )
            else:
                self.viewer.show_local_regions(None, "risk")

    def on_toggle_slices(self, checked: bool):
        if self._analysis:
            self.viewer.toggle_slices(
                self._analysis, checked, self.slice_field.currentData()
            )

    def on_slice_field_changed(self):
        if self._analysis and self.slice_toggle.isChecked():
            self.viewer.toggle_slices(
                self._analysis, False, "sdf"
            )
            self.viewer.toggle_slices(
                self._analysis, True, self.slice_field.currentData()
            )
        if self._analysis and self.local_toggle.isChecked():
            self.viewer.show_local_regions(self._analysis, self.slice_field.currentData())

    def on_export_pdf(self):
        if self._analysis is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "PDF Kaydet", "josecast_rapor.pdf", "PDF Files (*.pdf)"
        )
        if not path:
            return
        try:
            screenshot = os.path.join(os.path.dirname(path), "_josecast_view.png")
            self.viewer.save_screenshot(screenshot)
            generate_report(self._analysis, path, screenshot)
            self.status_label.setText(f"Rapor kaydedildi: {path}")
            self.aiLog(f"PDF raporu kaydedildi: {path}", "ok")
        except Exception as e:
            import traceback

            self.aiLog(f"PDF hatası: {e}", "crit")
            QtWidgets.QMessageBox.critical(
                self, "Export Hatası", f"{e}\n{traceback.format_exc()}"
            )


def main():
    pv.set_plot_theme("dark")
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
