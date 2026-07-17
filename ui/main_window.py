"""Main PyQt6 application window for JoseCast Analyzer v5.0."""

import os
import sys
from typing import List, Optional

import pyvista as pv
from PyQt6 import QtCore, QtGui, QtWidgets

from core import analyze, analyze_gating, build_voxel_grid, generate_report, load_step
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
        icon = "✓" if ok else "✗"
        color = "#4CAF50" if ok else "#F44336"
        self.label = QtWidgets.QLabel(f"<span style='color:{color};font-weight:bold'>{icon}</span> {text}")
        layout.addWidget(self.label)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JoseCast Analyzer v5.0")
        self.resize(1400, 900)

        self._bodies: List[Body] = []
        self._analysis = None
        self._grid = None
        self._origin = None
        self._dx = None

        self._build_ui()
        self._apply_dark_theme()

    def _apply_dark_theme(self):
        self.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#2b2b2b"))
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#eeeeee"))
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#1e1e1e"))
        palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#3c3f41"))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor("#eeeeee"))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor("#eeeeee"))
        palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#eeeeee"))
        palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#3c3f41"))
        palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#eeeeee"))
        palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor("#ff0000"))
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
        left_panel.setFixedWidth(320)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        self.load_btn = QtWidgets.QPushButton("STEP Yükle")
        self.load_btn.clicked.connect(self.on_load_step)
        left_layout.addWidget(self.load_btn)

        left_layout.addWidget(QtWidgets.QLabel("Body Listesi (Tip Ata):"))
        self.body_list = QtWidgets.QListWidget()
        self.body_list.setMinimumHeight(200)
        left_layout.addWidget(self.body_list)

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

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # Center 3D viewer
        self.viewer = Analyzer3DViewer()
        main_layout.addWidget(self.viewer, stretch=1)

        # Right panel
        right_panel = QtWidgets.QWidget()
        right_panel.setFixedWidth(360)
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        right_layout.addWidget(QtWidgets.QLabel("Kontrol Listesi:"))
        self.checklist_layout = QtWidgets.QVBoxLayout()
        right_layout.addLayout(self.checklist_layout)

        right_layout.addWidget(QtWidgets.QLabel("Öneriler:"))
        self.rec_text = QtWidgets.QTextEdit()
        self.rec_text.setReadOnly(True)
        self.rec_text.setMinimumHeight(200)
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
        self.risk_toggle.toggled.connect(self.on_toggle_risk)
        right_layout.addWidget(self.risk_toggle)

        right_layout.addStretch()
        main_layout.addWidget(right_panel)

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
        for bt in (BodyType.PART, BodyType.RISER, BodyType.INGATE, BodyType.RUNNER, BodyType.SPRUE, BodyType.CORE):
            combo.addItem(BODY_TYPE_NAMES[bt], bt)
        combo.setCurrentIndex(combo.findData(body.body_type))
        combo.currentIndexChanged.connect(lambda _, b=body, c=combo: self.on_body_type_changed(b, c))
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
            self.body_list.clear()
            for body in self._bodies:
                self._add_body_row(body)
            self.viewer.clear_scene()
            self.viewer.show_bodies(self._bodies)
            self.status_label.setText(f"{len(self._bodies)} body yüklendi. Tip atamalarını yapıp voxelize edin.")
            self.voxelize_btn.setEnabled(True)
            self.analyze_btn.setEnabled(False)
            self._analysis = None
            self._clear_checklist()
            self.rec_text.clear()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Yükleme Hatası", str(e))

    def on_body_type_changed(self, body: Body, combo: QtWidgets.QComboBox):
        body.body_type = combo.currentData()
        self.viewer.show_bodies(self._bodies)

    def on_voxelize(self):
        if not self._bodies:
            return
        try:
            self.progress.setValue(0)
            self.status_label.setText("Voxelizasyon yapılıyor...")
            self._set_progress(10)
            grid, origin, dx, bodies = build_voxel_grid(
                self._bodies, target_dim=96, progress_callback=self._set_progress
            )
            self._grid = grid
            self._origin = origin
            self._dx = dx
            self._bodies = bodies
            self.progress.setValue(100)
            self.status_label.setText(f"Voxel grid hazır: {grid.shape} (dx={dx:.3f} mm)")
            self.analyze_btn.setEnabled(True)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Voxelizasyon Hatası", str(e))

    def on_analyze(self):
        if self._grid is None:
            return
        try:
            self.status_label.setText("Geometrik analiz çalışıyor...")
            self.progress.setValue(0)

            self._analysis = analyze(
                self._bodies,
                self._grid,
                self._origin,
                self._dx,
                progress_callback=self._set_progress,
            )

            # Gating analysis
            gate_result = analyze_gating(self._analysis, fill_time_s=10.0)
            self._analysis.gate_result = gate_result
            self._analysis.recommendations.extend(self._gating_recommendations(gate_result))

            self.progress.setValue(100)
            self.status_label.setText("Analiz tamamlandı.")
            self.export_btn.setEnabled(True)
            self._update_checklist()
            self._update_recommendations()
            self.viewer.show_hotspots(self._analysis)
            self.viewer.show_risk(self._analysis)
        except Exception as e:
            import traceback
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
            text = f"Hot spot M={hs.m_value_mm:.1f} mm - besleme mesafesi {hs.dist_to_riser_mm:.1f} mm"
            self.checklist_layout.addWidget(CheckListItem(text, hs.feed_ok))

        for rr in self._analysis.riser_results:
            text = f"{rr.name}: M={rr.m_value_mm:.1f} mm (gerekli {rr.target_hotspot_m_mm * 1.2:.1f} mm)"
            self.checklist_layout.addWidget(CheckListItem(text, rr.large_enough))

        if self._analysis.gate_result:
            gr = self._analysis.gate_result
            self.checklist_layout.addWidget(CheckListItem(
                f"Campbell yolluk kontrolü (Ag/Ar)", gr.campbell_ok))
            self.checklist_layout.addWidget(CheckListItem(
                f"Bernoulli döküm ağzı kontrolü", gr.bernoulli_ok))
            self.checklist_layout.addWidget(CheckListItem(
                f"Meme konumu (kalın bölgede olmamalı)", not gr.ingate_on_thick_region))

    def _update_recommendations(self):
        if self._analysis and self._analysis.recommendations:
            self.rec_text.setPlainText("\n\n".join(f"• {r}" for r in self._analysis.recommendations))
        else:
            self.rec_text.setPlainText("Henüz öneri yok.")

    def on_toggle_hotspots(self, checked: bool):
        if self._analysis:
            self.viewer.toggle_hotspots(self._analysis, checked)

    def on_toggle_risk(self, checked: bool):
        if self._analysis:
            self.viewer.toggle_risk(self._analysis, checked)

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
        except Exception as e:
            import traceback
            QtWidgets.QMessageBox.critical(self, "Export Hatası", f"{e}\n{traceback.format_exc()}")


def main():
    pv.set_plot_theme("dark")
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
