"""Main PyQt6 application window for JoseCast Analyzer v8.0 Titan."""

import os
import sys
import time
import webbrowser
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
from PyQt6 import QtCore, QtGui, QtWidgets

from core import (
    MAX_RES,
    ALLOYS,
    MOLDS,
    analyze,
    apply_unit_scale,
    build_voxel_grid,
    detect_unit_suggestion,
    generate_report,
    get_alloy,
    get_mold,
    load_step,
)
from core.materials import chvorinov_c_from_properties, make_effective_mold
from core.types import Body, BodyType, CastingParameters
from ui.body_row_widget import BodyRowWidget, FEEDER_TYPE_NAMES
from ui.mold_properties_dialog import MoldPropertiesDialog
from ui.section_dialog import SectionDialog
from ui.viewer import Analyzer3DViewer


BODY_TYPE_NAMES = {
    BodyType.PART: "PARÇA",
    BodyType.RISER: "BESLEYİCİ",
    BodyType.INGATE: "MEME",
    BodyType.RUNNER: "YOLLUK",
    BodyType.SPRUE: "DÖKÜM AĞZI",
    BodyType.CORE: "MAÇA",
    BodyType.COOLING_SPRUE: "SOĞUTUCU D.AĞZI",
    BodyType.FILTER: "FİLTRE",
    BodyType.POURING_BASIN: "DÖKÜM HAVZASI",
    BodyType.SPRUE_THROAT: "D.AĞZI BOĞAZI",
    BodyType.DISTRIBUTOR: "DAĞITICI",
    BodyType.CURUFLUK: "CURUFLUK",
    BodyType.CHILL: "SOĞUTUCU",
}


def _escape_html(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


class AnalyzeThread(QtCore.QThread):
    """Run the heavy analyze() call in a background thread."""

    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(self, parent, analyze_fn, args, kwargs):
        super().__init__(parent)
        self._analyze_fn = analyze_fn
        self._args = args
        self._kwargs = kwargs
        self._kwargs["progress_callback"] = self.progress.emit

    def run(self):
        try:
            result = self._analyze_fn(*self._args, **self._kwargs)
            self.finished.emit(result)
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")

    def __del__(self):
        self.wait(100)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JoseCast Analyzer v8.0 Titan")
        self.resize(1800, 1100)

        self._bodies: List[Body] = []
        self._analysis = None
        self._grid = None
        self._origin = None
        self._dx = None
        self._unit_scale = 1.0
        # User-selected section area for the velocity inlet (sprue throat/top).
        self._user_section_area_cm2 = 0.0
        self._user_section_key = "SPRUE_THROAT"
        self._user_section_body_name = ""
        self._body_items: Dict[str, QtWidgets.QListWidgetItem] = {}
        self._body_rows: Dict[str, BodyRowWidget] = {}


        self._build_ui()
        self._apply_light_theme()
        self._sync_casting_params_from_materials()
        self.aiLog(
            "JOSECAST TITAN ENGINE v8.0 BOOTING... [2040-READY]",
            "info",
        )
        self.aiLog("Siyah AI terminal hazır. Gelecekte LLM bağlantı noktası.", "ok")

    def _apply_light_theme(self):
        """Apply the #5 light blue-gray theme requested by the user."""
        self.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#E2E8F0"))
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#1E293B"))
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#F8FAFC"))
        palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#F1F5F9"))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor("#FFFFFF"))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor("#1E293B"))
        palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#334155"))
        palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#3B82F6"))
        palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#FFFFFF"))
        palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor("#EF4444"))
        palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#3B82F6"))
        palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#FFFFFF"))
        self.setPalette(palette)

        self.setStyleSheet(
            """
            QMainWindow { background: #E2E8F0; }
            QSplitter, QScrollArea, QScrollArea > QWidget { background: #E2E8F0; }
            QGroupBox {
                background: #FFFFFF;
                color: #334155;
                font-weight: 600;
                font-size: 12px;
                border: 1px solid #94A3B8;
                border-radius: 10px;
                margin-top: 14px;
                padding-top: 18px;
                padding-left: 12px;
                padding-right: 12px;
                padding-bottom: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                top: -2px;
                color: #1E293B;
                font-weight: 700;
                background: transparent;
            }
            QPushButton {
                background: #3B82F6; color: #FFFFFF; border: none;
                border-radius: 8px; padding: 8px 16px; font-weight: 600; font-size: 12px;
            }
            QPushButton:hover { background: #2563EB; }
            QPushButton:disabled { background: #CBD5E1; color: #64748B; }
            QPushButton:pressed { background: #1D4ED8; }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                background: #F8FAFC; color: #1E293B; border: 1px solid #94A3B8;
                border-radius: 6px; padding: 5px; min-height: 22px;
            }
            QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
                border: 1.5px solid #3B82F6;
            }
            QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled {
                background: #E2E8F0; color: #94A3B8; border: 1px solid #CBD5E1;
            }
            QProgressBar {
                background: #F8FAFC; border: 1px solid #94A3B8; border-radius: 6px;
                text-align: center; color: #1E293B; font-weight: 600;
            }
            QProgressBar::chunk { background: #3B82F6; border-radius: 5px; }
            QLabel { color: #334155; font-weight: 500; font-size: 12px; }
            QListWidget {
                background: #FFFFFF; border: 1px solid #94A3B8; border-radius: 8px;
                color: #334155; padding: 4px;
            }
            QListWidget::item { padding: 4px; border-radius: 4px; }
            QListWidget::item:selected { background: #DBEAFE; color: #1E293B; }
            QTextEdit {
                background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 8px;
                color: #334155; font-family: 'Consolas', monospace; font-size: 12px;
            }
            QCheckBox { color: #334155; spacing: 8px; font-weight: 500; font-size: 12px; }
            QCheckBox::indicator {
                width: 16px; height: 16px;
                border: 1px solid #94A3B8; border-radius: 4px;
                background: #F8FAFC;
            }
            QCheckBox::indicator:checked { background: #3B82F6; border: 1px solid #3B82F6; }
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical { background: #E2E8F0; width: 10px; border-radius: 5px; }
            QScrollBar::handle:vertical { background: #94A3B8; border-radius: 5px; }
            QScrollBar::handle:vertical:hover { background: #64748B; }
            QSlider::groove:horizontal { height: 6px; background: #CBD5E1; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #3B82F6; border-radius: 3px; }
            QSlider::handle:horizontal { background: #FFFFFF; border: 1px solid #94A3B8; width: 14px; height: 14px; border-radius: 7px; }
            QToolTip { background: #FFFFFF; color: #1E293B; border: 1px solid #CBD5E1; padding: 4px; border-radius: 4px; }
            """
        )

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        
        main_vbox = QtWidgets.QVBoxLayout(central)
        main_vbox.setContentsMargins(12, 12, 12, 12)
        main_vbox.setSpacing(10)

        # ---------------- TOP AREA (Splitter) ----------------
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # ---------------- LEFT PANEL (scrollable) ----------------
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(360)

        left_panel = QtWidgets.QWidget()
        left_scroll.setWidget(left_panel)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        left_layout.setSpacing(8)
        left_layout.setContentsMargins(10, 10, 10, 10)

        # File & body group
        file_group = QtWidgets.QGroupBox("1. STEP ve Body")
        file_layout = QtWidgets.QVBoxLayout(file_group)

        self.load_btn = QtWidgets.QPushButton("STEP Yükle")
        self.load_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton))
        self.load_btn.clicked.connect(self.on_load_step)
        file_layout.addWidget(self.load_btn)

        file_layout.addWidget(QtWidgets.QLabel("Body Listesi (Tip Ata):"))
        self.body_list = QtWidgets.QListWidget()
        self.body_list.setMinimumHeight(140)
        self.body_list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.body_list.currentRowChanged.connect(self.on_body_row_selected)
        file_layout.addWidget(self.body_list)
        left_layout.addWidget(file_group)

        # Settings group
        settings_group = QtWidgets.QGroupBox("2. Analiz Ayarları")
        settings_layout = QtWidgets.QVBoxLayout(settings_group)
        settings_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        settings_layout.setSpacing(6)

        def _settings_labeled(widget, label_text, tooltip=None):
            lbl = QtWidgets.QLabel(label_text)
            lbl.setWordWrap(True)
            lbl.setMinimumHeight(20) 
            lbl.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
            settings_layout.addWidget(lbl)
            if tooltip:
                widget.setToolTip(tooltip)
            settings_layout.addWidget(widget)
            widget._label = lbl

        self.unit_combo = QtWidgets.QComboBox()
        for unit, label in [("mm", "mm"), ("cm", "cm"), ("m", "m"), ("inch", "inch")]:
            self.unit_combo.addItem(label, unit)
        self.unit_combo.currentIndexChanged.connect(self.on_unit_changed)
        _settings_labeled(self.unit_combo, "Birim:")

        self.res_spin = QtWidgets.QSpinBox()
        self.res_spin.setRange(160, MAX_RES)
        self.res_spin.setValue(400)
        self.res_spin.setSingleStep(80)
        _settings_labeled(self.res_spin, "Max çözünürlük:", "160 = hızlı, 2040 = Titan mod (yavaş, yerel refine).")

        self.refine_check = QtWidgets.QCheckBox("Yerel adaptive refine")
        self.refine_check.setChecked(True)
        settings_layout.addWidget(self.refine_check)

        self.subvox_spin = QtWidgets.QSpinBox()
        self.subvox_spin.setRange(1, 3)
        self.subvox_spin.setValue(2)
        _settings_labeled(self.subvox_spin, "Sub-voxel faktör:", "Kenar vokseli kısmi saymak için 2x/3x upsample.")

        self.thermal_spin = QtWidgets.QSpinBox()
        self.thermal_spin.setRange(10, 2000)
        self.thermal_spin.setValue(300)
        self.thermal_spin.setSingleStep(50)
        _settings_labeled(self.thermal_spin, "Max soğuma süresi (sn):", "3-D transient entalpi çözücüsü için maksimum katılaşma süresi (sn).")

        self.alloy_combo = QtWidgets.QComboBox()
        for key, alloy in ALLOYS.items():
            self.alloy_combo.addItem(alloy.name, key)
        self.alloy_combo.currentIndexChanged.connect(self._sync_casting_params_from_materials)
        _settings_labeled(self.alloy_combo, "Alaşım:")

        self.mold_type_combo = QtWidgets.QComboBox()
        mold_category_names = {
            "sand": "Kum Kalıp",
            "metal": "Metal Kalıp",
            "ceramic": "Seramik Kalıp",
        }
        mold_category_order = ["sand", "metal", "ceramic"]
        for cat in mold_category_order:
            key = next((k for k, v in MOLDS.items() if v.mold_type == cat), None)
            if key:
                self.mold_type_combo.addItem(mold_category_names.get(cat, cat.capitalize()), key)
        self.mold_type_combo.currentIndexChanged.connect(self._on_mold_type_changed)
        _settings_labeled(self.mold_type_combo, "Kalıp tipi:")

        self.sand_type_combo = QtWidgets.QComboBox()
        for key, mold in MOLDS.items():
            if getattr(mold, "is_sand", True) and key != "sand":
                self.sand_type_combo.addItem(mold.name, key)
        self.sand_type_combo.currentIndexChanged.connect(self._on_sand_type_changed)
        _settings_labeled(self.sand_type_combo, "Kum tipi:")

        self.mold_props_btn = QtWidgets.QPushButton("Kum Parametrelerini Düzenle")
        self.mold_props_btn.setToolTip("Seçili kum tipinin AFS, nem, bağlayıcı, compactability ve rijitlik değerlerini düzenle ve JSON kütüphanesine kaydet")
        self.mold_props_btn.clicked.connect(self._on_mold_properties)
        _settings_labeled(self.mold_props_btn, "Kum parametreleri:")

        # v10.3: global mould-sand property overrides
        self.mold_afs_spin = QtWidgets.QDoubleSpinBox()
        self.mold_afs_spin.setRange(0.0, 200.0)
        self.mold_afs_spin.setDecimals(1)
        self.mold_afs_spin.setSuffix(" AFS")
        _settings_labeled(self.mold_afs_spin, "AFS tane inceliği:")

        self.mold_moisture_spin = QtWidgets.QDoubleSpinBox()
        self.mold_moisture_spin.setRange(0.0, 30.0)
        self.mold_moisture_spin.setDecimals(1)
        self.mold_moisture_spin.setSuffix(" %")
        _settings_labeled(self.mold_moisture_spin, "Nem oranı:")

        self.mold_binder_spin = QtWidgets.QDoubleSpinBox()
        self.mold_binder_spin.setRange(0.0, 20.0)
        self.mold_binder_spin.setDecimals(1)
        self.mold_binder_spin.setSuffix(" %")
        _settings_labeled(self.mold_binder_spin, "Bağlayıcı oranı:")

        self.mold_compactability_spin = QtWidgets.QDoubleSpinBox()
        self.mold_compactability_spin.setRange(0.0, 100.0)
        self.mold_compactability_spin.setDecimals(1)
        self.mold_compactability_spin.setSuffix(" %")
        _settings_labeled(self.mold_compactability_spin, "Compactability:")

        self.mold_rigidity_spin = QtWidgets.QDoubleSpinBox()
        self.mold_rigidity_spin.setRange(-1.0, 1.0)
        self.mold_rigidity_spin.setDecimals(2)
        self.mold_rigidity_spin.setValue(-1.0)
        self.mold_rigidity_spin.setSuffix(" (-1=auto)")
        _settings_labeled(self.mold_rigidity_spin, "Kalıp rijitliği:")

        # Set defaults after combos exist; block signals to avoid partial sync.
        self.alloy_combo.blockSignals(True)
        self.mold_type_combo.blockSignals(True)
        self.sand_type_combo.blockSignals(True)
        self.alloy_combo.setCurrentIndex(list(ALLOYS.keys()).index("42CrMo4"))
        mold_default_idx = self.mold_type_combo.findData("sand")
        self.mold_type_combo.setCurrentIndex(mold_default_idx if mold_default_idx >= 0 else 0)
        sand_default_idx = self.sand_type_combo.findData("green_sand")
        self.sand_type_combo.setCurrentIndex(sand_default_idx if sand_default_idx >= 0 else 0)
        self.alloy_combo.blockSignals(False)
        self.mold_type_combo.blockSignals(False)
        self.sand_type_combo.blockSignals(False)
        self._update_sand_type_visibility()

        left_layout.addWidget(settings_group)

        # Casting parameters group
        params_group = QtWidgets.QGroupBox("3. Döküm Parametreleri")
        params_layout = QtWidgets.QVBoxLayout(params_group)
        params_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        params_layout.setSpacing(6)

        def _params_labeled(widget, label_text, tooltip=None):
            lbl = QtWidgets.QLabel(label_text)
            lbl.setWordWrap(True)
            lbl.setMinimumHeight(20) 
            lbl.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
            params_layout.addWidget(lbl)
            if tooltip:
                widget.setToolTip(tooltip)
            params_layout.addWidget(widget)

        self.t_pour_spin = QtWidgets.QDoubleSpinBox()
        self.t_pour_spin.setRange(0, 2000)
        self.t_pour_spin.setDecimals(1)
        self.t_pour_spin.setValue(1600.0)
        _params_labeled(self.t_pour_spin, "Döküm sıcaklığı T_pour (°C):")

        self.t_liq_spin = QtWidgets.QDoubleSpinBox()
        self.t_liq_spin.setRange(0, 2000)
        self.t_liq_spin.setDecimals(1)
        self.t_liq_spin.setValue(1510.0)
        _params_labeled(self.t_liq_spin, "Liquidus T_liq (°C):")

        self.t_sol_spin = QtWidgets.QDoubleSpinBox()
        self.t_sol_spin.setRange(0, 2000)
        self.t_sol_spin.setDecimals(1)
        self.t_sol_spin.setValue(1410.0)
        _params_labeled(self.t_sol_spin, "Solidus T_sol (°C):")

        self.t_mold_spin = QtWidgets.QDoubleSpinBox()
        self.t_mold_spin.setRange(-50, 500)
        self.t_mold_spin.setDecimals(1)
        self.t_mold_spin.setValue(25.0)
        _params_labeled(self.t_mold_spin, "Kalıp sıcaklığı T_mold (°C):")

        self.t_fill_spin = QtWidgets.QDoubleSpinBox()
        self.t_fill_spin.setRange(0.0, 300.0)
        self.t_fill_spin.setDecimals(1)
        self.t_fill_spin.setValue(0.0)
        self.t_fill_spin.setSpecialValueText("Otomatik")
        _params_labeled(self.t_fill_spin, "Döküm süresi t_fill (s):")

        self.rho_spin = QtWidgets.QDoubleSpinBox()
        self.rho_spin.setRange(100, 20000)
        self.rho_spin.setDecimals(1)
        self.rho_spin.setValue(7000.0)
        self.rho_spin.setSingleStep(100)
        _params_labeled(self.rho_spin, "Sıvı yoğunluk ρ (kg/m³):")

        self.visc_spin = QtWidgets.QDoubleSpinBox()
        self.visc_spin.setRange(0.0001, 10.0)
        self.visc_spin.setDecimals(4)
        self.visc_spin.setValue(0.0060)
        _params_labeled(self.visc_spin, "Viskozite μ (Pa·s):")

        self.velocity_section_combo = QtWidgets.QComboBox()
        self.velocity_section_combo.addItem("Döküm ağzı boğazı (sprue throat)", "SPRUE_THROAT")
        self.velocity_section_combo.addItem("Döküm ağzı en üst noktası (sprue top)", "SPRUE_BASE")
        self.velocity_section_combo.setCurrentIndex(self.velocity_section_combo.findData("SPRUE_THROAT"))
        _params_labeled(self.velocity_section_combo, "Hız kesiti:", "Seçilen sprue kesitinin hızı girilir; program bu kesit alanından Q hesaplar, düzeltme yapmaz.")

        self.section_pick_button = QtWidgets.QPushButton("Kesit seçiniz")
        self.section_pick_button.setToolTip("Seçili sprue elemanının gerçek kesit alanını 3D modelden seç. 0 = otomatik CAD ölçümü.")
        self.section_pick_button.clicked.connect(self.on_pick_section)
        self.section_pick_label = QtWidgets.QLabel("A=otomatik")

        pick_layout = QtWidgets.QHBoxLayout()
        pick_layout.addWidget(self.section_pick_button)
        pick_layout.addWidget(self.section_pick_label)
        pick_layout.addStretch()
        params_layout.addLayout(pick_layout)

        self.v_ingate_spin = QtWidgets.QDoubleSpinBox()
        self.v_ingate_spin.setRange(0.0, 20.0)
        self.v_ingate_spin.setDecimals(2)
        self.v_ingate_spin.setValue(0.0)
        self.v_ingate_spin.setSingleStep(0.1)
        _params_labeled(
            self.v_ingate_spin,
            "Giriş hızı v (m/s):",
            "0 = otomatik (tasarım debisi). >0 kullanıcı girişi; seçili sprue kesitinde geçerlidir. Program düzeltmez.",
        )

        self.h_eff_spin = QtWidgets.QDoubleSpinBox()
        self.h_eff_spin.setRange(0.0, 10.0)
        self.h_eff_spin.setDecimals(2)
        self.h_eff_spin.setValue(0.0)
        self.h_eff_spin.setSingleStep(0.05)
        self.h_eff_spin.setSpecialValueText("Otomatik")
        _params_labeled(
            self.h_eff_spin,
            "H_eff (m):",
            "0 = otomatik. Etkin metal yüksekliği (m). Şimdilik bağlanmadı.",
        )

        self.fast_flow_chk = QtWidgets.QCheckBox("Hızlı akış hesabı (animasyon yok)")
        self.fast_flow_chk.setToolTip(
            "İşaretlenirse 3-B Darcy/VOF çözümü atlanır; sadece Q=vA ile düğüm hızları "
            "hesaplanır. Animasyon çalışmaz. Varsayılan: kapalı (tam simülasyon)."
        )
        self.fast_flow_chk.setChecked(False)
        params_layout.addWidget(self.fast_flow_chk)

        left_layout.addWidget(params_group)

        # Gravity direction group
        gravity_group = QtWidgets.QGroupBox("Yerçekimi Yönü")
        gravity_layout = QtWidgets.QVBoxLayout(gravity_group)
        gravity_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        gravity_layout.setSpacing(6)
        self.gravity_combo = QtWidgets.QComboBox()
        for label, value in [
            ("Aşağı (-Z)", "0,0,-1"),
            ("Yukarı (+Z)", "0,0,1"),
            ("Ön (-Y)", "0,-1,0"),
            ("Arka (+Y)", "0,1,0"),
            ("Sol (-X)", "-1,0,0"),
            ("Sağ (+X)", "1,0,0"),
            ("Özel", "custom"),
        ]:
            self.gravity_combo.addItem(label, value)
        self.gravity_combo.setCurrentIndex(0)
        self.gravity_custom = QtWidgets.QLineEdit()
        self.gravity_custom.setPlaceholderText("x,y,z (örn: 0.0,-1.0,0.0)")
        self.gravity_custom.setEnabled(False)
        self.gravity_combo.currentIndexChanged.connect(self._on_gravity_preset_changed)
        gravity_layout.addWidget(self.gravity_combo)
        gravity_layout.addWidget(self.gravity_custom)
        left_layout.addWidget(gravity_group)

        # Sync casting parameter defaults now that all parameter spin boxes exist.
        self._sync_casting_params_from_materials()

        # Actions group
        actions_group = QtWidgets.QGroupBox("4. Motor")
        actions_layout = QtWidgets.QVBoxLayout(actions_group)

        self.voxelize_btn = QtWidgets.QPushButton("Mesh Ata (Voxelize)")
        self.voxelize_btn.setEnabled(False)
        self.voxelize_btn.clicked.connect(self.on_voxelize)
        actions_layout.addWidget(self.voxelize_btn)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        actions_layout.addWidget(self.progress)

        self.analyze_btn = QtWidgets.QPushButton("Geometrik Analiz Et")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.on_analyze)
        actions_layout.addWidget(self.analyze_btn)

        self.status_label = QtWidgets.QLabel("Hazır. STEP dosyası yükleyin.")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(40) 
        actions_layout.addWidget(self.status_label)
        left_layout.addWidget(actions_group)
        left_layout.addStretch()

        # ---------------- CENTER 3D VIEWER ----------------
        self.viewer = Analyzer3DViewer()
        self.viewer.flow_animator.frameChanged.connect(self._on_flow_frame_changed)
        self.viewer.flow_animator.stateChanged.connect(self._on_flow_state_changed)

        # ---------------- RIGHT PANEL (scrollable) ----------------
        right_scroll = QtWidgets.QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(380) 
        right_panel = QtWidgets.QWidget()
        right_scroll.setWidget(right_panel)
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(10, 10, 10, 10)

        rec_group = QtWidgets.QGroupBox("Mühendis Önerileri")
        rec_inner = QtWidgets.QVBoxLayout(rec_group)
        self.rec_text = QtWidgets.QTextEdit()
        self.rec_text.setReadOnly(True)
        self.rec_text.setMinimumHeight(160)
        rec_inner.addWidget(self.rec_text)
        right_layout.addWidget(rec_group)

        vis_group = QtWidgets.QGroupBox("Görselleştirme")
        vis_layout = QtWidgets.QVBoxLayout(vis_group)

        self.hotspot_toggle = QtWidgets.QCheckBox("Hot Spot")
        self.hotspot_toggle.setToolTip("Hot spot kürelerini göster/gizle")
        self.hotspot_toggle.setChecked(True)
        self.hotspot_toggle.toggled.connect(self.on_toggle_hotspots)
        vis_layout.addWidget(self.hotspot_toggle)

        self.risk_toggle = QtWidgets.QCheckBox("Risk Bulutu")
        self.risk_toggle.setToolTip("Porozite risk bulutunu göster/gizle")
        self.risk_toggle.setChecked(False)
        self.risk_toggle.toggled.connect(self.on_toggle_risk)
        vis_layout.addWidget(self.risk_toggle)

        self.porosity_toggle = QtWidgets.QCheckBox("Porozite Bulutu")
        self.porosity_toggle.setToolTip("Yüksek porozite riski hacimsel bulut")
        self.porosity_toggle.setChecked(True)
        self.porosity_toggle.toggled.connect(self.on_toggle_porosity)
        vis_layout.addWidget(self.porosity_toggle)

        self.porosity_noise_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.porosity_noise_slider.setMinimum(0)      # 0.00%
        self.porosity_noise_slider.setMaximum(10000)   # 100.00%
        self.porosity_noise_slider.setValue(10000)     # default: tüm sınıf
        self.porosity_noise_slider.setToolTip("Porozite bulutunda görünür risk yüzdesi: 100 = sınıfın tamamı, 0 = sadece en yüksek risk")
        self.porosity_noise_slider.valueChanged.connect(self.on_porosity_noise_changed)
        self.porosity_noise_label = QtWidgets.QLabel("Risk: %100.00")
        vis_layout.addWidget(self.porosity_noise_label)
        vis_layout.addWidget(self.porosity_noise_slider)

        self.porosity_size_filter = QtWidgets.QComboBox()
        self.porosity_size_filter.addItem("Tüm poroziteler", "all")
        self.porosity_size_filter.addItem("Makro (>1000 µm)", "macro")
        self.porosity_size_filter.addItem("Mikro (100–1000 µm)", "micro")
        self.porosity_size_filter.addItem("İnce (<100 µm)", "fine")
        self.porosity_size_filter.setToolTip("Gösterilecek gözenek boyutu sınıfı")
        self.porosity_size_filter.currentIndexChanged.connect(self.on_porosity_size_filter_changed)
        vis_layout.addWidget(self.porosity_size_filter)

        self.niyama_toggle = QtWidgets.QCheckBox("Niyama İzosurface")
        self.niyama_toggle.setToolTip("Niyama 0.775 / 1.5 izoyüzeyleri")
        self.niyama_toggle.setChecked(False)
        self.niyama_toggle.toggled.connect(self.on_toggle_niyama)
        vis_layout.addWidget(self.niyama_toggle)

        self.mold_wall_toggle = QtWidgets.QCheckBox("Kalıp Şişmesi Riski")
        self.mold_wall_toggle.setToolTip("Grafit genleşmesinin kalıp duvarını dışarı ittiği bölgeleri göster")
        self.mold_wall_toggle.setChecked(False)
        self.mold_wall_toggle.toggled.connect(self.on_toggle_mold_wall_movement)
        vis_layout.addWidget(self.mold_wall_toggle)

        self.cold_shot_toggle = QtWidgets.QCheckBox("Soğuk Birleşme Riski")
        self.cold_shot_toggle.setToolTip("Düşük sıcaklık ve yavaş cephe hızından kaynaklanan soğuk birleşme riski")
        self.cold_shot_toggle.setChecked(False)
        self.cold_shot_toggle.toggled.connect(self.on_toggle_cold_shot_risk)
        vis_layout.addWidget(self.cold_shot_toggle)

        self.erosion_toggle = QtWidgets.QCheckBox("Kalıp Erozyonu Riski")
        self.erosion_toggle.setToolTip("Yüksek metal hızına bağlı kum kalıp erozyon riski")
        self.erosion_toggle.setChecked(False)
        self.erosion_toggle.toggled.connect(self.on_toggle_erosion_risk)
        vis_layout.addWidget(self.erosion_toggle)

        self.air_entrapment_toggle = QtWidgets.QCheckBox("Hava Sıkışması")
        self.air_entrapment_toggle.setToolTip("LBM/VOF serbest yüzey çözücüsünün bulduğu kapanmış hava ceplerini göster")
        self.air_entrapment_toggle.setChecked(False)
        self.air_entrapment_toggle.toggled.connect(self.on_toggle_air_entrapment)
        vis_layout.addWidget(self.air_entrapment_toggle)

        anim_group = QtWidgets.QGroupBox("Akış & Katılaşma")
        anim_layout = QtWidgets.QVBoxLayout(anim_group)

        self.flow_anim_toggle = QtWidgets.QCheckBox("Dolum + Katılaşma")
        self.flow_anim_toggle.setToolTip("İki fazlı animasyon: önce dolum, sonra katılaşma")
        self.flow_anim_toggle.setChecked(False)
        self.flow_anim_toggle.toggled.connect(self.on_toggle_flow_animation)
        anim_layout.addWidget(self.flow_anim_toggle)

        play_layout = QtWidgets.QHBoxLayout()
        self.flow_play_btn = QtWidgets.QPushButton("▶ Oynat")
        self.flow_play_btn.setEnabled(False)
        self.flow_play_btn.clicked.connect(self.on_flow_play_clicked)
        play_layout.addWidget(self.flow_play_btn)

        self.flow_time_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.flow_time_slider.setMinimum(0)
        self.flow_time_slider.setMaximum(1000)
        self.flow_time_slider.setValue(0)
        self.flow_time_slider.setEnabled(False)
        self.flow_time_slider.valueChanged.connect(self.on_flow_time_changed)
        play_layout.addWidget(self.flow_time_slider)
        anim_layout.addLayout(play_layout)

        self.flow_time_label = QtWidgets.QLabel("t: 0.000 s / 0.000 s")
        self.flow_time_label.setEnabled(False)
        anim_layout.addWidget(self.flow_time_label)

        speed_layout = QtWidgets.QHBoxLayout()
        speed_label = QtWidgets.QLabel("Hız:")
        self.flow_speed_spin = QtWidgets.QDoubleSpinBox()
        self.flow_speed_spin.setRange(0.01, 20.0)
        self.flow_speed_spin.setValue(1.0)
        self.flow_speed_spin.setSingleStep(0.1)
        self.flow_speed_spin.setDecimals(2)
        self.flow_speed_spin.setSuffix("x")
        self.flow_speed_spin.valueChanged.connect(self.on_flow_speed_changed)
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.flow_speed_spin)

        count_label = QtWidgets.QLabel("Kare:")
        self.flow_particle_label = QtWidgets.QLabel("—")
        self.flow_particle_label.setToolTip("Dolum + katılaşma kare sayısı")
        speed_layout.addWidget(count_label)
        speed_layout.addWidget(self.flow_particle_label)
        anim_layout.addLayout(speed_layout)

        self.flow_surface_check = QtWidgets.QCheckBox("Akış Yolları")
        self.flow_surface_check.setToolTip("Akış yollarını ve ilerleyen marker'ları göster/gizle")
        self.flow_surface_check.setChecked(True)
        self.flow_surface_check.setEnabled(False)
        self.flow_surface_check.toggled.connect(self.on_flow_surface_toggled)
        anim_layout.addWidget(self.flow_surface_check)

        vis_layout.addWidget(anim_group)

        self.path_toggle = QtWidgets.QCheckBox("Besleme Yolları")
        self.path_toggle.setToolTip("Hot spot'tan besleyiciye/gating'e giden yol")
        self.path_toggle.setChecked(True)
        self.path_toggle.toggled.connect(self.on_toggle_feeding_paths)
        vis_layout.addWidget(self.path_toggle)

        self.local_toggle = QtWidgets.QCheckBox("Yerel Refine")
        self.local_toggle.setToolTip("Yerel adaptive refine bölgelerini göster/gizle")
        self.local_toggle.setChecked(False)
        self.local_toggle.toggled.connect(self.on_toggle_local)
        vis_layout.addWidget(self.local_toggle)

        slice_layout = QtWidgets.QHBoxLayout()
        self.slice_toggle = QtWidgets.QCheckBox("Kesit")
        self.slice_toggle.setToolTip("Kesit düzlemlerini göster/gizle")
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
        self.slice_field.setMaximumWidth(130)
        slice_layout.addWidget(self.slice_field)
        vis_layout.addLayout(slice_layout)
        right_layout.addWidget(vis_group)

        export_group = QtWidgets.QGroupBox("Rapor")
        export_layout = QtWidgets.QVBoxLayout(export_group)
        self.export_btn = QtWidgets.QPushButton("PDF Raporu Kaydet")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.on_export_pdf)
        export_layout.addWidget(self.export_btn)

        self.html_btn = QtWidgets.QPushButton("HTML Raporu Tarayıcıda Aç")
        self.html_btn.setEnabled(False)
        self.html_btn.clicked.connect(self.on_view_html_report)
        export_layout.addWidget(self.html_btn)
        right_layout.addWidget(export_group)
        right_layout.addStretch()

        # Add to splitter
        splitter.addWidget(left_scroll)
        splitter.addWidget(self.viewer)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([400, 860, 560])

        # Sync porosity size-filter labels with the default alloy.
        self._update_porosity_filter_labels(get_alloy(self.alloy_combo.currentData()))

        # Add splitter to main VBox
        main_vbox.addWidget(splitter, stretch=1)

        # ---------------- BOTTOM AREA: AI TERMINAL ----------------
        terminal_group = QtWidgets.QGroupBox("AI Asistan")
        terminal_layout = QtWidgets.QVBoxLayout(terminal_group)

        self.ai_terminal = QtWidgets.QTextEdit()
        self.ai_terminal.setReadOnly(True)
        self.ai_terminal.setMaximumHeight(90)
        terminal_layout.addWidget(self.ai_terminal)

        self.ai_input = QtWidgets.QLineEdit()
        self.ai_input.setPlaceholderText("> Komut girin...")
        terminal_layout.addWidget(self.ai_input)

        main_vbox.addWidget(terminal_group, stretch=0)

    def _current_mold_key(self) -> str:
        mtype_key = self.mold_type_combo.currentData()
        if mtype_key:
            try:
                if get_mold(mtype_key).mold_type == "sand":
                    return self.sand_type_combo.currentData() or mtype_key
            except Exception:
                pass
        return mtype_key or "sand"

    def _update_sand_type_visibility(self):
        mtype_key = self.mold_type_combo.currentData()
        is_sand = False
        if mtype_key:
            try:
                is_sand = get_mold(mtype_key).mold_type == "sand"
            except Exception:
                pass
        self.sand_type_combo.setVisible(is_sand)
        if hasattr(self.sand_type_combo, "_label"):
            self.sand_type_combo._label.setVisible(is_sand)
        # v10.3: hide sand-property overrides when not a sand mould
        for widget in (
            self.sand_type_combo,
            self.mold_props_btn,
            self.mold_afs_spin,
            self.mold_moisture_spin,
            self.mold_binder_spin,
            self.mold_compactability_spin,
            self.mold_rigidity_spin,
        ):
            widget.setVisible(is_sand)
            if hasattr(widget, "_label"):
                widget._label.setVisible(is_sand)

    def _on_mold_type_changed(self):
        self._update_sand_type_visibility()
        self._sync_mold_params_from_preset()
        self._sync_casting_params_from_materials()

    def _on_sand_type_changed(self):
        self._sync_mold_params_from_preset()
        self._sync_casting_params_from_materials()

    def _on_mold_properties(self):
        """Open the reusable sand-property dialog for the selected sand preset."""
        preset = self._current_mold_key()
        try:
            if get_mold(preset).mold_type != "sand":
                preset = self.sand_type_combo.currentData() or "green_sand"
        except Exception:
            preset = "green_sand"
        dialog = MoldPropertiesDialog(
            parent=self,
            body=None,
            preset_key=preset,
            title="Kum Parametreleri",
        )
        dialog.saved.connect(self._sync_mold_params_from_preset)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._sync_mold_params_from_preset()
            self._sync_casting_params_from_materials()

    def _sync_mold_params_from_preset(self):
        """Set global mould-sand overrides from the selected preset."""
        try:
            mold = get_mold(self._current_mold_key())
        except Exception:
            return
        self.mold_afs_spin.setValue(mold.afs_grain_size)
        self.mold_moisture_spin.setValue(mold.moisture_percent)
        self.mold_binder_spin.setValue(mold.binder_percent)
        self.mold_compactability_spin.setValue(mold.compactability_percent)
        self.mold_rigidity_spin.setValue(mold.mold_rigidity_factor)

    def _sync_casting_params_from_materials(self):
        """Set parameter defaults from the selected alloy and mould."""
        alloy = get_alloy(self.alloy_combo.currentData())
        mold = get_mold(self._current_mold_key())
        self.t_pour_spin.setValue(alloy.t_pour_c)
        self.t_liq_spin.setValue(alloy.t_liquidus_c)
        self.t_sol_spin.setValue(alloy.t_solidus_c)
        self.t_mold_spin.setValue(mold.t0_c)
        self.rho_spin.setValue(alloy.rho_kg_m3)
        self.visc_spin.setValue(alloy.viscosity_pa_s)

    def _casting_params_from_ui(self) -> CastingParameters:
        gstr = self.gravity_combo.currentData() or "0,0,-1"
        gravity_direction = tuple(float(x.strip()) for x in gstr.split(","))
        return CastingParameters(
            t_pour_c=self.t_pour_spin.value(),
            t_liquidus_c=self.t_liq_spin.value(),
            t_solidus_c=self.t_sol_spin.value(),
            t_mold_c=self.t_mold_spin.value(),
            t_fill_s=self.t_fill_spin.value(),
            rho_liquid_kg_m3=self.rho_spin.value(),
            viscosity_pa_s=self.visc_spin.value(),
            ingate_velocity_m_s=self.v_ingate_spin.value(),
            velocity_section_key=self.velocity_section_combo.currentData(),
            gravity_direction=gravity_direction,
            h_eff_m=self.h_eff_spin.value(),
            fast_flow=self.fast_flow_chk.isChecked(),
            mold_afs_grain_size=self.mold_afs_spin.value(),
            mold_moisture_percent=self.mold_moisture_spin.value(),
            mold_binder_percent=self.mold_binder_spin.value(),
            mold_compactability_percent=self.mold_compactability_spin.value(),
            mold_rigidity_factor=self.mold_rigidity_spin.value(),
        )

    def _gravity_vector_from_ui(self) -> Tuple[float, float, float]:
        data = self.gravity_combo.currentData()
        if data == "custom":
            text = self.gravity_custom.text().strip()
            if text:
                try:
                    parts = [float(x.strip()) for x in text.split(",")]
                    if len(parts) == 3:
                        v = np.array(parts, dtype=np.float64)
                        norm = float(np.linalg.norm(v))
                        if norm > 0:
                            return tuple((v / norm).tolist())
                except Exception:
                    pass
            return (0.0, 0.0, -1.0)
        return tuple(float(x) for x in data.split(","))

    def _on_gravity_preset_changed(self):
        is_custom = self.gravity_combo.currentData() == "custom"
        self.gravity_custom.setEnabled(is_custom)
        if not is_custom:
            self.gravity_custom.clear()

    def aiLog(self, msg: str, type_: str = "info"):
        """Print a line to the AI terminal."""
        color = {
            "crit": "#EF4444",
            "ok": "#10B981",
            "info": "#3B82F6",
            "warn": "#F59E0B",
        }.get(type_, "#3B82F6")
        line = (
            f'<span style="color:{color};margin:2px 0;font-family:Consolas,monospace;"'
            f'>&gt; {_escape_html(msg)}</span>'
        )
        self.ai_terminal.append(line)
        scrollbar = self.ai_terminal.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _set_progress(self, value: int):
        self.progress.setValue(value)

    def _add_body_row(self, body: Body):
        """Add a body row with a dynamic, body-type aware property panel."""
        item = QtWidgets.QListWidgetItem()
        widget = BodyRowWidget(body, BODY_TYPE_NAMES)

        widget.body_type_changed.connect(self.on_body_type_changed)
        widget.body_focused.connect(self.on_body_focused)
        widget.body_unfocused.connect(self.on_body_unfocused)
        widget.feeder_type_changed.connect(self.on_body_feeder_type_changed)
        widget.feeder_m_changed.connect(self.on_body_feeder_m_changed)
        widget.mold_settings_changed.connect(self.on_body_mold_changed)

        self._body_items[body.name] = item
        self._body_rows[body.name] = widget

        self.body_list.addItem(item)
        self.body_list.setItemWidget(item, widget)
        item.setSizeHint(
            QtCore.QSize(
                widget.sizeHint().width(),
                max(widget.sizeHint().height(), widget.minimumSizeHint().height()) + 6,
            )
        )

    def on_load_step(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "STEP Dosyası Seç", "", "STEP Files (*.step *.stp *.STEP *.STP)"
        )
        if not path:
            return
        try:
            self._bodies = load_step(path)
            self.aiLog(f"STEP yüklendi: {os.path.basename(path)}", "ok")
            self._body_rows.clear()
            self._body_items.clear()
            self.body_list.clear()
            for body in self._bodies:
                self._add_body_row(body)

            suggested = detect_unit_suggestion(self._bodies)
            idx = self.unit_combo.findData(suggested)
            if idx >= 0:
                self.unit_combo.setCurrentIndex(idx)
            self.viewer.clear_scene()
            self.viewer.show_bodies(self._bodies)
            self.status_label.setText(
                f"{len(self._bodies)} body yüklendi. Tip atamalarını yapıp voxelize edin."
            )
            self.voxelize_btn.setEnabled(True)
            self.analyze_btn.setEnabled(False)
            self._analysis = None

            self.rec_text.clear()
            self._grid = None
            self._origin = None
            self._dx = None
        except Exception as e:
            self.aiLog(f"Yükleme hatası: {e}", "crit")
            QtWidgets.QMessageBox.critical(self, "Yükleme Hatası", str(e))

    def on_unit_changed(self):
        if not self._bodies:
            return
        unit = self.unit_combo.currentData()
        if self._grid is not None:
            self.aiLog("Birim değişikliği için STEP'i yeniden yükleyin.", "warn")
            return
        self._unit_scale = apply_unit_scale(self._bodies, unit)
        self.viewer.show_bodies(self._bodies)

    def on_body_type_changed(self, body: Body, body_type_value: int):
        """A body row changed type; update the 3-D preview."""
        try:
            new_type = BodyType(body_type_value)
        except Exception:
            new_type = BodyType.PART
        body.body_type = new_type
        if new_type != BodyType.RISER:
            body.feeder_type = ""
            body.feeder_m_mm = 0.0
        self.viewer.show_bodies(self._bodies, reset_camera=False)

    def on_body_row_selected(self, row: int):
        """Highlight the selected body in the 3D viewer (red) for easier identification."""
        if not self._bodies or row < 0:
            self.viewer.show_bodies(self._bodies, reset_camera=False)
            return
        item = self.body_list.item(row)
        if item is None:
            return
        widget = self.body_list.itemWidget(item)
        if widget is None:
            return
        self.viewer.show_bodies(self._bodies, selected_body=widget.body(), reset_camera=False)

    def on_body_focused(self, body: Body):
        """Highlight the body whose type dropdown is open."""
        if self._bodies:
            self.viewer.show_bodies(self._bodies, selected_body=body, reset_camera=False)

    def on_body_unfocused(self):
        """Clear the temporary body highlight when the dropdown closes."""
        if self._bodies:
            self.viewer.show_bodies(self._bodies, reset_camera=False)

    def on_body_feeder_type_changed(self, body: Body, feeder_type: str):
        self.aiLog(
            f"{body.name} - besleyici tipi: {FEEDER_TYPE_NAMES.get(feeder_type, feeder_type)}",
            "ok",
        )

    def on_body_feeder_m_changed(self, body: Body, feeder_m_mm: float):
        if feeder_m_mm > 0:
            self.aiLog(
                f"{body.name} - besleyici modülü: M={feeder_m_mm / 10.0:.2f} cm",
                "info",
            )

    def on_body_mold_changed(self, body: Body):
        preset = body.mold_preset
        self.aiLog(
            f"{body.name} - kalıp kumu: {preset} (AFS={body.mold_afs_grain_size:.1f}, "
            f"nem={body.mold_moisture_percent:.1f}%, bağlayıcı={body.mold_binder_percent:.1f}%, "
            f"compactability={body.mold_compactability_percent:.1f}%)",
            "info",
        )

    def on_pick_section(self):
        """Open SectionDialog for the selected velocity-section body."""
        if not self._bodies:
            QtWidgets.QMessageBox.warning(self, "UYARI", "Önce STEP dosyası yükleyin.")
            return
        section_key = self.velocity_section_combo.currentData()
        if section_key == "SPRUE_THROAT":
            target_types = {BodyType.SPRUE_THROAT, BodyType.SPRUE}
        elif section_key == "SPRUE_BASE":
            target_types = {BodyType.SPRUE, BodyType.POURING_BASIN}
        else:
            target_types = {BodyType.SPRUE, BodyType.SPRUE_THROAT, BodyType.POURING_BASIN}
        candidates = [b for b in self._bodies if b.body_type in target_types]
        if not candidates:
            QtWidgets.QMessageBox.warning(
                self, "UYARI",
                f"{section_key} tipinde body bulunamadı. Lütfen body tipini doğru atayın."
            )
            return
        if len(candidates) == 1:
            body = candidates[0]
        else:
            names = [b.name for b in candidates]
            name, ok = QtWidgets.QInputDialog.getItem(
                self, "Body Seçimi", f"{section_key} için body seçin:", names, 0, False
            )
            if not ok or not name:
                return
            body = next((b for b in candidates if b.name == name), None)
            if body is None:
                return
        try:
            dialog = SectionDialog(body, section_key=section_key, parent=self)
        except Exception as e:
            self.aiLog(f"Kesit dialogu açılamadı: {e}", "crit")
            QtWidgets.QMessageBox.critical(self, "Hata", f"Kesit dialogu açılamadı:\n{e}")
            return
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            if dialog.area_cm2 and dialog.area_cm2 > 0.0:
                self._user_section_area_cm2 = float(dialog.area_cm2)
                self._user_section_key = str(dialog.section_key)
                self._user_section_body_name = body.name
                self.section_pick_label.setText(f"A={self._user_section_area_cm2:.2f} cm² ({body.name})")
                self.aiLog(
                    f"{body.name} - {dialog.section_key}: A = {self._user_section_area_cm2:.4f} cm²", "ok"
                )

    def on_voxelize(self):
        if not self._bodies:
            return
        try:
            self.progress.setValue(0)
            self.status_label.setText("Voxelizasyon yapılıyor...")
            self.aiLog("AŞAMA 1/6: STEP'den çoklu body voxel grid oluşturuluyor...", "info")
            self._set_progress(10)
            target_dim = self.res_spin.value()
            grid, body_index, origin, dx, bodies = build_voxel_grid(
                self._bodies,
                target_dim=target_dim,
                progress_callback=self._set_progress,
                gravity_vector=self._gravity_vector_from_ui(),
            )
            self._grid = grid
            self._body_index = body_index
            self._origin = origin
            self._dx = dx
            self._bodies = bodies

            # Surface watertight warnings from the voxelizer's repair step.
            watertight_warnings = [
                b.watertight_warning for b in bodies if b.watertight_warning
            ]
            if watertight_warnings:
                msg = "\n".join(watertight_warnings)
                self.aiLog(msg, "crit")
                QtWidgets.QMessageBox.warning(self, "UYARI", msg)

            self.viewer.show_bodies(self._bodies)
            self.viewer.set_gating_data(self._bodies, self._body_index, self._origin, self._dx)
            self.progress.setValue(100)
            self.status_label.setText(
                f"Voxel grid hazır: {grid.shape} (dx={dx:.3f} mm)"
            )
            self.aiLog(
                f"Voxel grid: {grid.shape} | dx={dx:.3f} mm | metal voxel={int((grid > 0).sum())}",
                "ok",
            )
            self.analyze_btn.setEnabled(True)
        except Exception as e:
            import traceback
            self.aiLog(f"Voxelizasyon hatası: {e}", "crit")
            QtWidgets.QMessageBox.critical(
                self, "Voxelizasyon Hatası", f"{e}\n{traceback.format_exc()}"
            )

    def on_analyze(self):
        if self._grid is None:
            return
        try:
            self.status_label.setText("Titan motoru çalışıyor, 2-3 dk sürebilir...")
            self.aiLog("AŞAMA 2/6: SDF + Chvorinov + eğrilik + iskelet hesaplanıyor...", "info")
            self.progress.setValue(0)
            self._analysis_t0 = time.time()
            self.analyze_btn.setEnabled(False)

            alloy_key = self.alloy_combo.currentData()
            mold_key = self._current_mold_key()
            max_res = self.res_spin.value()
            refine_local = self.refine_check.isChecked()
            sub_voxel = self.subvox_spin.value()
            thermal_max_time_s = self.thermal_spin.value()
            casting_params = self._casting_params_from_ui()

            alloy = get_alloy(alloy_key)
            mold = make_effective_mold(get_mold(mold_key), casting_params)
            chvorinov_c = chvorinov_c_from_properties(alloy, mold)
            self.aiLog(
                f"Alaşım: {alloy.name} | Kalıp: {mold.name} | C={chvorinov_c:.4f} dk/cm² | "
                f"Superheat={casting_params.superheat_c:.1f}°C",
                "info",
            )

            self._pending_casting_params = casting_params
            self._analysis_thread = AnalyzeThread(
                self,
                analyze,
                (self._bodies, self._grid, self._body_index, self._origin, self._dx),
                dict(
                    alloy_key=alloy_key,
                    mold_key=mold_key,
                    base_res=160,
                    max_res=max_res,
                    refine_local=refine_local,
                    sub_voxel=sub_voxel,
                    thermal_max_time_s=thermal_max_time_s,
                    thermal_downsample=3,
                    casting_params=casting_params,
                    user_section_areas_cm2=(
                        {self._user_section_key: self._user_section_area_cm2}
                        if self._user_section_area_cm2 > 0.0
                        else None
                    ),
                ),
            )
            self._analysis_thread.progress.connect(self._set_progress)
            self._analysis_thread.finished.connect(self._on_analysis_finished)
            self._analysis_thread.error.connect(self._on_analysis_error)
            self._analysis_thread.finished.connect(self._analysis_thread.deleteLater)
            self._analysis_thread.error.connect(self._analysis_thread.deleteLater)
            self._analysis_thread.start()
        except Exception as e:
            import traceback
            self.aiLog(f"Analiz hatası: {e}", "crit")
            QtWidgets.QMessageBox.critical(
                self, "Analiz Hatası", f"{e}\n{traceback.format_exc()}"
            )
            self.analyze_btn.setEnabled(True)

    def _on_analysis_finished(self, analysis):
        casting_params = getattr(self, "_pending_casting_params", None)
        analysis.casting_params = casting_params
        self._analysis = analysis
        # Use the alloy key carried by the analysis result; the UI combo is the
        # final fallback in case an older result is loaded without one.
        alloy_key = getattr(analysis, "alloy_key", None)
        if not alloy_key and casting_params is not None:
            alloy_key = getattr(casting_params, "alloy_key", None)
        if not alloy_key:
            alloy_key = self.alloy_combo.currentData()
        alloy = get_alloy(alloy_key) if alloy_key else None
        self._update_porosity_filter_labels(alloy)

        gate_result = self._analysis.gate_result
        if gate_result:
            self._analysis.recommendations.extend(
                self._gating_recommendations(gate_result)
            )

        elapsed = time.time() - getattr(self, "_analysis_t0", time.time())
        self.aiLog(f"AŞAMA 6/6: Analiz tamamlandı ({elapsed:.1f} sn)", "ok")

        self.progress.setValue(100)
        n_visible = sum(1 for hs in self._analysis.hotspots if not hs.solved)
        self.status_label.setText(
            f"Analiz tamamlandı ({elapsed:.1f} sn). {n_visible}/{len(self._analysis.hotspots)} hot spot görünür."
        )
        self.analyze_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.html_btn.setEnabled(True)
        self._update_recommendations()
        # Post-analysis: all bodies are translucent so internal markers,
        # porosity, paths, hot-spots and flow/Niyama overlays are visible.
        self.viewer.show_bodies(self._bodies, reset_camera=True, analysis_mode=True)
        self.viewer.set_gating_data(self._bodies, self._body_index, self._origin, self._dx)
        if self.risk_toggle.isChecked():
            self.viewer.show_risk(self._analysis)
        if self.porosity_toggle.isChecked():
            noise, mp, size_filter = self._porosity_cloud_params()
            self.viewer.show_porosity_cloud(self._analysis, noise_percent=noise, max_points=mp, pore_size_filter=size_filter)
        if self.niyama_toggle.isChecked():
            self.viewer.show_niyama_isosurfaces(self._analysis)
        if self.mold_wall_toggle.isChecked():
            self.viewer.toggle_mold_wall_movement(self._analysis, True)
        if self.cold_shot_toggle.isChecked():
            self.viewer.toggle_cold_shot_risk(self._analysis, True)
        if self.erosion_toggle.isChecked():
            self.viewer.toggle_erosion_risk(self._analysis, True)
        if self.air_entrapment_toggle.isChecked():
            self.viewer.toggle_air_entrapment(self._analysis, True)
        if self.path_toggle.isChecked():
            self.viewer.show_feeding_paths(self._analysis)
        if self.local_toggle.isChecked():
            self.viewer.show_local_regions(self._analysis, self.slice_field.currentData())
        self.viewer.show_hotspots(self._analysis)
        self.viewer.show_flow_node_labels(self._analysis)
        self._update_flow_controls()
        if self.flow_anim_toggle.isChecked() and self._analysis.flow_result is not None:
            self.viewer.toggle_flow_animation(self._analysis, True)

    def _on_analysis_error(self, msg):
        import traceback
        self.aiLog(f"Analiz hatası: {msg}", "crit")
        QtWidgets.QMessageBox.critical(self, "Analiz Hatası", msg)
        self.analyze_btn.setEnabled(True)

    def _gating_recommendations(self, gr) -> List[str]:
        if gr is None:
            return []
        section_names = {
            "INGATE": "Meme",
            "RUNNER": "Yolluk",
            "SPRUE_THROAT": "Döküm ağzı boğazı",
            "SPRUE_BASE": "Döküm ağzı tabanı",
        }
        recs = []
        # Geometry / location notes only; do not force area changes.
        if gr.ingate_on_thick_region:
            recs.append(
                f"Not: giriş/kontakt bölgesi kalın kesimde (ortalama M={gr.ingate_avg_m_mm/10.0:.2f} cm)."
            )
        # Per-section velocity / Re / Fr report with reference target ranges.
        for key, sf in getattr(gr, "section_flows", {}).items():
            if sf.area_cm2 <= 0:
                continue
            name = section_names.get(key, key)
            if key == "INGATE" and gr.effective_gate_section.startswith("RUNNER"):
                name = "Yolluk (meme yok)"
            target = ""
            if sf.target_v_min_m_s > 0 and sf.target_v_max_m_s > 0:
                target = (
                    f" (referans v={sf.target_v_min_m_s:.1f}-{sf.target_v_max_m_s:.1f} m/s, "
                    f"A={sf.target_area_min_cm2:.2f}-{sf.target_area_max_cm2:.2f} cm²)"
                )
            turb_note = ""
            if sf.turbulent:
                turb_note = " - yüksek türbülans notu"
            recs.append(
                f"{name}: v={sf.velocity_m_s:.2f} m/s, Re={sf.reynolds:.0f}, Fr={sf.froude:.2f}, "
                f"A={sf.area_cm2:.2f} cm²{turb_note}.{target}"
            )

        # Per-gate velocities when multiple INGATE bodies are detected.
        if gr.flow_result is not None:
            per_gate = getattr(gr.flow_result, "per_gate_contact_velocity_m_s", {})
            per_area = getattr(gr.flow_result, "per_gate_contact_area_cm2", {})
            if per_gate:
                gate_lines = []
                for name, v in per_gate.items():
                    a = per_area.get(name, 0.0)
                    gate_lines.append(f"{name}: v={v:.2f} m/s, A={a:.2f} cm²")
                if gate_lines:
                    recs.append("Meme başına temas hızı/alan: " + " | ".join(gate_lines))

        if gr.ingate_velocity_m_s > 0:
            recs.append(
                f"Toplam debi Q={gr.ingate_flow_rate_m3_s*1e3:.2f} L/s, doldurma süresi={gr.ingate_fill_time_s:.2f}s, "
                f"maks. güvenli meme hızı={gr.ingate_max_velocity_m_s:.2f} m/s."
            )

        # v8.6: practical auto + Campbell fill times
        if getattr(gr, "auto_fill_time_s", 0.0) > 0:
            recs.append(
                f"Dolum süresi: kullanılan {gr.ingate_fill_time_s:.2f} s, "
                f"pratik öneri {gr.auto_fill_time_s:.2f} s, "
                f"Campbell önerisi {getattr(gr, 'campbell_fill_time_s', 0.0):.2f} s."
            )
        return recs

    def _update_checklist(self):
        """Deprecated: the checklist panel was removed from the UI."""
        pass

    def _update_recommendations(self):
        if self._analysis and self._analysis.recommendations:
            html = "<ul style='margin:0;padding-left:16px;color:#334155;'>"
            for r in self._analysis.recommendations:
                html += f"<li style='margin:4px 0'><b>{_escape_html(r)}</b></li>"
            html += "</ul>"
            self.rec_text.setHtml(html)
        else:
            self.rec_text.setPlainText("Henüz öneri yok.")

    def on_toggle_hotspots(self, checked: bool):
        if self._analysis:
            self.viewer.toggle_hotspots(self._analysis, checked)

    def on_toggle_risk(self, checked: bool):
        if self._analysis:
            self.viewer.toggle_risk(self._analysis, checked)

    def on_toggle_porosity(self, checked: bool):
        if self._analysis:
            noise, mp, size_filter = self._porosity_cloud_params()
            self.viewer.toggle_porosity(self._analysis, checked, noise_percent=noise, max_points=mp, pore_size_filter=size_filter)

    def on_porosity_noise_changed(self, value: int):
        noise_percent = value / 100.0
        self.porosity_noise_label.setText(f"Risk: %{noise_percent:.2f}")
        if self._analysis and self.porosity_toggle.isChecked():
            noise, mp, size_filter = self._porosity_cloud_params()
            self.viewer.show_porosity_cloud(self._analysis, noise_percent=noise, max_points=mp, pore_size_filter=size_filter)

    def on_porosity_size_filter_changed(self, index: int):
        if self._analysis and self.porosity_toggle.isChecked():
            noise, mp, size_filter = self._porosity_cloud_params()
            self.viewer.show_porosity_cloud(self._analysis, noise_percent=noise, max_points=mp, pore_size_filter=size_filter)

    def _porosity_cloud_params(self) -> Tuple[float, Optional[int], str]:
        noise_percent = self.porosity_noise_slider.value() / 100.0  # 0.00 .. 100.00
        # max_points is now computed dynamically from part volume and GPU VRAM in the viewer.
        max_points = None
        size_filter = str(self.porosity_size_filter.currentData() or "all")
        return float(noise_percent), max_points, size_filter

    def _update_porosity_filter_labels(self, alloy) -> None:
        """Set class combo labels from the alloy's physical micron limits."""
        macro = int(round(alloy.macro_pore_limit_um))
        micro = int(round(alloy.micro_pore_limit_um))
        self.porosity_size_filter.setItemText(0, "Tüm poroziteler")
        self.porosity_size_filter.setItemText(1, f"Makro (>{macro} µm)")
        self.porosity_size_filter.setItemText(2, f"Mikro ({micro}–{macro} µm)")
        self.porosity_size_filter.setItemText(3, f"İnce (<{micro} µm)")

    def on_toggle_niyama(self, checked: bool):
        if self._analysis:
            self.viewer.toggle_niyama(self._analysis, checked)

    def on_toggle_mold_wall_movement(self, checked: bool):
        if self._analysis:
            self.viewer.toggle_mold_wall_movement(self._analysis, checked)

    def on_toggle_cold_shot_risk(self, checked: bool):
        if self._analysis:
            self.viewer.toggle_cold_shot_risk(self._analysis, checked)

    def on_toggle_erosion_risk(self, checked: bool):
        if self._analysis:
            self.viewer.toggle_erosion_risk(self._analysis, checked)

    def on_toggle_air_entrapment(self, checked: bool):
        if self._analysis:
            self.viewer.toggle_air_entrapment(self._analysis, checked)

    def _update_flow_controls(self):
        has_flow = bool(self._analysis and self._analysis.flow_result)
        self.flow_anim_toggle.setEnabled(has_flow)
        if not self.flow_anim_toggle.isChecked():
            self.flow_play_btn.setEnabled(False)
            self.flow_time_slider.setEnabled(False)
            self.flow_surface_check.setEnabled(False)
            self.flow_time_label.setEnabled(False)
        animator = self.viewer.flow_animator
        if animator and animator._max_time > 0:
            t = animator._current_time
            ratio = t / animator._max_time
            self.flow_time_slider.blockSignals(True)
            self.flow_time_slider.setValue(int(round(ratio * 1000)))
            self.flow_time_slider.blockSignals(False)
            self.flow_time_label.setText(f"t: {t:.3f} s / {animator._max_time:.3f} s")
            if animator:
                self.flow_particle_label.setText(
                    f"{animator.current_frame_index() + 1}/{animator.frame_count()} kare"
                )

    def on_toggle_flow_animation(self, checked: bool):
        if self._analysis:
            self.viewer.toggle_flow_animation(self._analysis, checked)
        self.flow_play_btn.setEnabled(checked and bool(self._analysis and self._analysis.flow_result))
        self.flow_time_slider.setEnabled(checked)
        self.flow_surface_check.setEnabled(checked)
        self.flow_time_label.setEnabled(checked)
        if checked:
            self._update_flow_controls()
            # Do not auto-play; user presses the play button.
        else:
            self.flow_play_btn.setText("▶ Oynat")

    def on_flow_play_clicked(self):
        if self.viewer.flow_animator is None:
            return
        self.viewer.flow_animator.play()
        self.flow_play_btn.setText(
            "⏸ Duraklat" if self.viewer.flow_animator._is_running else "▶ Oynat"
        )

    def on_flow_time_changed(self, value: int):
        if self.viewer.flow_animator is None or not self.flow_anim_toggle.isChecked():
            return
        ratio = value / 1000.0
        t = ratio * self.viewer.flow_animator._max_time
        self.viewer.flow_animator.set_current_time(t)
        self._update_flow_controls()

    def on_flow_speed_changed(self, value: float):
        if self.viewer.flow_animator is not None:
            self.viewer.flow_animator.set_speed_multiplier(value)

    def _on_flow_frame_changed(self, frame: int, t: float, t_max: float):
        if t_max > 0.0:
            ratio = t / t_max
            self.flow_time_slider.blockSignals(True)
            self.flow_time_slider.setValue(int(round(ratio * 1000)))
            self.flow_time_slider.blockSignals(False)
            self.flow_time_label.setText(f"t: {t:.3f} s / {t_max:.3f} s")
        n_frames = self.viewer.flow_animator.frame_count()
        if n_frames > 0:
            self.flow_particle_label.setText(f"{frame + 1}/{n_frames} kare")

    def _on_flow_state_changed(self, is_playing: bool):
        self.flow_play_btn.setText("⏸ Duraklat" if is_playing else "▶ Oynat")

    def on_flow_surface_toggled(self, checked: bool):
        if self.viewer.flow_animator is not None:
            self.viewer.flow_animator.set_show_streamlines(checked)

    def on_toggle_feeding_paths(self, checked: bool):
        if self._analysis:
            self.viewer.toggle_feeding_paths(self._analysis, checked)

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
            self.viewer.toggle_slices(self._analysis, False, "sdf")
            self.viewer.toggle_slices(
                self._analysis, True, self.slice_field.currentData()
            )
        if self._analysis and self.local_toggle.isChecked():
            self.viewer.show_local_regions(self._analysis, self.slice_field.currentData())

    def _generate_report_html(self, path: str):
        """Generate a self-contained HTML report (no PDF conversion)."""
        screenshot = path.replace(".html", ".png")
        self.viewer.save_screenshot(screenshot)
        from core.reporter import generate_report
        generate_report(self._analysis, path.replace(".html", ".pdf"), screenshot)

    def on_export_pdf(self):
        if self._analysis is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "PDF Kaydet", "josecast_rapor.pdf", "PDF Files (*.pdf)"
        )
        if not path:
            return
        try:
            screenshot = os.path.splitext(path)[0] + ".png"
            self.viewer.save_screenshot(screenshot)
            from core.reporter import generate_report
            generate_report(self._analysis, path, screenshot)
            self.status_label.setText(f"Rapor kaydedildi: {path}")
            self.aiLog(f"PDF raporu kaydedildi: {path}", "ok")
        except Exception as e:
            import traceback
            self.aiLog(f"PDF hatası: {e}", "crit")
            QtWidgets.QMessageBox.critical(
                self, "Export Hatası", f"{e}\n{traceback.format_exc()}"
            )

    def on_view_html_report(self):
        if self._analysis is None:
            return
        try:
            path = os.path.join(os.path.expanduser("~"), "josecast_rapor.html")
            screenshot = path.replace(".html", ".png")
            self.viewer.save_screenshot(screenshot)
            from core.reporter import _generate_html
            _generate_html(self._analysis, path, screenshot)
            webbrowser.open(f"file://{path}")
            self.status_label.setText(f"HTML rapor açıldı: {path}")
            self.aiLog(f"HTML rapor tarayıcıda açıldı: {path}", "ok")
        except Exception as e:
            import traceback
            self.aiLog(f"HTML rapor hatası: {e}", "crit")
            QtWidgets.QMessageBox.critical(
                self, "Rapor Hatası", f"{e}\n{traceback.format_exc()}"
            )


def main():
    pv.set_plot_theme("dark")
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()