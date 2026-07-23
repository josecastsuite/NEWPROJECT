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
from core.materials import chvorinov_c_from_properties
from core.types import Body, BodyType, CastingParameters
from ui.feeder_dialog import FeederDialog, FEEDER_TYPE_NAMES
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
}


def _escape_html(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


class CheckListItem(QtWidgets.QWidget):
    """Row in the checklist panel."""

    def __init__(self, text: str, ok: bool, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)
        icon = "✓" if ok else "✗"
        color = "#00ff88" if ok else "#ff4444"
        self.label = QtWidgets.QLabel(
            f'<span style="color:{color};font-weight:bold;font-size:14px">{icon}</span> '
            f'<span style="color:#00ffff;font-weight:bold;">{_escape_html(text)}</span>'
        )
        self.label.setWordWrap(True)
        self.label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        self.label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.label)


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
        self._body_feeder_buttons: Dict[str, QtWidgets.QPushButton] = {}
        self._body_feeder_labels: Dict[str, QtWidgets.QLabel] = {}
        self._body_items: Dict[str, QtWidgets.QListWidgetItem] = {}


        self._build_ui()
        self._apply_dark_theme()
        self._sync_casting_params_from_materials()
        self.aiLog(
            "JOSECAST TITAN ENGINE v8.0 BOOTING... [2040-READY]",
            "info",
        )
        self.aiLog("Siyah AI terminal hazır. Gelecekte LLM bağlantı noktası.", "ok")

    def _apply_dark_theme(self):
        self.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#18181b"))
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#00ffff"))
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#09090b"))
        palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#27272a"))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor("#18181b"))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor("#00ffff"))
        palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#00ffff"))
        palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#3f3f46"))
        palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#000000"))
        palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor("#ff4444"))
        palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#00ff88"))
        palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#000000"))
        self.setPalette(palette)
        
        self.setStyleSheet(
            """
            QMainWindow { background: #18181b; }
            QGroupBox {
                color: #00ffff;
                font-weight: bold;
                font-size: 12px;
                border: 1px solid #3f3f46;
                border-radius: 8px;
                margin-top: 18px; 
                padding-top: 18px; 
                padding-left: 8px;
                padding-right: 8px;
                padding-bottom: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; 
                subcontrol-position: top left;
                left: 12px; 
                top: 0px;
                color: #00ff88; 
                font-weight: bold;
            }
            QPushButton {
                background: #00ff88; color: #000000; border: none;
                border-radius: 6px; padding: 10px 16px; font-weight: bold; font-size: 12px;
            }
            QPushButton:hover { background: #00cc6a; }
            QPushButton:disabled { background: #27272a; color: #55aa88; }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                background: #27272a; color: #00ffff; border: 1px solid #52525b;
                border-radius: 5px; padding: 5px; min-height: 20px; font-weight: bold;
            }
            QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled {
                background: #18181b; color: #55aa88; border: 1px solid #3f3f46;
            }
            QProgressBar {
                background: #27272a; border: 1px solid #52525b; border-radius: 5px;
                text-align: center; color: #00ffff; font-weight: bold;
            }
            QProgressBar::chunk { background: #00ff88; border-radius: 4px; }
            QLabel { color: #00ffff; font-weight: 800; font-size: 13px; }
            QListWidget {
                background: #09090b; border: 1px solid #3f3f46; border-radius: 6px;
                color: #00ffff; font-weight: bold;
            }
            QTextEdit {
                background: #000000; border: 2px solid #00ff88; border-radius: 6px;
                color: #00ff88; font-weight: 800; font-size: 12px;
            }
            QCheckBox { color: #00ffff; spacing: 6px; font-weight: bold; font-size: 12px; }
            QCheckBox::indicator { width: 16px; height: 16px; }
            QCheckBox::indicator:checked { background: #00ff88; border: 1px solid #00ff88; }
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical { background: #27272a; width: 10px; }
            QScrollBar::handle:vertical { background: #00ff88; border-radius: 5px; }
            """
        )

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        
        main_vbox = QtWidgets.QVBoxLayout(central)
        main_vbox.setContentsMargins(8, 8, 8, 8)
        main_vbox.setSpacing(6)

        # ---------------- TOP AREA (Splitter) ----------------
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # ---------------- LEFT PANEL (scrollable) ----------------
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(560) 
        
        left_panel = QtWidgets.QWidget()
        left_scroll.setWidget(left_panel)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(8, 8, 8, 8)

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

        self.unit_combo = QtWidgets.QComboBox()
        for unit, label in [("mm", "mm"), ("cm", "cm"), ("m", "m"), ("inch", "inch")]:
            self.unit_combo.addItem(label, unit)
        self.unit_combo.currentIndexChanged.connect(self.on_unit_changed)
        _settings_labeled(self.unit_combo, "Birim:")

        self.res_spin = QtWidgets.QSpinBox()
        self.res_spin.setRange(160, MAX_RES)
        self.res_spin.setValue(160)
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

        self.mold_combo = QtWidgets.QComboBox()
        for key, mold in MOLDS.items():
            self.mold_combo.addItem(mold.name, key)
        self.mold_combo.currentIndexChanged.connect(self._sync_casting_params_from_materials)
        _settings_labeled(self.mold_combo, "Kalıp:")

        # Set defaults after both combos exist; block signals to avoid partial sync.
        self.alloy_combo.blockSignals(True)
        self.mold_combo.blockSignals(True)
        self.alloy_combo.setCurrentIndex(list(ALLOYS.keys()).index("42CrMo4"))
        self.mold_combo.setCurrentIndex(list(MOLDS.keys()).index("sand"))
        self.alloy_combo.blockSignals(False)
        self.mold_combo.blockSignals(False)

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

        # ---------------- RIGHT PANEL (scrollable) ----------------
        right_scroll = QtWidgets.QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(380) 
        right_panel = QtWidgets.QWidget()
        right_scroll.setWidget(right_panel)
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(4, 4, 4, 4)

        check_group = QtWidgets.QGroupBox("Kontrol Listesi")
        check_inner = QtWidgets.QVBoxLayout(check_group)
        self.checklist_layout = QtWidgets.QVBoxLayout()
        check_inner.addLayout(self.checklist_layout)
        right_layout.addWidget(check_group)

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
        self.porosity_noise_slider.setMinimum(1)    # 0.01%
        self.porosity_noise_slider.setMaximum(300)  # 3.00%
        self.porosity_noise_slider.setValue(300)    # default 3.00%
        self.porosity_noise_slider.setToolTip("Porozite gürültü filtresi: tek slider, ama Makro/Mikro daha geniş, İnce daha dar gösterir")
        self.porosity_noise_slider.valueChanged.connect(self.on_porosity_noise_changed)
        self.porosity_noise_label = QtWidgets.QLabel("Filtre: %3.00")
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

        self.flow_toggle = QtWidgets.QCheckBox("Akış Hızı")
        self.flow_toggle.setToolTip("3-B Darcy akış hızını metal yüzeylerinde göster")
        self.flow_toggle.setChecked(False)
        self.flow_toggle.toggled.connect(self.on_toggle_flow_velocity)
        vis_layout.addWidget(self.flow_toggle)

        self.flow_node_toggle = QtWidgets.QCheckBox("Düğüm Hızları")
        self.flow_node_toggle.setToolTip("Her gating elemanında nokta + hız değeri göster")
        self.flow_node_toggle.setChecked(True)
        self.flow_node_toggle.toggled.connect(self.on_toggle_flow_node_labels)
        vis_layout.addWidget(self.flow_node_toggle)

        anim_group = QtWidgets.QGroupBox("Akış Animasyonu")
        anim_layout = QtWidgets.QVBoxLayout(anim_group)

        self.flow_anim_toggle = QtWidgets.QCheckBox("Akış Animasyonu")
        self.flow_anim_toggle.setToolTip("Döküm ağzından parçaya kırmızı akış yollarını oynatır")
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
        self.flow_speed_spin.setRange(0.01, 5.0)
        self.flow_speed_spin.setValue(1.0)
        self.flow_speed_spin.setSingleStep(0.05)
        self.flow_speed_spin.setDecimals(2)
        self.flow_speed_spin.setSuffix("x")
        self.flow_speed_spin.valueChanged.connect(self.on_flow_speed_changed)
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.flow_speed_spin)

        count_label = QtWidgets.QLabel("Akış:")
        self.flow_particle_label = QtWidgets.QLabel("—")
        self.flow_particle_label.setToolTip("Akış hattı ve dolma noktası sayısı")
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
        splitter.setSizes([560, 680, 560]) 
        
        # Add splitter to main VBox
        main_vbox.addWidget(splitter, stretch=1)

        # ---------------- BOTTOM AREA: AI TERMINAL ----------------
        terminal_group = QtWidgets.QGroupBox("AI ASİSTAN (Yapay Zeka)")
        terminal_group.setStyleSheet("QGroupBox { color: #00ff88; font-weight: bold; border: 2px solid #00ff88; }")
        terminal_layout = QtWidgets.QVBoxLayout(terminal_group)
        
        self.ai_terminal = QtWidgets.QTextEdit()
        self.ai_terminal.setReadOnly(True)
        self.ai_terminal.setMaximumHeight(90)
        self.ai_terminal.setStyleSheet(
            "QTextEdit { background-color: #000000; color: #00ffff; "
            "font-family: 'Consolas', monospace; font-weight: 800; border: none;}"
        )
        terminal_layout.addWidget(self.ai_terminal)
        
        self.ai_input = QtWidgets.QLineEdit()
        self.ai_input.setPlaceholderText("> Komut girişi yapın...")
        self.ai_input.setStyleSheet(
            "QLineEdit { background-color: #18181b; color: #00ff88; border: 1px solid #00ff88; font-weight: 800;}"
        )
        terminal_layout.addWidget(self.ai_input)
        
        main_vbox.addWidget(terminal_group, stretch=0)

    def _sync_casting_params_from_materials(self):
        """Set parameter defaults from the selected alloy and mould."""
        alloy = get_alloy(self.alloy_combo.currentData())
        mold = get_mold(self.mold_combo.currentData())
        self.t_pour_spin.setValue(alloy.t_pour_c)
        self.t_liq_spin.setValue(alloy.t_liquidus_c)
        self.t_sol_spin.setValue(alloy.t_solidus_c)
        self.t_mold_spin.setValue(mold.t0_c)
        self.rho_spin.setValue(alloy.rho_kg_m3)
        self.visc_spin.setValue(alloy.viscosity_pa_s)

    def _casting_params_from_ui(self) -> CastingParameters:
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
            gravity_vector=self._gravity_vector_from_ui(),
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
        """Print a line to the black AI terminal."""
        color = {
            "crit": "#ff4444",
            "ok": "#00ff88",
            "info": "#00ffff",
            "warn": "#ffaa00",
        }.get(type_, "#00ffff")
        line = (
            f'<span style="color:{color};margin:2px 0;font-family:Consolas,monospace;"'
            f'>&gt; {_escape_html(msg)}</span>'
        )
        self.ai_terminal.append(line)
        scrollbar = self.ai_terminal.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _set_progress(self, value: int):
        self.progress.setValue(value)
        QtCore.QCoreApplication.processEvents()

    def _add_body_row(self, body: Body):
        item = QtWidgets.QListWidgetItem()
        widget = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(widget)
        row.setContentsMargins(2, 1, 2, 1)
        row.setSpacing(2)

        # Compact body label; full volume/centroid info in tooltip.
        label = QtWidgets.QLabel(body.name)
        label.setToolTip(
            f"Hacim: {body.volume_cm3:.2f} cm³\nMerkez: {body.center}"
        )
        label.setStyleSheet("font-size: 11px;")
        label.setMaximumWidth(80)
        label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Fixed
        )
        row.addWidget(label)

        # Feeder type controls (always visible; dialog warns if not a RISER).
        feeder_btn = QtWidgets.QPushButton("Besleyici tipi")
        feeder_btn.setToolTip("Bu besleyicinin tipini ve opsiyonel modülünü ayarla")
        feeder_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Fixed
        )
        feeder_btn.clicked.connect(lambda _, b=body: self.on_body_feeder(b))
        row.addWidget(feeder_btn)
        self._body_feeder_buttons[body.name] = feeder_btn

        combo = QtWidgets.QComboBox()
        combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Fixed
        )
        combo.setMaximumWidth(120)
        for bt in (
            BodyType.PART,
            BodyType.RISER,
            BodyType.INGATE,
            BodyType.RUNNER,
            BodyType.SPRUE,
            BodyType.SPRUE_THROAT,
            BodyType.DISTRIBUTOR,
            BodyType.CURUFLUK,
            BodyType.COOLING_SPRUE,
            BodyType.FILTER,
            BodyType.POURING_BASIN,
            BodyType.CORE,
        ):
            combo.addItem(BODY_TYPE_NAMES[bt], bt)
        combo.setCurrentIndex(combo.findData(body.body_type))
        combo.currentIndexChanged.connect(
            lambda _, b=body, c=combo: self.on_body_type_changed(b, c)
        )
        row.addWidget(combo)

        self._body_items[body.name] = item

        self.body_list.addItem(item)
        self.body_list.setItemWidget(item, widget)
        self._update_body_row_state(body)
        self._update_body_feeder_label(body.name)
        item.setSizeHint(widget.sizeHint())

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
            self._body_feeder_buttons.clear()
            self._body_feeder_labels.clear()
            self._body_items.clear()

            self._clear_checklist()
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
        self.aiLog(f"Birim ölçeği uygulandı: {unit} (x{self._unit_scale:.2f})", "ok")
        self.viewer.show_bodies(self._bodies)

    def on_body_type_changed(self, body: Body, combo: QtWidgets.QComboBox):
        data = combo.currentData()
        try:
            new_type = BodyType(data) if isinstance(data, int) else data
        except Exception:
            new_type = BodyType.PART

        old_type = body.body_type
        body.body_type = new_type

        # If the body is no longer a riser, clear feeder overrides.
        if old_type != new_type and new_type != BodyType.RISER:
            body.feeder_type = ""
            body.feeder_m_mm = 0.0
            body.feeder_note = ""
            self._update_body_feeder_label(body.name)

        self._update_body_row_state(body)
        self.viewer.show_bodies(self._bodies)

    def _update_body_row_state(self, body: Body):
        # Keeping this hook for any future row-specific updates.
        pass

    def _update_body_feeder_label(self, body_name: str):
        label = self._body_feeder_labels.get(body_name)
        if label is None:
            return
        body = next((b for b in self._bodies if b.name == body_name), None)
        if body is None or not body.feeder_type:
            label.setText("")
            return
        short_names = {
            "conventional": "konv",
            "exothermic": "ekzo",
            "insulated": "izol",
            "sleeve": "göm",
            "chilled": "chill",
        }
        type_text = short_names.get(body.feeder_type, body.feeder_type[:4])
        m_text = f" M={body.feeder_m_mm:.1f}" if body.feeder_m_mm > 0 else " auto"
        label.setText(f"{type_text}{m_text}")

    def on_body_feeder(self, body: Body):
        """Open the per-riser feeder type / optional modulus dialog."""
        if body is None:
            return
        try:
            bt = BodyType(body.body_type) if isinstance(body.body_type, int) else body.body_type
        except Exception:
            bt = body.body_type
        if bt != BodyType.RISER:
            QtWidgets.QMessageBox.information(
                self, "Tip Uyarısı",
                "Besleyici tipi seçimi sadece BESLEYİCİ (RISER) tipindeki body'ler için geçerlidir.\n"
                "Lütfen önce body tipini değiştirin."
            )
            return

        try:
            dialog = FeederDialog(body, parent=self)
        except Exception as e:
            self.aiLog(f"Besleyici dialogu açılamadı: {e}", "crit")
            QtWidgets.QMessageBox.critical(self, "Hata", f"Besleyici dialogu açılamadı:\n{e}")
            return

        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            body.feeder_type = dialog.feeder_type or "conventional"
            body.feeder_m_mm = float(dialog.feeder_m_mm)
            body.feeder_note = dialog.feeder_note
            self._update_body_feeder_label(body.name)
            self.aiLog(
                f"{body.name} - besleyici tipi: {FEEDER_TYPE_NAMES.get(body.feeder_type, body.feeder_type)}"
                f"{', M=' + f'{body.feeder_m_mm:.2f} mm' if body.feeder_m_mm > 0 else ''}",
                "ok",
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
            grid, origin, dx, bodies = build_voxel_grid(
                self._bodies,
                target_dim=target_dim,
                progress_callback=self._set_progress,
                gravity_vector=self._gravity_vector_from_ui(),
            )
            self._grid = grid
            self._origin = origin
            self._dx = dx
            self._bodies = bodies
            self.viewer.show_bodies(self._bodies)
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
            t0 = time.time()

            alloy_key = self.alloy_combo.currentData()
            mold_key = self.mold_combo.currentData()
            max_res = self.res_spin.value()
            refine_local = self.refine_check.isChecked()
            sub_voxel = self.subvox_spin.value()
            thermal_max_time_s = self.thermal_spin.value()
            casting_params = self._casting_params_from_ui()

            alloy = get_alloy(alloy_key)
            mold = get_mold(mold_key)
            chvorinov_c = chvorinov_c_from_properties(alloy, mold)
            self.aiLog(
                f"Alaşım: {alloy.name} | Kalıp: {mold.name} | C={chvorinov_c:.4f} s/mm² | "
                f"Superheat={casting_params.superheat_c:.1f}°C",
                "info",
            )

            self._analysis = analyze(
                self._bodies,
                self._grid,
                self._origin,
                self._dx,
                alloy_key=alloy_key,
                mold_key=mold_key,
                base_res=160,
                max_res=max_res,
                refine_local=refine_local,
                sub_voxel=sub_voxel,
                thermal_max_time_s=thermal_max_time_s,
                thermal_downsample=3,
                casting_params=casting_params,
                progress_callback=self._set_progress,
                user_section_areas_cm2=(
                    {self._user_section_key: self._user_section_area_cm2}
                    if self._user_section_area_cm2 > 0.0
                    else None
                ),
            )
            self._analysis.casting_params = casting_params

            gate_result = self._analysis.gate_result
            if gate_result:
                self._analysis.recommendations.extend(
                    self._gating_recommendations(gate_result)
                )

            elapsed = time.time() - t0
            self.aiLog(f"AŞAMA 6/6: Analiz tamamlandı ({elapsed:.1f} sn)", "ok")

            self.progress.setValue(100)
            n_visible = sum(1 for hs in self._analysis.hotspots if not hs.solved)
            self.status_label.setText(
                f"Analiz tamamlandı ({elapsed:.1f} sn). {n_visible}/{len(self._analysis.hotspots)} hot spot görünür."
            )
            self.export_btn.setEnabled(True)
            self.html_btn.setEnabled(True)
            self._update_checklist()
            self._update_recommendations()
            # Post-analysis: all bodies are translucent so internal markers,
            # porosity, paths, hot-spots and flow/Niyama overlays are visible.
            self.viewer.show_bodies(self._bodies, reset_camera=True, analysis_mode=True)
            if self.risk_toggle.isChecked():
                self.viewer.show_risk(self._analysis)
            if self.porosity_toggle.isChecked():
                noise, mp, size_filter = self._porosity_cloud_params()
                self.viewer.show_porosity_cloud(self._analysis, noise_percent=noise, max_points=mp, pore_size_filter=size_filter)
            if self.niyama_toggle.isChecked():
                self.viewer.show_niyama_isosurfaces(self._analysis)
            if self.flow_toggle.isChecked():
                self.viewer.show_flow_velocity(self._analysis)
            if self.flow_node_toggle.isChecked():
                self.viewer.show_flow_node_labels(self._analysis)
            if self.path_toggle.isChecked():
                self.viewer.show_feeding_paths(self._analysis)
            if self.local_toggle.isChecked():
                self.viewer.show_local_regions(self._analysis, self.slice_field.currentData())
            self.viewer.show_hotspots(self._analysis)
            self._update_flow_controls()
            if self.flow_anim_toggle.isChecked() and self._analysis.flow_result is not None:
                self.viewer.toggle_flow_animation(self._analysis, True)
        except Exception as e:
            import traceback
            self.aiLog(f"Analiz hatası: {e}", "crit")
            QtWidgets.QMessageBox.critical(
                self, "Analiz Hatası", f"{e}\n{traceback.format_exc()}"
            )

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
                f"Not: giriş/kontakt bölgesi kalın kesimde (ortalama M={gr.ingate_avg_m_mm:.2f} mm)."
            )
        if getattr(gr, "gating_system_reason", ""):
            recs.append(gr.gating_system_reason)

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

    def _clear_checklist(self):
        while self.checklist_layout.count():
            item = self.checklist_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _update_checklist(self):
        self._clear_checklist()
        if self._analysis is None:
            return

        visible_hotspots = [hs for hs in self._analysis.hotspots if not hs.solved]
        for hs in visible_hotspots:
            status = "OK" if hs.feed_ok else "DARALMA/UZAK"
            text = (
                f"Hot spot M={hs.m_value_mm:.1f} mm, t={hs.t_section_mm:.1f} mm, "
                f"mesafe {hs.dist_to_riser_mm:.1f} mm / limit {hs.max_feeding_distance_mm:.1f} mm, "
                f"Niyama={hs.niyama_ensemble:.2f}"
            )
            self.checklist_layout.addWidget(CheckListItem(text, hs.feed_ok))
        if any(hs.solved for hs in self._analysis.hotspots):
            n_total = len(self._analysis.hotspots)
            n_hidden = n_total - len(visible_hotspots)
            note = QtWidgets.QLabel(
                f"{n_hidden} adet hot spot yeterli besleyici/çıkıcı ile çözüldü; "
                "sadece çözülmemişler listeleniyor."
            )
            note.setStyleSheet("color: green;")
            self.checklist_layout.addWidget(note)

        for rr in self._analysis.riser_results:
            eff_m = max(rr.effective_m_value_mm, rr.m_value_mm)
            type_text = f" [{rr.feeder_type}]" if rr.feeder_type else ""
            text = (
                f"{rr.name}{type_text}: M={rr.m_value_mm / 10.0:.1f} / etkin {eff_m / 10.0:.1f} cm, "
                f"V={rr.volume_cm3:.2f} cm³ (gerekli {rr.required_volume_cm3:.2f} cm³)"
            )
            self.checklist_layout.addWidget(CheckListItem(text, rr.large_enough and rr.volume_ratio_ok))

        for rp in self._analysis.riser_proposals:
            pos = f"({rp.placement_mm[0] / 10.0:.1f}, {rp.placement_mm[1] / 10.0:.1f}, {rp.placement_mm[2] / 10.0:.1f})"
            if rp.infeasible:
                text = (
                    f"UYARI Hotspot #{rp.target_hotspot_index + 1}: besleyici/çıkıcı parçaya sığmıyor. "
                    f"Mini exotermik besleyici veya çıkıcı (chill) önerilir, konum={pos} cm."
                )
                ok = False
            elif rp.shape == "chill":
                text = (
                    f"ÖNERİ Hotspot #{rp.target_hotspot_index + 1}: çıkıcı (chill) ekle -> "
                    f"çap={rp.diameter_mm / 10.0:.1f} cm, yükseklik={rp.height_mm / 10.0:.1f} cm, "
                    f"V={rp.volume_cm3:.2f} cm³, konum={pos} cm"
                )
                ok = True
            elif rp.exothermic:
                text = (
                    f"ÖNERİ Hotspot #{rp.target_hotspot_index + 1}: ekzotermik mini besleyici ekle -> "
                    f"çap={rp.diameter_mm / 10.0:.1f} cm, yükseklik={rp.height_mm / 10.0:.1f} cm, "
                    f"V={rp.volume_cm3:.2f} cm³, konum={pos} cm"
                )
                ok = True
            else:
                text = (
                    f"ÖNERİ Hotspot #{rp.target_hotspot_index + 1}: {rp.shape} besleyici ekle -> "
                    f"çap={rp.diameter_mm / 10.0:.1f} cm, yükseklik={rp.height_mm / 10.0:.1f} cm, "
                    f"V={rp.volume_cm3:.2f} cm³, M={rp.m_required_mm / 10.0:.2f} cm, konum={pos} cm"
                )
                ok = True
            self.checklist_layout.addWidget(CheckListItem(text, ok))

        if self._analysis.gate_result:
            gr = self._analysis.gate_result
            section_names = {
                "INGATE": "Meme",
                "RUNNER": "Yolluk",
                "SPRUE_THROAT": "D.Ağzı boğazı",
                "SPRUE_BASE": "D.Ağzı tabanı",
            }
            self.checklist_layout.addWidget(
                CheckListItem(
                    f"Yolluk: {gr.runner_min_area_cm2:.2f} cm² (gerekli {gr.required_runner_area_cm2:.2f} cm²)",
                    gr.runner_ok,
                )
            )
            self.checklist_layout.addWidget(
                CheckListItem(
                    f"Döküm ağzı boğazı: {gr.sprue_throat_area_cm2:.2f} cm² (gerekli {gr.required_sprue_area_cm2:.2f} cm²)",
                    gr.bernoulli_ok,
                )
            )
            self.checklist_layout.addWidget(
                CheckListItem(
                    "Meme konumu (kalın bölgede olmamalı)",
                    not gr.ingate_on_thick_region,
                )
            )
            if gr.detected_gating_system:
                self.checklist_layout.addWidget(
                    CheckListItem(
                        f"Sistem: {gr.detected_gating_system} | Önerilen: {gr.recommended_gating_system} | Cidar: {gr.wall_thickness_category}",
                        gr.detected_gating_system == gr.recommended_gating_system,
                    )
                )
            # v8.4: per-section velocity / Re / Fr checklist items with target ranges
            for key, sf in getattr(gr, "section_flows", {}).items():
                if sf.area_cm2 <= 0:
                    continue
                name = section_names.get(key, key)
                if key == "INGATE" and gr.effective_gate_section.startswith("RUNNER"):
                    name = "Yolluk (meme yok)"
                target = ""
                if sf.target_v_min_m_s > 0 and sf.target_v_max_m_s > 0:
                    target = (
                        f" hedef v={sf.target_v_min_m_s:.1f}-{sf.target_v_max_m_s:.1f}, "
                        f"A={sf.target_area_min_cm2:.2f}-{sf.target_area_max_cm2:.2f}"
                    )
                ok = not sf.turbulent
                if sf.target_v_min_m_s > 0 and sf.target_v_max_m_s > 0:
                    ok = ok and (sf.target_v_min_m_s <= sf.velocity_m_s <= sf.target_v_max_m_s)
                self.checklist_layout.addWidget(
                    CheckListItem(
                        f"{name}: v={sf.velocity_m_s:.2f}{target}",
                        ok,
                    )
                )
            if hasattr(gr, "velocity_fill_time_match_ok"):
                velocity_ok = (
                    gr.velocity_fill_time_match_ok
                    and getattr(gr, "velocity_area_ok", True)
                    and not gr.turbulent
                )
                vtext = (
                    f"Seçili kesit: {section_names.get(getattr(gr, 'selected_section_key', 'INGATE'), 'Meme')} "
                    f"v={gr.ingate_velocity_m_s:.2f} m/s, "
                    f"doldurma {gr.ingate_fill_time_s:.2f}s, Q={gr.ingate_flow_rate_m3_s*1e3:.2f} L/s"
                )
                self.checklist_layout.addWidget(
                    CheckListItem(vtext, velocity_ok)
                )
            # v8.5: Campbell fill time and theoretical area cross-checks
            if gr.recommended_fill_time_s > 0:
                fill_ok = abs(gr.recommended_fill_time_s - gr.ingate_fill_time_s) <= 0.2 * gr.recommended_fill_time_s
                self.checklist_layout.addWidget(
                    CheckListItem(
                        f"Campbell tavsiye dolum süresi: {gr.recommended_fill_time_s:.2f} s; girilen: {gr.ingate_fill_time_s:.2f} s",
                        fill_ok,
                    )
                )
            if gr.design_sprue_base_area_cm2 > 0:
                self.checklist_layout.addWidget(
                    CheckListItem(
                        f"Sprue taban: gerçek {gr.sprue_base_area_cm2:.2f} cm² / teorik {gr.design_sprue_base_area_cm2:.2f} cm²",
                        gr.sprue_design_ok,
                    )
                )
            if gr.design_runner_area_cm2 > 0:
                self.checklist_layout.addWidget(
                    CheckListItem(
                        f"Yolluk: gerçek {gr.runner_min_area_cm2:.2f} cm² / teorik {gr.design_runner_area_cm2:.2f} cm²",
                        gr.runner_design_ok,
                    )
                )
            if gr.design_gate_total_area_cm2 > 0:
                self.checklist_layout.addWidget(
                    CheckListItem(
                        f"Gate toplam: gerçek {gr.total_ingate_contact_area_cm2:.2f} cm² / teorik {gr.design_gate_total_area_cm2:.2f} cm²",
                        gr.gate_design_ok,
                    )
                )

        if self._analysis and self._analysis.flow_result:
            fr = self._analysis.flow_result
            self.checklist_layout.addWidget(
                CheckListItem(
                    f"3-B Akış: Q={fr.Q_m3_s*1e3:.2f} L/s, doldurma={fr.fill_time_s:.2f} s, meme temas v={fr.ingate_contact_velocity_m_s:.2f} m/s",
                    True,
                )
            )
            section_names = {
                "SPRUE_THROAT": "D.ağzı boğazı",
                "SPRUE_BASE": "D.ağzı tabanı",
                "RUNNER": "Yolluk",
                "DISTRIBUTOR": "Dağıtıcı",
                "CURUFLUK": "Curufluk",
                "INGATE": "Meme",
                "FILTER": "Filtre",
                "RISER": "Besleyici",
            }
            for key, val in fr.node_velocities.items():
                if val <= 1e-9:
                    continue
                name = section_names.get(key, key)
                self.checklist_layout.addWidget(
                    CheckListItem(
                        f"  {name}: v={val:.3f} m/s ({val*100:.1f} cm/s)",
                        True,
                    )
                )

    def _update_recommendations(self):
        if self._analysis and self._analysis.recommendations:
            html = "<ul style='margin:0;padding-left:16px;color:#00ffff;'>"
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
        self.porosity_noise_label.setText(f"Filtre: %{noise_percent:.2f}")
        if self._analysis and self.porosity_toggle.isChecked():
            noise, mp, size_filter = self._porosity_cloud_params()
            self.viewer.show_porosity_cloud(self._analysis, noise_percent=noise, max_points=mp, pore_size_filter=size_filter)

    def on_porosity_size_filter_changed(self, index: int):
        if self._analysis and self.porosity_toggle.isChecked():
            noise, mp, size_filter = self._porosity_cloud_params()
            self.viewer.show_porosity_cloud(self._analysis, noise_percent=noise, max_points=mp, pore_size_filter=size_filter)

    def _porosity_cloud_params(self) -> Tuple[float, int, str]:
        noise_percent = self.porosity_noise_slider.value() / 100.0  # 0.01 .. 3.00
        max_points = 5000
        size_filter = str(self.porosity_size_filter.currentData() or "all")
        return float(noise_percent), int(max_points), size_filter

    def on_toggle_niyama(self, checked: bool):
        if self._analysis:
            self.viewer.toggle_niyama(self._analysis, checked)

    def on_toggle_flow_velocity(self, checked: bool):
        if self._analysis:
            self.viewer.toggle_flow_velocity(self._analysis, checked)

    def on_toggle_flow_node_labels(self, checked: bool):
        if self._analysis:
            self.viewer.toggle_flow_node_labels(self._analysis, checked)

    def _update_flow_controls(self):
        enabled = bool(self._analysis and self._analysis.flow_result)
        self.flow_anim_toggle.setEnabled(enabled)
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
                    f"{animator.line_count()} hat, {animator.particle_count()} nokta"
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