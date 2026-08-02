"""Enhanced 2-D flow velocity graphs for the Darcy flow result.

The dialog now contains three panels and a metric selector:

1. Akış metriği - Dolum zamanı:  Darcy ön cephe metriğinin (hız, Reynolds,
   Froude veya hidrolik çap) dolum süresi boyunca değişimi.  Çizgi renkli,
   10.-90. yüzdelik bant şeffaf renkli ve gating düğümlerinin ulaşma
   zamanları nokta olarak işaretlenmiş.

2. Akış yolu profili:  Döküm ağzından (sprue) her meme girişine kadar gating
   düğümleri boyunca seçili metrik profili.  Her nokta gerçek kesit ortalaması
   (Q/A) üzerinden Reynolds/Froude ile renklendirilir; yarıçap/hidrolik çap
   etkisi görünür.

3. Kesit geometrisi:  Her gating düğümünde kesit alanı (cm²) ve hidrolik çap
   (mm) çift y-eksenli olarak çizilir; dar bölgeler (radyus etkisi) net görülür.

Tüm metrikler gerçek Darcy vektör alanından (`flow_result.velocity`) veya
Q/A süreklilik hesabından gelir; ayrıca kesit yarıçapı, hidrolik çap ve
malzeme yoğunluğu/viskozitesi ile Reynolds/Froude hesaplanır.
"""
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from PyQt6 import QtCore, QtWidgets
from matplotlib import cm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_hex
from matplotlib.figure import Figure

from core.materials import get_alloy
from core.types import AnalysisResult, GatingNode


class FlowVelocityGraph(QtWidgets.QDialog):
    """Popup dialog showing Darcy velocity / Re / Fr / hydraulic diameter vs fill time
    and flow-path profile, plus a cross-section geometry trace."""

    def __init__(self, result: AnalysisResult, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Akış Hızı - Reynolds - Froude - Geometri")
        self.setMinimumSize(1200, 1050)
        self._setup_matplotlib_style()

        self._result = result
        self._rho, self._mu, self._g = self._material_and_gravity(result)

        layout = QtWidgets.QVBoxLayout(self)

        # Controls
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Renkli metrik:"))
        self._metric_combo = QtWidgets.QComboBox()
        self._metric_combo.addItem("Hız |v| (m/s)", "velocity")
        self._metric_combo.addItem("Reynolds (türbülans)", "reynolds")
        self._metric_combo.addItem("Froude (dalga/çalkantı)", "froude")
        self._metric_combo.addItem("Hidrolik Çap Dh (mm)", "diameter")
        self._metric_combo.setCurrentIndex(0)
        self._metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        controls.addWidget(self._metric_combo)

        self._radius_check = QtWidgets.QCheckBox("Kesit yarıçapı etkisini göster")
        self._radius_check.setChecked(True)
        self._radius_check.stateChanged.connect(self._on_metric_changed)
        controls.addWidget(self._radius_check)

        self._smooth_check = QtWidgets.QCheckBox("26-komşu medyan ile düğüm örnekle")
        self._smooth_check.setChecked(True)
        self._smooth_check.stateChanged.connect(self._on_metric_changed)
        controls.addStretch()
        layout.addLayout(controls)

        self.figure = Figure(figsize=(12, 10.5), dpi=100, tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)

        self._plot()

    def _setup_matplotlib_style(self) -> None:
        import matplotlib as mpl

        rc = {
            "figure.facecolor": "#1a1a2e",
            "axes.facecolor": "#16213e",
            "axes.edgecolor": "#e0e0e0",
            "axes.labelcolor": "#e0e0e0",
            "xtick.color": "#e0e0e0",
            "ytick.color": "#e0e0e0",
            "text.color": "#e0e0e0",
            "grid.color": "#0f3460",
            "grid.alpha": 0.35,
            "axes.grid": True,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
        for k, v in rc.items():
            try:
                mpl.rcParams[k] = v
            except Exception:
                pass

    def _style_axes(self, ax) -> None:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#e0e0e0")
        for spine in ax.spines.values():
            spine.set_color("#e0e0e0")
        ax.xaxis.label.set_color("#e0e0e0")
        ax.yaxis.label.set_color("#e0e0e0")
        ax.title.set_color("#e0e0e0")
        ax.grid(True, alpha=0.35, color="#0f3460")

    @staticmethod
    def _material_and_gravity(result: AnalysisResult) -> Tuple[float, float, float]:
        rho = 7000.0
        mu = 0.006
        g = 9.81
        cp = result.casting_params
        if cp is not None:
            rho = float(cp.rho_liquid_kg_m3) if cp.rho_liquid_kg_m3 > 0 else rho
            mu = float(cp.viscosity_pa_s) if cp.viscosity_pa_s > 0 else mu
            if hasattr(cp, "gravity_vector") and cp.gravity_vector:
                try:
                    gv = np.asarray(cp.gravity_vector, dtype=np.float64)
                    g = float(np.linalg.norm(gv))
                    if g <= 0:
                        g = 9.81
                except Exception:
                    g = 9.81
        else:
            try:
                alloy = get_alloy(result.alloy_key)
                rho = float(getattr(alloy, "rho_liquid_kg_m3", rho))
                mu = float(getattr(alloy, "viscosity_pa_s", mu))
            except Exception:
                pass
        return rho, mu, g

    def _on_metric_changed(self, *_) -> None:
        self._plot()

    def _metric_label(self, metric: str) -> str:
        return {
            "velocity": "|v| (m/s)",
            "reynolds": "Re",
            "froude": "Fr",
            "diameter": "Dh (mm)",
        }.get(metric, metric)

    def _cmap_for_metric(self, metric: str):
        return {
            "velocity": cm.get_cmap("turbo"),
            "reynolds": cm.get_cmap("coolwarm"),
            "froude": cm.get_cmap("plasma"),
            "diameter": cm.get_cmap("viridis"),
        }.get(metric, cm.get_cmap("turbo"))

    def _sample_neighbor_median(
        self,
        field: np.ndarray,
        ijk: np.ndarray,
        radius: int = 1,
        order: int = 1,
    ) -> float:
        """Sample `field` at a 3x3x3 neighborhood around ijk and return the median.

        Coordinates are in array-index order [x_idx, y_idx, z_idx] because the
        voxel grid is stored with axes (x, y, z).
        """
        offsets = np.array(
            [[i, j, k] for i in range(-radius, radius + 1)
             for j in range(-radius, radius + 1)
             for k in range(-radius, radius + 1)],
            dtype=np.float64,
        ).T  # shape (3, 27)
        coords = ijk[:, None] + offsets
        vals = ndimage.map_coordinates(
            field,
            coords,
            order=order,
            mode="nearest",
            cval=np.nan,
        )
        vals = np.asarray(vals, dtype=np.float64)
        finite = np.isfinite(vals)
        if finite.any():
            return float(np.median(vals[finite]))
        return float(np.nan)

    def _compute_per_voxel_metrics(
        self,
        result: AnalysisResult,
        ft: np.ndarray,
        vm: np.ndarray,
        grid: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        metal = grid > 0
        valid = metal & np.isfinite(ft) & np.isfinite(vm) & (vm > 0.0)

        # Hydraulic diameter: D_h = 2 * distance_to_surface (in metres).
        dx_m = float(result.dx_mm) / 1000.0
        if result.sdf is not None and result.sdf.size == grid.size:
            sdf_m = np.asarray(result.sdf, dtype=np.float64) / 1000.0
            dh_m = 2.0 * np.maximum(sdf_m, dx_m * 0.5)
        else:
            dt = ndimage.distance_transform_edt(metal, sampling=dx_m)
            dh_m = 2.0 * np.maximum(dt, dx_m * 0.5)

        re = np.where(
            valid & (dh_m > 0),
            self._rho * vm * dh_m / self._mu,
            0.0,
        )
        fr = np.where(
            valid & (dh_m > 0),
            vm / np.sqrt(self._g * dh_m),
            0.0,
        )

        return {
            "velocity": np.where(valid, vm, 0.0),
            "reynolds": re,
            "froude": fr,
            "diameter": np.where(valid, dh_m * 1000.0, 0.0),  # mm
            "valid": valid,
            "dh_m": dh_m,
        }

    def _compute_node_metrics(self, gn: GatingNode) -> Dict[str, float]:
        """Compute accurate section-based Re/Fr/Dh for a gating node."""
        v = float(gn.velocity_m_s) if gn.velocity_m_s > 1e-12 else 0.0
        area_cm2 = float(gn.section_area_cm2)
        area_m2 = area_cm2 * 1e-4
        if area_m2 > 0:
            dh_m = 2.0 * math.sqrt(area_m2 / math.pi)
        else:
            dh_m = 1e-3
        re = self._rho * v * dh_m / self._mu if dh_m > 0 else 0.0
        fr = v / math.sqrt(self._g * dh_m) if dh_m > 0 else 0.0
        return {
            "velocity": v,
            "reynolds": re,
            "froude": fr,
            "diameter": dh_m * 1000.0,
            "area_cm2": area_cm2,
        }

    def _sample_node_arrival_time(
        self,
        result: AnalysisResult,
        gn: GatingNode,
        ft: np.ndarray,
        grid: np.ndarray,
    ) -> float:
        origin = np.asarray(result.origin_mm, dtype=np.float64)
        dx = float(result.dx_mm)
        pos = np.asarray(gn.centroid_mm, dtype=np.float64)
        ijk = (pos - origin) / dx
        try:
            grid_sample = self._sample_neighbor_median(
                grid.astype(np.float64), ijk, radius=1, order=0
            )
            if grid_sample > 0.5:
                t_arrival = self._sample_neighbor_median(ft, ijk, radius=1, order=1)
                if np.isfinite(t_arrival) and t_arrival >= 0.0:
                    return float(t_arrival)
        except Exception:
            pass
        return float(np.nan)

    def _plot(self) -> None:
        self.figure.clear()
        flow = self._result.flow_result
        if flow is None:
            ax = self.figure.add_subplot(111)
            self._style_axes(ax)
            ax.text(
                0.5,
                0.5,
                "Akış sonucu yok.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#e0e0e0",
                fontsize=12,
            )
            self.canvas.draw()
            return

        ft = flow.fill_time
        vm = flow.velocity_magnitude
        vel = flow.velocity
        grid = self._result.grid
        if (
            ft is None
            or vm is None
            or grid is None
            or ft.size == 0
            or vm.size == 0
            or grid.size == 0
        ):
            ax = self.figure.add_subplot(111)
            self._style_axes(ax)
            ax.text(
                0.5,
                0.5,
                "Hız / dolum zamanı verisi yok.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#e0e0e0",
                fontsize=12,
            )
            self.canvas.draw()
            return

        metric = self._metric_combo.currentData()
        show_geometry = self._radius_check.isChecked()

        metrics = self._compute_per_voxel_metrics(self._result, ft, vm, grid)
        m_arr = metrics[metric]
        valid = metrics["valid"]

        # Shared color scale from the selected metric.
        m_vals = np.asarray(m_arr[valid], dtype=np.float64)
        if m_vals.size:
            vmin = float(np.percentile(m_vals, 2.0))
            vmax = float(np.percentile(m_vals, 98.0))
            if vmax <= vmin:
                vmax = vmin + 1e-6
        else:
            vmin, vmax = 0.0, 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = self._cmap_for_metric(metric)

        if show_geometry:
            gs = self.figure.add_gridspec(
                3, 2, height_ratios=[1, 1, 0.65], width_ratios=[1, 0.04]
            )
            ax_time = self.figure.add_subplot(gs[0, 0])
            ax_path = self.figure.add_subplot(gs[1, 0])
            ax_geom = self.figure.add_subplot(gs[2, 0])
            cax = self.figure.add_subplot(gs[:2, 1])
        else:
            gs = self.figure.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 0.04])
            ax_time = self.figure.add_subplot(gs[0, 0])
            ax_path = self.figure.add_subplot(gs[1, 0])
            ax_geom = None
            cax = self.figure.add_subplot(gs[:, 1])

        self._style_axes(ax_time)
        self._style_axes(ax_path)
        if ax_geom is not None:
            self._style_axes(ax_geom)
        cax.set_facecolor("#1a1a2e")
        cax.tick_params(colors="#e0e0e0")

        self._plot_metric_vs_time(ax_time, ft, m_arr, valid, metric, cmap, norm)
        self._plot_path_profile(ax_path, metric, cmap, norm)
        if ax_geom is not None:
            self._plot_path_geometry(ax_geom)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = self.figure.colorbar(sm, cax=cax)
        cbar.ax.set_ylabel(self._metric_label(metric), color="#e0e0e0")
        cbar.ax.tick_params(colors="#e0e0e0")

        self.canvas.draw()

    def _plot_metric_vs_time(
        self,
        ax,
        ft: np.ndarray,
        m_arr: np.ndarray,
        valid: np.ndarray,
        metric: str,
        cmap,
        norm,
    ) -> None:
        if not valid.any():
            ax.text(
                0.5,
                0.5,
                "Geçerli veri yok.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#e0e0e0",
                fontsize=11,
            )
            return

        ft_vals = np.asarray(ft[valid], dtype=np.float64)
        m_vals = np.asarray(m_arr[valid], dtype=np.float64)

        max_t = float(np.percentile(ft_vals, 99.9))
        if max_t <= 0.0:
            max_t = float(ft_vals.max())

        n_bins = 80
        bins = np.linspace(0.0, max_t, n_bins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        means = np.full(n_bins, np.nan)
        medians = np.full(n_bins, np.nan)
        p10 = np.full(n_bins, np.nan)
        p25 = np.full(n_bins, np.nan)
        p75 = np.full(n_bins, np.nan)
        p90 = np.full(n_bins, np.nan)

        for i in range(n_bins):
            mask = (ft_vals >= bins[i]) & (ft_vals < bins[i + 1])
            if mask.any():
                vals = m_vals[mask]
                means[i] = float(np.mean(vals))
                medians[i] = float(np.median(vals))
                p10[i] = float(np.percentile(vals, 10.0))
                p25[i] = float(np.percentile(vals, 25.0))
                p75[i] = float(np.percentile(vals, 75.0))
                p90[i] = float(np.percentile(vals, 90.0))

        valid_bins = ~np.isnan(means)

        for i in range(n_bins - 1):
            if valid_bins[i] and valid_bins[i + 1]:
                avg_m = 0.5 * (means[i] + means[i + 1])
                color = to_hex(cmap(norm(avg_m)))
                ax.fill_between(
                    [centers[i], centers[i + 1]],
                    [p10[i], p10[i + 1]],
                    [p90[i], p90[i + 1]],
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                    zorder=1,
                )

        x = centers[valid_bins]
        y = means[valid_bins]
        if x.size >= 2:
            points = np.column_stack([x, y]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            colors = y[:-1]
            lc = LineCollection(
                segments,
                cmap=cmap,
                norm=norm,
                linewidth=3.0,
                capstyle="round",
                joinstyle="round",
                zorder=3,
                label="Ortalama",
            )
            lc.set_array(colors)
            ax.add_collection(lc)

        if p25[valid_bins].size:
            ax.plot(
                centers[valid_bins],
                p25[valid_bins],
                color="white",
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
                zorder=2,
                label="25-75 yüzdelik",
            )
            ax.plot(
                centers[valid_bins],
                p75[valid_bins],
                color="white",
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
                zorder=2,
            )

        flow = self._result.flow_result
        if flow and flow.gating_nodes:
            node_times: List[float] = []
            node_ms: List[float] = []
            node_names: List[str] = []
            for gn in flow.gating_nodes:
                t_arrival = self._sample_node_arrival_time(self._result, gn, ft, self._result.grid)
                if np.isfinite(t_arrival) and t_arrival >= 0.0:
                    node_times.append(t_arrival)
                    nm = self._compute_node_metrics(gn)
                    node_ms.append(nm.get(metric, 0.0))
                    node_names.append(gn.name)
            if node_times:
                ax.scatter(
                    node_times,
                    node_ms,
                    c=node_ms,
                    cmap=cmap,
                    norm=norm,
                    s=120,
                    zorder=5,
                    edgecolors="white",
                    linewidths=1.2,
                    label="Gating düğümleri",
                )
                for i, (x_n, y_n, name) in enumerate(zip(node_times, node_ms, node_names)):
                    xoff = 10 if i % 2 == 0 else -10
                    yoff = 12 if (i // 2) % 2 == 0 else -14
                    ax.annotate(
                        name,
                        (x_n, y_n),
                        textcoords="offset points",
                        xytext=(xoff, yoff),
                        fontsize=7,
                        color="#e0e0e0",
                        ha="left" if xoff > 0 else "right",
                        va="bottom" if yoff > 0 else "top",
                        bbox=dict(boxstyle="round,pad=0.25", fc="#264653", ec="#e0e0e0", alpha=0.85),
                        arrowprops=dict(arrowstyle="-", color="#e0e0e0", lw=0.5),
                        zorder=6,
                    )

        if flow and flow.fill_time_s > 0.0:
            ax.axvline(
                flow.fill_time_s,
                color="#f4a261",
                linestyle="--",
                linewidth=1.5,
                zorder=2,
                label=f"Toplam dolum: {flow.fill_time_s:.2f} s",
            )

        ax.set_xlabel("Dolum zamanı (s)")
        ax.set_ylabel(self._metric_label(metric))
        ax.set_title(f"Darcy ön cephe {self._metric_label(metric)} - Dolum zamanı")
        ax.set_xlim(0.0, max_t * 1.05 if max_t > 0.0 else 1.0)
        if m_vals.size:
            ax.set_ylim(0.0, float(np.percentile(m_vals, 99.5)) * 1.15)
        ax.legend(loc="upper right", fontsize=8, facecolor="#16213e", edgecolor="#e0e0e0")
        ax.text(
            0.02,
            0.02,
            f"ρ={self._rho:.0f} kg/m³, μ={self._mu:.4f} Pa·s, g={self._g:.2f} m/s²",
            transform=ax.transAxes,
            fontsize=7,
            color="#e0e0e0",
            verticalalignment="bottom",
        )

    def _build_gating_paths(self) -> Tuple[Optional[str], Dict[str, List[GatingNode]]]:
        flow = self._result.flow_result
        if flow is None or not flow.gating_nodes:
            return None, {}

        children: Dict[str, List[GatingNode]] = {}
        root: Optional[str] = None
        source_node: Optional[GatingNode] = None
        for gn in flow.gating_nodes:
            if gn.body_type.startswith("SOURCE"):
                source_node = gn
                parts = gn.name.split(" → ")
                if len(parts) == 2:
                    root = parts[1]
                continue
            parts = gn.name.split(" → ")
            if len(parts) == 2:
                up, _down = parts
                children.setdefault(up, []).append(gn)

        if root is None and source_node is None and flow.gating_nodes:
            root = flow.gating_nodes[0].name.split(" → ")[0]

        paths: List[List[GatingNode]] = []

        def _walk(current: str, path: List[GatingNode]):
            if current not in children or not children[current]:
                paths.append(list(path))
                return
            for child in children[current]:
                path.append(child)
                _walk(child.name.split(" → ")[1], path)
                path.pop()

        if source_node is not None and root is not None:
            _walk(root, [source_node])
        elif root is not None:
            _walk(root, [])

        if not paths:
            paths = [list(flow.gating_nodes)]

        return root, children, paths

    def _plot_path_profile(
        self,
        ax,
        metric: str,
        cmap,
        norm,
    ) -> None:
        flow = self._result.flow_result
        if flow is None or not flow.gating_nodes:
            ax.text(
                0.5,
                0.5,
                "Gating düğümü yok, akış yolu profili çizilemiyor.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#e0e0e0",
                fontsize=11,
            )
            return

        root, children, paths = self._build_gating_paths()

        all_x: List[float] = []
        all_y: List[float] = []
        all_nodes: List[Tuple[float, float, str, Dict[str, float]]] = []

        for path_idx, path in enumerate(paths):
            distances = [0.0]
            metric_values = []
            names = []
            prev = None
            for gn in path:
                if prev is not None:
                    d = float(
                        np.linalg.norm(np.asarray(gn.centroid_mm) - np.asarray(prev.centroid_mm))
                        / 1000.0
                    )
                    distances.append(distances[-1] + d)
                prev = gn

                nm = self._compute_node_metrics(gn)
                m_val = nm.get(metric, 0.0)
                metric_values.append(m_val)
                names.append(gn.name)
                all_x.append(distances[-1])
                all_y.append(m_val)
                all_nodes.append((distances[-1], m_val, gn.name, nm))

            distances = np.asarray(distances, dtype=np.float64)
            metric_values = np.asarray(metric_values, dtype=np.float64)

            if distances.size >= 2:
                points = np.column_stack([distances, metric_values]).reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                colors = metric_values[:-1]
                lc = LineCollection(
                    segments,
                    cmap=cmap,
                    norm=norm,
                    linewidth=3.0,
                    capstyle="round",
                    joinstyle="round",
                    zorder=2,
                    label=f"Meme {path_idx + 1}" if len(paths) > 1 else "Akış yolu",
                )
                lc.set_array(colors)
                ax.add_collection(lc)

            ax.scatter(
                distances,
                metric_values,
                c=metric_values,
                cmap=cmap,
                norm=norm,
                s=100,
                zorder=4,
                edgecolors="white",
                linewidths=1.2,
            )

            for i, (x_n, y_n, name) in enumerate(zip(distances, metric_values, names)):
                xoff = 10 if i % 2 == 0 else -10
                yoff = 12 if (i // 2) % 2 == 0 else -14
                ax.annotate(
                    name,
                    (x_n, y_n),
                    textcoords="offset points",
                    xytext=(xoff, yoff),
                    fontsize=7,
                    color="#e0e0e0",
                    ha="left" if xoff > 0 else "right",
                    va="bottom" if yoff > 0 else "top",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#264653", ec="#e0e0e0", alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color="#e0e0e0", lw=0.5),
                    zorder=5,
                )

        if all_x:
            pad = 0.05 * (max(all_x) - min(all_x)) if max(all_x) > min(all_x) else 0.01
            ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
            ax.set_ylim(0.0, max(all_y) * 1.15)

        ax.set_xlabel("Akış yolu mesafesi (m)")
        ax.set_ylabel(self._metric_label(metric))
        ax.set_title(f"Akış yolu {self._metric_label(metric)} profili")
        if len(paths) > 1:
            ax.legend(loc="upper right", fontsize=8, facecolor="#16213e", edgecolor="#e0e0e0")
        ax.text(
            0.02,
            0.02,
            "Not: Her nokta Q/A kesit hızı üzerinden Re/Fr/Dh hesaplanır.",
            transform=ax.transAxes,
            fontsize=7,
            color="#e0e0e0",
            verticalalignment="bottom",
        )

    def _plot_path_geometry(self, ax) -> None:
        """Bottom panel: cross-section area and hydraulic diameter along the gating path."""
        flow = self._result.flow_result
        if flow is None or not flow.gating_nodes:
            ax.text(
                0.5,
                0.5,
                "Geometri verisi yok.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#e0e0e0",
                fontsize=11,
            )
            return

        _, _, paths = self._build_gating_paths()
        ax2 = ax.twinx()
        self._style_axes(ax2)
        ax2.tick_params(colors="#e0e0e0")
        ax2.spines["right"].set_color("#e0e0e0")

        all_diameters: List[float] = []
        all_areas: List[float] = []
        all_x: List[float] = []

        for path in paths:
            distances = [0.0]
            areas = []
            diameters = []
            prev = None
            for gn in path:
                if prev is not None:
                    d = float(
                        np.linalg.norm(np.asarray(gn.centroid_mm) - np.asarray(prev.centroid_mm))
                        / 1000.0
                    )
                    distances.append(distances[-1] + d)
                prev = gn
                nm = self._compute_node_metrics(gn)
                areas.append(nm["area_cm2"])
                diameters.append(nm["diameter"])
                all_x.append(distances[-1])
                all_areas.append(nm["area_cm2"])
                all_diameters.append(nm["diameter"])

            distances = np.asarray(distances, dtype=np.float64)
            areas = np.asarray(areas, dtype=np.float64)
            diameters = np.asarray(diameters, dtype=np.float64)

            ax.fill_between(distances, 0, areas, alpha=0.2, color="#2a9d8f")
            ax.plot(distances, areas, color="#2a9d8f", linewidth=2.0, label="Kesit alanı")
            ax2.plot(distances, diameters, color="#e9c46a", linewidth=2.0, linestyle="--", label="Dh")

        ax.set_xlabel("Akış yolu mesafesi (m)")
        ax.set_ylabel("Kesit alanı (cm²)", color="#2a9d8f")
        ax2.set_ylabel("Hidrolik çap Dh (mm)", color="#e9c46a")
        ax.set_title("Kesit geometrisi (radyus etkisi)")
        ax.tick_params(axis="y", colors="#2a9d8f")
        ax2.tick_params(axis="y", colors="#e9c46a")

        if all_x:
            pad = 0.05 * (max(all_x) - min(all_x)) if max(all_x) > min(all_x) else 0.01
            ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
            ax.set_ylim(0.0, max(all_areas) * 1.2)
            ax2.set_ylim(0.0, max(all_diameters) * 1.2)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8,
                  facecolor="#16213e", edgecolor="#e0e0e0")
