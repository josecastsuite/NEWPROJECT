"""Enhanced 2-D flow velocity graphs for the Darcy flow result.

The dialog now contains two panels:

1. Akış hızı - Dolum zamanı:  Darcy ön cephe hızının dolum süresi boyunca
   değişimi.  Çizgi renkli, 10.-90. yüzdelik bant şeffaf renkli ve gating
   düğümlerinin ulaşma zamanları nokta olarak işaretlenmiş.

2. Akış yolu profili:  Döküm ağzından (sprue) her meme girişine kadar gating
   düğümleri boyunca hız profili.  Her nokta gerçek kesit ortalaması hızı
   (Q/A) ve Darcy hız alanından örneklenen |v| ile renklendirilir.  Çizgiler
   meme adedi kadardır, renk skalası turbo'dur.

Tüm hız değerleri gerçek Darcy vektör alanından (flow_result.velocity) veya
Q/A süreklilik hesabından gelir; görselleştirme pürüzsüzleştirme hız
verilerini kaydırmaz.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from PyQt6 import QtWidgets
from matplotlib import cm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_hex
from matplotlib.figure import Figure

from core.types import AnalysisResult, GatingNode


class FlowVelocityGraph(QtWidgets.QDialog):
    """Popup dialog showing Darcy velocity vs fill time and flow-path profile."""

    def __init__(self, result: AnalysisResult, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Akış Hızı - Dolum Zamanı ve Akış Yolu Profili")
        self.setMinimumSize(1100, 900)
        self._setup_matplotlib_style()

        layout = QtWidgets.QVBoxLayout(self)
        self.figure = Figure(figsize=(11, 9), dpi=100, tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)

        self._plot(result)

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

    def _plot(self, result: AnalysisResult) -> None:
        self.figure.clear()
        flow = result.flow_result
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
        grid = result.grid
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

        metal = grid > 0
        valid = metal & np.isfinite(ft) & np.isfinite(vm) & (vm > 0.0)

        gs = self.figure.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 0.04])
        ax_time = self.figure.add_subplot(gs[0, 0])
        ax_path = self.figure.add_subplot(gs[1, 0])
        cax = self.figure.add_subplot(gs[:, 1])

        self._style_axes(ax_time)
        self._style_axes(ax_path)
        cax.set_facecolor("#1a1a2e")
        cax.tick_params(colors="#e0e0e0")

        # Shared velocity scale from the actual Darcy field.
        vm_vals = np.asarray(vm[valid], dtype=np.float64)
        if vm_vals.size:
            vmin = float(np.percentile(vm_vals, 2.0))
            vmax = float(np.percentile(vm_vals, 98.0))
            if vmax <= vmin:
                vmax = vmin + 1e-6
        else:
            vmin, vmax = 0.0, 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap("turbo")

        self._plot_velocity_vs_time(ax_time, result, ft, vm, grid, valid, cmap, norm)
        self._plot_path_profile(ax_path, result, cmap, norm)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = self.figure.colorbar(sm, cax=cax)
        cbar.ax.set_ylabel("|v| (m/s)", color="#e0e0e0")
        cbar.ax.tick_params(colors="#e0e0e0")

        self.canvas.draw()

    def _plot_velocity_vs_time(
        self,
        ax,
        result: AnalysisResult,
        ft: np.ndarray,
        vm: np.ndarray,
        grid: np.ndarray,
        valid: np.ndarray,
        cmap,
        norm,
    ) -> None:
        if not valid.any():
            ax.text(
                0.5,
                0.5,
                "Geçerli hız verisi yok.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#e0e0e0",
                fontsize=11,
            )
            return

        ft_vals = np.asarray(ft[valid], dtype=np.float64)
        vm_vals = np.asarray(vm[valid], dtype=np.float64)

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
        counts = np.zeros(n_bins, dtype=int)

        for i in range(n_bins):
            mask = (ft_vals >= bins[i]) & (ft_vals < bins[i + 1])
            if mask.any():
                vals = vm_vals[mask]
                means[i] = float(np.mean(vals))
                medians[i] = float(np.median(vals))
                p10[i] = float(np.percentile(vals, 10.0))
                p25[i] = float(np.percentile(vals, 25.0))
                p75[i] = float(np.percentile(vals, 75.0))
                p90[i] = float(np.percentile(vals, 90.0))
                counts[i] = int(mask.sum())

        valid_bins = ~np.isnan(means)

        # Gradient fill between p10 and p90 per bin segment.
        for i in range(n_bins - 1):
            if valid_bins[i] and valid_bins[i + 1]:
                avg_v = 0.5 * (means[i] + means[i + 1])
                color = to_hex(cmap(norm(avg_v)))
                ax.fill_between(
                    [centers[i], centers[i + 1]],
                    [p10[i], p10[i + 1]],
                    [p90[i], p90[i + 1]],
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                    zorder=1,
                )

        # Main colored line of mean velocity.
        x = centers[valid_bins]
        y = means[valid_bins]
        if x.size >= 2:
            points = np.column_stack([x, y]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            colors = y[:-1]  # color by segment start
            lc = LineCollection(
                segments,
                cmap=cmap,
                norm=norm,
                linewidth=3.0,
                capstyle="round",
                joinstyle="round",
                zorder=3,
                label="Ortalama ön cephe hızı",
            )
            lc.set_array(colors)
            ax.add_collection(lc)

        # Percentile boundaries.
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

        # Mark gating nodes with Darcy-sampled |v| at the estimated arrival time.
        flow = result.flow_result
        origin = np.asarray(result.origin_mm, dtype=np.float64)
        dx = float(result.dx_mm)
        dx_m = dx / 1000.0
        if flow and flow.gating_nodes:
            node_times: List[float] = []
            node_vels: List[float] = []
            node_names: List[str] = []
            for gn in flow.gating_nodes:
                pos = np.asarray(gn.centroid_mm, dtype=np.float64)
                ijk = (pos - origin) / dx
                t_arrival = np.nan
                try:
                    grid_sample = ndimage.map_coordinates(
                        grid.astype(np.float64),
                        ijk[:, None],
                        order=0,
                        mode="nearest",
                        cval=0.0,
                    )
                    if float(np.asarray(grid_sample).flat[0]) > 0.5:
                        sampled_t = ndimage.map_coordinates(
                            ft,
                            ijk[:, None],
                            order=1,
                            mode="nearest",
                            cval=np.nan,
                        )
                        t_arrival = float(np.asarray(sampled_t).flat[0])
                except Exception:
                    pass
                if np.isfinite(t_arrival) and t_arrival >= 0.0:
                    node_times.append(t_arrival)
                    node_vels.append(float(gn.velocity_m_s))
                    node_names.append(gn.name)
            if node_times:
                ax.scatter(
                    node_times,
                    node_vels,
                    c=node_vels,
                    cmap=cmap,
                    norm=norm,
                    s=120,
                    zorder=5,
                    edgecolors="white",
                    linewidths=1.2,
                    label="Gating düğümleri (Q/A |v|)",
                )
                for i, (x_n, y_n, name) in enumerate(zip(node_times, node_vels, node_names)):
                    xoff = 10 if i % 2 == 0 else -10
                    yoff = 12 if (i // 2) % 2 == 0 else -14
                    ax.annotate(
                        name,
                        (x_n, y_n),
                        textcoords="offset points",
                        xytext=(xoff, yoff),
                        fontsize=8,
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
        ax.set_ylabel("Hız büyüklüğü |v| (m/s)")
        ax.set_title("Darcy ön cephe hızı - Dolum zamanı")
        ax.set_xlim(0.0, max_t * 1.05 if max_t > 0.0 else 1.0)
        if vm_vals.size:
            ax.set_ylim(0.0, float(np.percentile(vm_vals, 99.5)) * 1.15)
        ax.legend(loc="upper right", fontsize=8, facecolor="#16213e", edgecolor="#e0e0e0")
        ax.text(
            0.02,
            0.02,
            "Not: Renkler gerçek |v|=sqrt(vx²+vy²+vz²) değerlerine göre turbo skalasıyla eşlenir.",
            transform=ax.transAxes,
            fontsize=7,
            color="#e0e0e0",
            verticalalignment="bottom",
        )

    def _plot_path_profile(
        self,
        ax,
        result: AnalysisResult,
        cmap,
        norm,
    ) -> None:
        flow = result.flow_result
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

        origin = np.asarray(result.origin_mm, dtype=np.float64)
        dx = float(result.dx_mm)

        # Build a tree from gating node names (UP -> DOWN).
        root: Optional[str] = None
        children: Dict[str, List[GatingNode]] = {}
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

        if root is None and source_node is None:
            # Fallback: first node is root.
            root = flow.gating_nodes[0].name.split(" → ")[0] if flow.gating_nodes else None

        # Gather root-to-leaf paths.
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
            # Degenerate graph: just plot nodes in order.
            paths = [list(flow.gating_nodes)]

        all_x: List[float] = []
        all_y: List[float] = []

        for path_idx, path in enumerate(paths):
            distances = [0.0]
            velocities = []
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

                # Use the physically correct section-average velocity Q/A for this
                # gating node.  Darcy |v| at a single centroid is noisy near
                # boundaries/resampled faces; Q/A is the reliable value we already
                # compute for labels and reports.
                v_node = float(gn.velocity_m_s) if gn.velocity_m_s > 1e-12 else 0.0
                velocities.append(v_node)
                names.append(gn.name)
                all_x.append(distances[-1])
                all_y.append(v_node)

            distances = np.asarray(distances, dtype=np.float64)
            velocities = np.asarray(velocities, dtype=np.float64)

            if distances.size >= 2:
                points = np.column_stack([distances, velocities]).reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                colors = velocities[:-1]
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
                velocities,
                c=velocities,
                cmap=cmap,
                norm=norm,
                s=100,
                zorder=4,
                edgecolors="white",
                linewidths=1.2,
            )

            # Annotate every node; stagger to reduce overlap.
            for i, (x_n, y_n, name) in enumerate(zip(distances, velocities, names)):
                xoff = 10 if i % 2 == 0 else -10
                yoff = 12 if (i // 2) % 2 == 0 else -14
                ax.annotate(
                    name,
                    (x_n, y_n),
                    textcoords="offset points",
                    xytext=(xoff, yoff),
                    fontsize=8,
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
        ax.set_ylabel("Hız büyüklüğü |v| (m/s)")
        ax.set_title("Akış yolu hız profili (Q/A kesit hızı)")
        if len(paths) > 1:
            ax.legend(loc="upper right", fontsize=8, facecolor="#16213e", edgecolor="#e0e0e0")
        ax.text(
            0.02,
            0.02,
            "Not: Her nokta o kesitteki Q/A ortalama hızıdır; düz çizgi segmentleri iki düğüm arasındadır.",
            transform=ax.transAxes,
            fontsize=7,
            color="#e0e0e0",
            verticalalignment="bottom",
        )
