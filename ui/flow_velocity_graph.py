"""Simple 2-D velocity vs. fill-time graph for the Darcy flow result."""
from typing import Optional

import numpy as np
from scipy import ndimage

from PyQt6 import QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from core.types import AnalysisResult


class FlowVelocityGraph(QtWidgets.QDialog):
    """Popup dialog showing Darcy front velocity as a function of fill time."""

    def __init__(self, result: AnalysisResult, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Akış Hızı - Zaman Grafiği")
        self.setMinimumSize(800, 600)

        layout = QtWidgets.QVBoxLayout(self)
        self.figure = Figure(figsize=(8, 6), dpi=100, tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)

        self._plot(result)

    def _plot(self, result: AnalysisResult) -> None:
        ax = self.figure.add_subplot(111)
        flow = result.flow_result
        if flow is None:
            ax.text(0.5, 0.5, "Akış sonucu yok.", ha="center", va="center", transform=ax.transAxes)
            self.canvas.draw()
            return

        ft = flow.fill_time
        vm = flow.velocity_magnitude
        if ft is None or vm is None or ft.size == 0 or vm.size == 0:
            ax.text(0.5, 0.5, "Hız / dolum zamanı verisi yok.", ha="center", va="center", transform=ax.transAxes)
            self.canvas.draw()
            return

        # Use the same metal mask as the main analysis (result.grid > 0).
        metal = result.grid > 0
        valid = metal & np.isfinite(ft) & np.isfinite(vm) & (vm > 0.0)
        if not valid.any():
            ax.text(0.5, 0.5, "Geçerli hız verisi yok.", ha="center", va="center", transform=ax.transAxes)
            self.canvas.draw()
            return

        ft_vals = np.asarray(ft[valid], dtype=np.float64)
        vm_vals = np.asarray(vm[valid], dtype=np.float64)

        max_t = float(np.percentile(ft_vals, 99.9))
        if max_t <= 0.0:
            max_t = float(ft_vals.max())

        # Bin by fill time to reduce noise and produce a readable line.
        n_bins = 120
        bins = np.linspace(0.0, max_t, n_bins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        means = np.full(n_bins, np.nan)
        p25 = np.full(n_bins, np.nan)
        p75 = np.full(n_bins, np.nan)
        for i in range(n_bins):
            mask = (ft_vals >= bins[i]) & (ft_vals < bins[i + 1])
            if mask.any():
                vals = vm_vals[mask]
                means[i] = float(np.mean(vals))
                p25[i] = float(np.percentile(vals, 25.0))
                p75[i] = float(np.percentile(vals, 75.0))

        valid_bins = ~np.isnan(means)
        ax.fill_between(
            centers[valid_bins],
            p25[valid_bins],
            p75[valid_bins],
            alpha=0.25,
            color="tab:red",
            label="25-75 yüzdelik",
        )
        ax.plot(
            centers[valid_bins],
            means[valid_bins],
            color="tab:red",
            linewidth=2.0,
            label="Ön cephe hızı (m/s)",
        )

        # Mark gating node velocities at their estimated arrival times.
        if flow.gating_nodes:
            node_times = []
            node_vels = []
            node_names = []
            origin = np.asarray(result.origin_mm, dtype=np.float64)
            dx = float(result.dx_mm)
            for gn in flow.gating_nodes:
                pos = np.asarray(gn.centroid_mm, dtype=np.float64)
                ijk = (pos - origin) / dx
                try:
                    sampled = ndimage.map_coordinates(
                        ft,
                        ijk[:, None],
                        order=1,
                        mode="constant",
                        cval=np.nan,
                    )
                    t_arrival = float(np.asarray(sampled).flat[0])
                except Exception:
                    t_arrival = np.nan
                if np.isfinite(t_arrival):
                    node_times.append(t_arrival)
                    node_vels.append(float(gn.velocity_m_s))
                    node_names.append(gn.name)
            if node_times:
                ax.scatter(
                    node_times,
                    node_vels,
                    color="tab:blue",
                    zorder=5,
                    s=60,
                    label="Gating düğümleri",
                )
                for x, y, name in zip(node_times, node_vels, node_names):
                    ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)

        if flow.fill_time_s > 0.0:
            ax.axvline(
                flow.fill_time_s,
                color="black",
                linestyle="--",
                linewidth=1.0,
                label=f"Toplam dolum: {flow.fill_time_s:.2f} s",
            )

        ax.set_xlabel("Dolum zamanı (s)")
        ax.set_ylabel("Hız büyüklüğü (m/s)")
        ax.set_title("Darcy ön cephe hızı - Dolum zamanı")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        ax.text(
            0.02,
            0.02,
            "Not: Darcy hız alanı ve dolum zamanından türetilmiştir; anlık türbülans/serbest yüzey içermez.",
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="bottom",
        )
        self.canvas.draw()
