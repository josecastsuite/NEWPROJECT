"""Flow-velocity profile dialog for JoseCast Analyzer.

The graph is built directly from the Darcy solver's gating-node list, so every
plotted point is a real throat/contact section with a known Q, A and v.
Section changes (runner narrowing, gate expansion, multiple runners drawn as one
body, etc.) therefore immediately move the velocity/Re/Froude curves.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt6 import QtWidgets
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from core.materials import get_alloy
from core.types import AnalysisResult, GatingNode


_DEFAULT_TARGET_V = (0.25, 1.0)  # m/s


class FlowVelocityGraph(QtWidgets.QDialog):
    """Popup showing the Darcy flow profile along the gating system."""

    def __init__(
        self,
        result: AnalysisResult,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._result = result
        self.setWindowTitle("Akış Hızı Profili")
        self.setMinimumSize(1100, 950)

        self._rho, self._mu, self._g = self._material_and_gravity(result)
        self._target_v = self._target_velocity(result)
        self._branches, self._branch_names = self._build_branches(result)

        self._setup_matplotlib_style()
        self._setup_ui()
        self._plot()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _setup_matplotlib_style(self) -> None:
        import matplotlib as mpl

        rc = {
            "figure.facecolor": "#F8FAFC",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#334155",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "text.color": "#334155",
            "grid.color": "#E2E8F0",
            "grid.alpha": 0.6,
            "axes.grid": True,
            "axes.titlesize": 11,
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

    def _setup_ui(self) -> None:
        self.setStyleSheet("background-color: #F8FAFC;")
        layout = QtWidgets.QVBoxLayout(self)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Renk metriği:"))
        self._metric_combo = QtWidgets.QComboBox()
        self._metric_combo.addItem("Hız |v| (m/s)", "velocity")
        self._metric_combo.addItem("Reynolds (Re)", "reynolds")
        self._metric_combo.addItem("Froude (Fr)", "froude")
        self._metric_combo.addItem("Hidrolik Çap Dh (mm)", "diameter")
        self._metric_combo.setCurrentIndex(0)
        self._metric_combo.currentIndexChanged.connect(self._plot)
        controls.addWidget(self._metric_combo)

        controls.addWidget(QtWidgets.QLabel("Kol:"))
        self._branch_combo = QtWidgets.QComboBox()
        self._branch_combo.addItem("Tüm kollar")
        for name in self._branch_names:
            self._branch_combo.addItem(name)
        self._branch_combo.currentIndexChanged.connect(self._plot)
        controls.addWidget(self._branch_combo)

        self._target_check = QtWidgets.QCheckBox("Hedef hız bandı")
        self._target_check.setChecked(True)
        self._target_check.stateChanged.connect(self._plot)
        controls.addWidget(self._target_check)

        self._label_check = QtWidgets.QCheckBox("Düğüm etiketleri")
        self._label_check.setChecked(True)
        self._label_check.stateChanged.connect(self._plot)
        controls.addStretch()
        layout.addLayout(controls)

        self.figure = Figure(figsize=(11, 11), dpi=100, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _material_and_gravity(result: AnalysisResult) -> Tuple[float, float, float]:
        rho = 7000.0
        mu = 0.006
        g = 9.81
        cp = getattr(result, "casting_params", None)
        if cp is not None:
            rho = float(cp.rho_liquid_kg_m3) if cp.rho_liquid_kg_m3 > 0 else rho
            mu = float(cp.viscosity_pa_s) if cp.viscosity_pa_s > 0 else mu
            gv = getattr(cp, "gravity_vector", None)
            if gv:
                try:
                    gv = np.asarray(gv, dtype=np.float64)
                    g = float(np.linalg.norm(gv)) or 9.81
                except Exception:
                    pass
        else:
            try:
                alloy = get_alloy(result.alloy_key)
                rho = float(getattr(alloy, "rho_liquid_kg_m3", rho))
                mu = float(getattr(alloy, "viscosity_pa_s", mu))
            except Exception:
                pass
        return rho, mu, g

    def _target_velocity(self, result: AnalysisResult) -> Tuple[float, float]:
        gr = getattr(result, "gate_result", None)
        if gr is not None:
            vmin = float(getattr(gr, "target_v_min_m_s", 0.0) or 0.0)
            vmax = float(getattr(gr, "target_v_max_m_s", 0.0) or 0.0)
            if vmin > 0 and vmax > vmin:
                return vmin, vmax
        return _DEFAULT_TARGET_V

    def _build_branches(
        self, result: AnalysisResult
    ) -> Tuple[List[List[GatingNode]], List[str]]:
        """Return every source -> ingate path as a list of GatingNodes."""
        flow = getattr(result, "flow_result", None)
        nodes = getattr(flow, "gating_nodes", None) or []
        if not nodes:
            return [], []

        # down_name -> node is unique in a tree.
        down_to_node: Dict[str, GatingNode] = {}
        source_node: Optional[GatingNode] = None
        for n in nodes:
            try:
                up, down = n.name.split(" → ", 1)
            except ValueError:
                continue
            if up == "Kaynak":
                source_node = n
            down_to_node[down] = n

        if source_node is None:
            # Fallback: first node is treated as source.
            source_node = nodes[0]

        up_names = {n.name.split(" → ", 1)[0] for n in nodes if " → " in n.name}
        leaves = [n for n in nodes if n is not source_node and n.name.split(" → ", 1)[1] not in up_names]

        branches: List[List[GatingNode]] = []
        for leaf in leaves:
            path = [leaf]
            while True:
                up = path[0].name.split(" → ", 1)[0]
                if up == "Kaynak" or up not in down_to_node:
                    break
                path.insert(0, down_to_node[up])
            branches.append(path)

        # If no leaf found (unusual), show the whole chain as one branch.
        if not branches:
            branches = [[source_node] + [n for n in nodes if n is not source_node]]

        names = [self._branch_name(b, i) for i, b in enumerate(branches)]
        return branches, names

    @staticmethod
    def _branch_name(path: List[GatingNode], index: int) -> str:
        """Return a human name for the branch (prefer the ingate name)."""
        if not path:
            return f"Kol {index + 1}"
        up, down = path[-1].name.split(" → ", 1)
        if down in ("Parça", "PART", "part"):
            return up
        return down

    def _node_metrics(self, node: GatingNode) -> Dict[str, float]:
        v = float(node.velocity_m_s)
        a_cm2 = float(node.section_area_cm2)
        a_m2 = a_cm2 * 1e-4
        if a_m2 > 1e-18:
            dh_m = 2.0 * np.sqrt(a_m2 / np.pi)
        else:
            dh_m = 0.0
        re = self._rho * v * dh_m / self._mu if dh_m > 0 and self._mu > 0 else 0.0
        fr = v / np.sqrt(self._g * dh_m) if dh_m > 0 and v > 0 else 0.0
        return {
            "velocity": v,
            "area_cm2": a_cm2,
            "area_m2": a_m2,
            "dh_mm": dh_m * 1000.0,
            "reynolds": re,
            "froude": fr,
            "flow_rate_m3_s": float(getattr(node, "flow_rate_m3_s", 0.0)),
        }

    def _metric_for_node(self, node: GatingNode, metric: str) -> float:
        m = self._node_metrics(node)
        return float(m.get(metric, 0.0))

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def _style_ax(self, ax) -> None:
        ax.set_facecolor("#FFFFFF")
        ax.tick_params(colors="#334155")
        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")
        ax.xaxis.label.set_color("#334155")
        ax.yaxis.label.set_color("#334155")
        ax.title.set_color("#334155")
        ax.grid(True, alpha=0.6, color="#E2E8F0")

    def _plot(self, *_) -> None:
        self.figure.clear()
        if not self._branches:
            ax = self.figure.add_subplot(111)
            self._style_ax(ax)
            ax.text(
                0.5,
                0.5,
                "Akış düğümü verisi yok;\nanaliz sonucu gating düğümleri içermiyor.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color="#64748B",
            )
            self.canvas.draw()
            return

        metric = self._metric_combo.currentData()
        selected_branch = self._branch_combo.currentIndex() - 1

        # ------------------------------------------------------------------
        # 1. Build per-branch 1D data along the real gating network.
        # ------------------------------------------------------------------
        branch_data: List[Dict] = []
        metric_values: List[float] = []
        for path in self._branches:
            s = [0.0]
            for i in range(1, len(path)):
                c0 = np.asarray(path[i - 1].centroid_mm, dtype=np.float64)
                c1 = np.asarray(path[i].centroid_mm, dtype=np.float64)
                s.append(s[-1] + float(np.linalg.norm(c1 - c0)))
            metrics = [self._metric_for_node(n, metric) for n in path]
            vels = [self._metric_for_node(n, "velocity") for n in path]
            metric_values.extend(metrics)
            branch_data.append(
                {
                    "path": path,
                    "s": np.asarray(s, dtype=np.float64),
                    "metrics": np.asarray(metrics, dtype=np.float64),
                    "vels": np.asarray(vels, dtype=np.float64),
                }
            )

        all_s = np.concatenate([b["s"] for b in branch_data])
        all_v = np.concatenate([b["vels"] for b in branch_data])
        all_re = np.concatenate(
            [np.asarray([self._metric_for_node(n, "reynolds") for n in b["path"]]) for b in branch_data]
        )
        all_fr = np.concatenate(
            [np.asarray([self._metric_for_node(n, "froude") for n in b["path"]]) for b in branch_data]
        )
        all_area = np.concatenate(
            [np.asarray([self._metric_for_node(n, "area_cm2") for n in b["path"]]) for b in branch_data]
        )
        all_dh = np.concatenate(
            [np.asarray([self._metric_for_node(n, "dh_mm") for n in b["path"]]) for b in branch_data]
        )

        x_max = float(np.nanmax(all_s)) * 1.05 if all_s.size else 1.0

        # ------------------------------------------------------------------
        # 2. Colormap for the top velocity profile.
        # ------------------------------------------------------------------
        if metric_values:
            vmax_m = float(np.nanpercentile(metric_values, 98.0))
            vmin_m = float(np.nanpercentile(metric_values, 2.0))
            if vmax_m <= vmin_m:
                vmax_m = vmin_m + 1e-6
        else:
            vmin_m, vmax_m = 0.0, 1.0
        norm = Normalize(vmin=vmin_m, vmax=vmax_m)
        cmap = {
            "velocity": matplotlib.colormaps["turbo"],
            "reynolds": matplotlib.colormaps["coolwarm"],
            "froude": matplotlib.colormaps["plasma"],
            "diameter": matplotlib.colormaps["viridis"],
        }.get(metric, matplotlib.colormaps["turbo"])

        def _branch_visible(idx: int) -> bool:
            return selected_branch < 0 or selected_branch == idx

        metric_label = {
            "velocity": "|v| (m/s)",
            "reynolds": "Re",
            "froude": "Fr",
            "diameter": "Dh (mm)",
        }.get(metric, metric)

        # ------------------------------------------------------------------
        # 3. Create axes and set common x-range.
        # ------------------------------------------------------------------
        gs = self.figure.add_gridspec(4, 1, height_ratios=[1.2, 1, 1, 1], hspace=0.18)
        ax_v = self.figure.add_subplot(gs[0, 0])
        ax_re = self.figure.add_subplot(gs[1, 0])
        ax_fr = self.figure.add_subplot(gs[2, 0])
        ax_area = self.figure.add_subplot(gs[3, 0])
        ax_dh = ax_area.twinx()

        for ax in (ax_v, ax_re, ax_fr, ax_area):
            self._style_ax(ax)
        ax_dh.tick_params(colors="#334155")
        ax_dh.yaxis.label.set_color("#334155")

        for ax in (ax_v, ax_re, ax_fr, ax_area):
            ax.set_xlim(0.0, x_max)
            ax.tick_params(labelbottom=(ax is ax_area))

        # ------------------------------------------------------------------
        # 4. Top: velocity profile coloured by the chosen metric.
        # ------------------------------------------------------------------
        n_branches = max(len(branch_data), 1)
        branch_cmap = matplotlib.colormaps["tab10"]

        def _branch_color(idx: int):
            return branch_cmap((idx % 10) / 9.0)

        annotated: set = set()
        for i, b in enumerate(branch_data):
            # Alternate label placement per branch to avoid overlap.
            label_dy = 10 if i % 2 == 0 else -15
            if not _branch_visible(i):
                continue
            s = b["s"]
            v = b["vels"]
            m = b["metrics"]
            color = _branch_color(i)
            label = self._branch_names[i]
            ax_v.plot(s, v, "-", color=color, linewidth=2.5, zorder=3, label=label)
            ax_v.scatter(
                s,
                v,
                c=m,
                cmap=cmap,
                norm=norm,
                s=60,
                zorder=4,
                edgecolors="#334155",
                linewidths=0.5,
            )
            if self._label_check.isChecked():
                for node, si, vi in zip(b["path"], s, v):
                    try:
                        up, down = node.name.split(" → ", 1)
                    except ValueError:
                        continue
                    label = down
                    if label in ("Parça", "PART", "part", "Kaynak", "SOURCE"):
                        continue
                    if label in annotated:
                        continue
                    annotated.add(label)
                    ax_v.annotate(
                        label,
                        (si, vi),
                        textcoords="offset points",
                        xytext=(0, label_dy),
                        ha="center",
                        fontsize=7,
                        color="#334155",
                        clip_on=True,
                    )

        if branch_data and selected_branch < 0:
            ax_v.legend(loc="upper right", fontsize=8, framealpha=0.9)

        if self._target_check.isChecked():
            tmin, tmax = self._target_v
            ax_v.axhspan(tmin, tmax, color="#22C55E", alpha=0.12, zorder=1)
            ax_v.axhline(tmin, color="#22C55E", linestyle="--", linewidth=1.0, zorder=2)
            ax_v.axhline(tmax, color="#22C55E", linestyle="--", linewidth=1.0, zorder=2)
            ax_v.text(
                0.02,
                0.98,
                f"Hedef: {tmin:.2f}-{tmax:.2f} m/s",
                transform=ax_v.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#15803D",
            )

        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = self.figure.colorbar(sm, ax=ax_v, pad=0.01)
        cbar.ax.set_ylabel(metric_label, color="#334155")
        cbar.ax.tick_params(colors="#334155")

        ax_v.set_ylabel("Hız (m/s)")
        ax_v.set_title("Akış Hızı Profili (renk: " + metric_label + ")")
        v_top = max(
            float(np.nanpercentile(all_v, 99.5)) * 1.2,
            self._target_v[1] * 1.2,
            0.01,
        )
        ax_v.set_ylim(0.0, v_top)

        # ------------------------------------------------------------------
        # 5. Re / Fr / Area / Dh panels.
        # ------------------------------------------------------------------
        for i, b in enumerate(branch_data):
            if not _branch_visible(i):
                continue
            s = b["s"]
            re = np.asarray([self._metric_for_node(n, "reynolds") for n in b["path"]])
            fr = np.asarray([self._metric_for_node(n, "froude") for n in b["path"]])
            area = np.asarray([self._metric_for_node(n, "area_cm2") for n in b["path"]])
            dh = np.asarray([self._metric_for_node(n, "dh_mm") for n in b["path"]])

            label = self._branch_names[i]
            ax_re.plot(s, re, "-", linewidth=2.0, label=label, zorder=3)
            ax_fr.plot(s, fr, "-", linewidth=2.0, label=label, zorder=3)
            ax_area.plot(s, area, "-", linewidth=2.0, label=f"A (cm²) - {label}", zorder=3)
            ax_dh.plot(s, dh, "--", linewidth=2.0, label=f"Dh (mm) - {label}", zorder=3)

        # Reynolds thresholds.
        re_top = max(float(np.nanmax(all_re)) * 1.15, 4500.0) if all_re.size else 4500.0
        re_log = re_top > 8000.0
        if re_log:
            ax_re.set_yscale("log")
            positive_re = all_re[all_re > 0.0]
            data_bottom = float(np.nanmin(positive_re)) * 0.5 if positive_re.size else 500.0
            # Always keep the Re=2000 / 4000 guidelines inside the y-range.
            re_bottom = min(data_bottom, 500.0)
        else:
            re_bottom = 0.0
        ax_re.axhline(2000.0, color="#16A34A", linestyle="-", linewidth=1.5, zorder=2, label="Re=2000")
        ax_re.axhline(4000.0, color="#DC2626", linestyle="--", linewidth=1.5, zorder=2, label="Re=4000")
        ax_re.fill_between([0.0, x_max], 2000.0, 4000.0, color="#FACC15", alpha=0.08, zorder=1)
        if re_top > 4000.0:
            ax_re.fill_between([0.0, x_max], 4000.0, re_top, color="#DC2626", alpha=0.08, zorder=1)
        ax_re.set_ylim(re_bottom, re_top)
        ax_re.set_ylabel("Reynolds (Re)")
        ax_re.set_title("Reynolds Sayısı")
        ax_re.legend(loc="upper right", fontsize=8)

        # Froude threshold.
        fr_top = max(float(np.nanmax(all_fr)) * 1.2, 1.5) if all_fr.size else 1.5
        ax_fr.axhline(1.0, color="#2563EB", linestyle="-", linewidth=2.0, zorder=2, label="Fr=1 (kritik)")
        if fr_top > 1.0:
            ax_fr.fill_between([0.0, x_max], 1.0, fr_top, color="#DC2626", alpha=0.08, zorder=1)
        ax_fr.set_ylim(0.0, fr_top)
        ax_fr.set_ylabel("Froude (Fr)")
        ax_fr.set_title("Froude Sayısı")
        ax_fr.legend(loc="upper right", fontsize=8)

        # Area / hydraulic diameter.
        a_top = max(float(np.nanpercentile(all_area, 99.5)) * 1.15, 0.01) if all_area.size else 0.01
        dh_top = max(float(np.nanpercentile(all_dh, 99.5)) * 1.15, 0.01) if all_dh.size else 0.01
        ax_area.set_ylim(0.0, a_top)
        ax_dh.set_ylim(0.0, dh_top)
        ax_area.set_ylabel("Kesit Alanı A (cm²)", color="#334155")
        ax_area.set_title("Kesit Alanı ve Hidrolik Çap")
        ax_area.tick_params(axis="y", colors="#334155")
        ax_dh.set_ylabel("Hidrolik Çap Dh (mm)", color="#7C3AED")
        ax_dh.tick_params(axis="y", colors="#7C3AED")
        lines, labels = ax_area.get_legend_handles_labels()
        lines2, labels2 = ax_dh.get_legend_handles_labels()
        ax_area.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8)

        ax_area.set_xlabel("Kaynaktan uzaklık (mm)")

        self.canvas.draw()
