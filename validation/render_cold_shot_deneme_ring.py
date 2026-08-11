"""Render Deneme_Ring cold-shot/lap risk screenshots for Faz E UI validation."""
import os
import sys
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
from PyQt6.QtWidgets import QApplication

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from ui.viewer import Analyzer3DViewer
from validation.test_v8_phase_d import run_case


def render_case(mold_key: str, alloy_key: str, out_dir: Path):
    report, result = run_case(mold_key, alloy_key)
    print(f"[{mold_key}] saddle_count={report['saddle_count']} cs_mean={report['cold_shot_risk_mean']:.3f}")

    app = QApplication.instance() or QApplication(sys.argv)
    viewer = Analyzer3DViewer(off_screen=True)
    viewer.ren_win.SetSize(1280, 960)

    # Cold-shot risk (inferno)
    viewer.show_cold_shot_risk(result)
    viewer.camera_position = "xz"
    viewer.reset_camera()
    cs_png = out_dir / f"Deneme_Ring_{mold_key}_cold_shot.png"
    viewer.screenshot(str(cs_png))
    print(f"wrote {cs_png}")

    # Lap risk (viridis) – toggle cold-shot off first
    viewer.toggle_cold_shot_risk(result, False)
    viewer.show_lap_risk(result)
    viewer.camera_position = "xz"
    viewer.reset_camera()
    lap_png = out_dir / f"Deneme_Ring_{mold_key}_lap.png"
    viewer.screenshot(str(lap_png))
    print(f"wrote {lap_png}")


def main():
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    for mold_key in ["sand", "metal_mold"]:
        render_case(mold_key, "AlSi7", out_dir)


if __name__ == "__main__":
    main()
