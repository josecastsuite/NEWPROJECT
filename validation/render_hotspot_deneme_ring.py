"""Render Deneme_Ring hotspot screenshot with corrected M labels."""
import os
import sys
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QApplication

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from ui.viewer import Analyzer3DViewer
from validation.test_v8_phase_d import run_case


def main():
    os.environ["JOSECAST_USE_CPP_LBM"] = "0"
    os.environ["JOSECAST_USE_CPP_VOF"] = "0"
    os.environ["JOSECAST_USE_CPP_DARCY"] = "1"
    os.environ["QT_QPA_PLATFORM"] = "xcb"

    report, result, bodies = run_case("metal_mold", "AlSi7")
    print(
        f"[hotspot] hotspots={len(result.hotspots)} "
        f"cs_mean={report['cold_shot_risk_mean']:.3f} "
        f"m_geo_cm={result.geometric_m_cm:.2f} "
        f"dominant_m_cm={result.dominant_m_mm/10.0:.2f}"
    )

    app = QApplication.instance() or QApplication(sys.argv)
    viewer = Analyzer3DViewer(off_screen=True)
    viewer.ren_win.SetSize(1280, 960)

    viewer.show_bodies(bodies, reset_camera=False, analysis_mode=True)
    viewer.show_hotspots(result)
    viewer.camera_position = "xy"
    viewer.reset_camera()

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "Deneme_Ring_metal_mold_AlSi7_hotspots_fixed.png"
    viewer.screenshot(str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
