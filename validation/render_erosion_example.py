import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import numpy as np

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from PyQt6.QtWidgets import QApplication

from core.materials import get_alloy, get_mold
from core.sdf_analyzer import analyze
from core.step_loader import load_step
from core.types import CastingParameters
from core.voxelizer import apply_unit_scale, build_voxel_grid
from ui.viewer import Analyzer3DViewer

t0 = time.time()


def run_case(mold_key: str, alloy_key: str, step_path: Path):
    bodies = load_step(str(step_path))
    apply_unit_scale(bodies, "mm")
    grid, body_index, origin, dx, bodies = build_voxel_grid(
        bodies, target_dim=60, gravity_vector=(0.0, 0.0, -1.0), max_dim=120
    )

    params = CastingParameters()
    result = analyze(
        bodies=bodies,
        grid=grid,
        body_index=body_index,
        origin_mm=origin,
        dx=dx,
        alloy_key=alloy_key,
        mold_key=mold_key,
        casting_params=params,
    )
    er = result.erosion_risk
    if er is not None and er.size > 0:
        nonzero = er[er > 0.01]
        print(
            f"[{alloy_key}/{mold_key}] elapsed {(time.time()-t0):.1f}s\n"
            f"  erosion max={er.max():.3f} "
            f"mean(nonzero)={(nonzero.mean() if nonzero.size else 0):.3f} "
            f"cells>0.01={(er > 0.01).sum()}"
        )
    return result, bodies


def render(result, bodies, filename):
    app = QApplication.instance() or QApplication(sys.argv)
    viewer = Analyzer3DViewer(off_screen=True)
    viewer.ren_win.SetSize(1280, 960)
    viewer.show_bodies(bodies, reset_camera=True, analysis_mode=True)
    viewer.show_erosion_risk(result)
    out = repo_root / "validation" / "results" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    viewer.screenshot(str(out))
    print(f"wrote {out}")
    QApplication.quit()


def main():
    global t0
    step_path = repo_root / "data" / "Deneme_Ring.STEP"
    t0 = time.time()
    sand, sand_bodies = run_case("sand", "AlSi7", step_path)
    t0 = time.time()
    metal, metal_bodies = run_case("metal_mold", "AlSi7", step_path)

    render(sand, sand_bodies, "Deneme_Ring_sand_AlSi7_erosion.png")
    render(metal, metal_bodies, "Deneme_Ring_metal_mold_AlSi7_erosion.png")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
