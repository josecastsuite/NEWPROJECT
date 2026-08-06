"""Quick smoke test: run GeoFC air entrapment on real STEP models with fast_flow.

This bypasses the expensive C++ LBM solve and exercises the new geometric
front-collision detector end-to-end.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.step_loader import load_step
from core.voxelizer import build_voxel_grid, apply_unit_scale
from core.sdf_analyzer import analyze
from core.materials import get_alloy, get_mold
from core.types import CastingParameters

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _run_one(step_path: Path) -> dict:
    t0 = time.time()
    gravity = (0.0, -1.0, 0.0) if "parca" in step_path.name.lower() else (0.0, 0.0, -1.0)

    bodies = load_step(str(step_path))
    apply_unit_scale(bodies, "mm")

    grid, body_index, origin, dx, bodies = build_voxel_grid(
        bodies, target_dim=160, gravity_vector=gravity
    )

    params = CastingParameters(
        t_pour_c=1600.0,
        t_liquidus_c=1510.0,
        t_solidus_c=1410.0,
        t_mold_c=25.0,
        t_fill_s=0.0,
        rho_liquid_kg_m3=7850.0,
        viscosity_pa_s=0.005,
        gravity_direction=gravity,
        ingate_velocity_m_s=0.0,
        velocity_section_key="SPRUE_THROAT",
        fast_flow=True,
    )

    result = analyze(
        bodies,
        grid,
        body_index,
        origin,
        dx,
        alloy_key="42CrMo4",
        mold_key="sand",
        base_res=160,
        max_res=600,
        refine_local=False,
        sub_voxel=2,
        thermal_max_time_s=0.0,
        thermal_downsample=2,
        casting_params=params,
    )

    air = result.air_entrapment
    info = {
        "model": step_path.name,
        "grid_shape": list(grid.shape),
        "dx_mm": float(dx),
        "elapsed_s": time.time() - t0,
        "air_max": float(air.max()) if air is not None and air.size else None,
        "air_min": float(air.min()) if air is not None and air.size else None,
        "trapped_air_volume_m3": float(result.trapped_air_volume_m3) if result is not None else None,
    }
    if info["air_max"] is not None and info["air_max"] > 0.0:
        print(f"  [geofc] {step_path.name}: air_max={info['air_max']:.4f} vol={info['trapped_air_volume_m3']:.3e}")
    else:
        print(f"  [geofc] {step_path.name}: no air entrapment detected")
    return info


def main():
    models = sorted(DATA_DIR.glob("*.STEP"))
    rows = []
    for mp in models:
        print(f"[geofc] {mp.name} ...", flush=True)
        try:
            rows.append(_run_one(mp))
        except Exception as exc:
            print(f"  ERROR: {exc}")
            rows.append({"model": mp.name, "error": str(exc)})
    out = OUT_DIR / "geofc_check.json"
    out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"[geofc] report: {out}")


if __name__ == "__main__":
    main()
