"""Compare D3Q7 trapped-air volume for Deneme_Ring across sand/ceramic/metal moulds."""
import json
import os
import sys
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from core.materials import get_alloy, get_mold
from core.sdf_analyzer import analyze
from core.step_loader import load_step
from core.types import CastingParameters
from core.voxelizer import apply_unit_scale, build_voxel_grid


def run_one(mold_key: str):
    step_path = repo_root / "data" / "Deneme_Ring.STEP"
    bodies = load_step(str(step_path))
    apply_unit_scale(bodies, "mm")
    gravity = (0.0, 0.0, -1.0)
    grid, body_index, origin, dx, bodies = build_voxel_grid(
        bodies, target_dim=120, gravity_vector=gravity
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
        ingate_velocity_m_s=1.5,
        velocity_section_key="SPRUE_THROAT",
    )
    t0 = time.time()
    res = analyze(
        bodies,
        grid,
        body_index,
        origin,
        dx,
        alloy_key="42CrMo4",
        mold_key=mold_key,
        base_res=120,
        max_res=240,
        refine_local=False,
        sub_voxel=1,
        thermal_max_time_s=120,
        thermal_downsample=2,
        casting_params=params,
    )
    elapsed = time.time() - t0
    ae = res.air_entrapment
    if ae is not None and ae.size:
        return {
            "mold": mold_key,
            "elapsed_s": elapsed,
            "max": float(np.max(ae)),
            "mean": float(np.mean(ae)),
            "trapped_air_volume_m3": float(res.trapped_air_volume_m3),
            "cells_gt_0": int(np.sum(ae > 0)),
        }
    return {"mold": mold_key, "elapsed_s": elapsed, "error": "no air entrapment data"}


def main():
    os.environ.setdefault("JOSECAST_CPP_LBM_MAX_STEPS", "2000")
    os.environ.setdefault("JOSECAST_CPP_LBM_CALLBACK_EVERY_N", "50")
    results = {}
    for mold_key in ["sand", "ceramic", "metal_mold"]:
        print(f"\n=== {mold_key} ===", flush=True)
        results[mold_key] = run_one(mold_key)
        for k, v in results[mold_key].items():
            print(f"  {k}: {v}", flush=True)

    out_path = Path(__file__).parent / "results" / "darcy_separation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")

    # quick separation check
    volumes = {k: v.get("trapped_air_volume_m3", 0.0) for k, v in results.items()}
    if volumes.get("sand", 0.0) > 0:
        metal_vs_sand = volumes.get("metal_mold", 0.0) / max(volumes["sand"], 1e-18)
        print(f"metal/sand trapped volume ratio = {metal_vs_sand:.2f}x")


if __name__ == "__main__":
    main()
