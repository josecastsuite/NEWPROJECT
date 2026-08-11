"""Faz D validation: Reeb/watershed saddles + SFER on Deneme_Ring.STEP."""
import json
import os
import sys
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
# Darcy/volume-layer fill-time estimator is much faster than C++ LBM.
os.environ["JOSECAST_USE_CPP_LBM"] = "0"
os.environ["JOSECAST_USE_CPP_VOF"] = "0"

import numpy as np

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from core.materials import get_alloy, get_mold
from core.sdf_analyzer import analyze
from core.step_loader import load_step
from core.types import CastingParameters
from core.voxelizer import apply_unit_scale, build_voxel_grid


def run_case(mold_key: str, alloy_key: str):
    step_candidates = [
        repo_root / "data" / "Deneme_Ring.STEP",
        Path("/home/ubuntu/attachments/eaa84a66-adda-41ec-aa2d-87f2182243e4/Deneme_Ring.STEP"),
        Path("/home/ubuntu/NEWPROJECT_stage/NEWPROJECT-main/data/Deneme_Ring.STEP"),
        Path("/home/ubuntu/NEWPROJECT-X64-84/data/Deneme_Ring.STEP"),
    ]
    step_path = next((p for p in step_candidates if p.exists()), None)
    if step_path is None:
        raise FileNotFoundError("Deneme_Ring.STEP not found")

    bodies = load_step(str(step_path))
    apply_unit_scale(bodies, "mm")

    grid, body_index, origin, dx, bodies = build_voxel_grid(
        bodies, target_dim=60, gravity_vector=(0.0, 0.0, -1.0), max_dim=120
    )

    alloy = get_alloy(alloy_key)
    mold = get_mold(mold_key)
    params = CastingParameters(
        t_pour_c=700.0,
        t_liquidus_c=alloy.t_liquidus_c,
        t_solidus_c=alloy.t_solidus_c,
        t_mold_c=25.0,
        t_fill_s=25.0,
        rho_liquid_kg_m3=alloy.rho_kg_m3,
        viscosity_pa_s=0.0012,
        gravity_direction=(0.0, 0.0, -1.0),
        ingate_velocity_m_s=0.0,
        velocity_section_key="SPRUE_THROAT",
    )

    t0 = time.time()
    result = analyze(
        bodies,
        grid,
        body_index,
        origin,
        dx,
        alloy_key=alloy_key,
        mold_key=mold_key,
        base_res=60,
        max_res=120,
        part_voxels_target=0,
        refine_local=False,
        sub_voxel=1,
        thermal_max_time_s=300,
        thermal_downsample=2,
        casting_params=params,
    )
    elapsed = time.time() - t0

    part_mask = result.is_metal == 1
    cs = result.cold_shot_risk[part_mask]
    lap = result.lap_risk[part_mask]
    saddles = result.cold_shot_saddles
    saddle_count = saddles.get("count", 0) if isinstance(saddles, dict) else 0

    report = {
        "model": step_path.name,
        "alloy": alloy_key,
        "mold": mold_key,
        "elapsed_s": elapsed,
        "part_voxels": int(part_mask.sum()),
        "saddle_count": int(saddle_count),
        "cold_shot_risk_min": float(cs.min()) if cs.size else 0.0,
        "cold_shot_risk_max": float(cs.max()) if cs.size else 0.0,
        "cold_shot_risk_mean": float(cs.mean()) if cs.size else 0.0,
        "lap_risk_max": float(lap.max()) if lap.size else 0.0,
        "lap_risk_mean": float(lap.mean()) if lap.size else 0.0,
    }
    return report


def main() -> None:
    alloy_key = "AlSi7"
    reports = {}
    for mold_key in ["sand", "metal_mold"]:
        print(f"[test_v8_phase_d] running {alloy_key} + {mold_key} ...")
        reports[mold_key] = run_case(mold_key, alloy_key)

    sand = reports["sand"]
    metal = reports["metal_mold"]
    reports["cold_shot_ratio_metal_sand"] = (
        metal["cold_shot_risk_mean"] / sand["cold_shot_risk_mean"]
        if sand["cold_shot_risk_mean"] > 1e-12
        else 0.0
    )
    reports["saddle_count_OK"] = 10 <= metal["saddle_count"] <= 1000

    out_path = Path(__file__).parent / "results" / "v8_phase_d_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
    print(f"[test_v8_phase_d] wrote {out_path}")
    print(json.dumps(reports, indent=2, default=str))


if __name__ == "__main__":
    main()
