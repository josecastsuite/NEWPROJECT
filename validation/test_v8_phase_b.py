"""Faz B validation: H_field eutectic-plateau on Deneme_Ring.STEP with AlSi12."""
import json
import os
import sys
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
# Use the Darcy/volume-layer fill-time estimator so we can target a slow fill
# and reach the eutectic plateau without the C++ LBM overhead.
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


def main() -> None:
    step_candidates = [
        repo_root / "data" / "Deneme_Ring.STEP",
        Path("/home/ubuntu/attachments/eaa84a66-adda-41ec-aa2d-87f2182243e4/Deneme_Ring.STEP"),
        Path("/home/ubuntu/NEWPROJECT_stage/NEWPROJECT-main/data/Deneme_Ring.STEP"),
        Path("/home/ubuntu/NEWPROJECT-X64-84/data/Deneme_Ring.STEP"),
    ]
    step_path = next((p for p in step_candidates if p.exists()), None)
    if step_path is None:
        raise FileNotFoundError("Deneme_Ring.STEP not found")

    t0 = time.time()
    bodies = load_step(str(step_path))
    apply_unit_scale(bodies, "mm")

    # Keep the grid coarse enough for a quick Faz B validation run.
    grid, body_index, origin, dx, bodies = build_voxel_grid(
        bodies, target_dim=80, gravity_vector=(0.0, 0.0, -1.0)
    )

    alloy = get_alloy("AlSi12")
    mold = get_mold("sand")
    # Slow fill (t_fill_s=25 s, no prescribed ingate velocity) so the last
    # metal to arrive has cooled through the eutectic plateau.
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

    result = analyze(
        bodies,
        grid,
        body_index,
        origin,
        dx,
        alloy_key="AlSi12",
        mold_key="sand",
        base_res=80,
        max_res=160,
        refine_local=False,
        sub_voxel=1,
        thermal_max_time_s=300,
        thermal_downsample=2,
        casting_params=params,
    )

    elapsed = time.time() - t0
    print(f"[test_v8_phase_b] completed in {elapsed:.1f} s")

    part_mask = result.is_metal == 1
    T = result.T_meet[part_mask]
    H = result.H_field[part_mask]
    fs = result.fs_meet[part_mask]
    ft = (
        result.flow_result.fill_time[part_mask]
        if result.flow_result is not None and result.flow_result.fill_time is not None
        else np.zeros(part_mask.sum())
    )

    Te = alloy.t_eutectic_or_solidus_c
    plateau = part_mask & (np.abs(result.T_meet - Te) < 1.0)

    report = {
        "model": step_path.name,
        "alloy": "AlSi12",
        "mold": "sand",
        "elapsed_s": elapsed,
        "part_voxels": int(part_mask.sum()),
        "plateau_voxels": int(plateau.sum()),
        "T_min": float(T.min()) if T.size else 0.0,
        "T_max": float(T.max()) if T.size else 0.0,
        "H_min": float(H.min()) if H.size else 0.0,
        "H_max": float(H.max()) if H.size else 0.0,
        "fs_min": float(fs.min()) if fs.size else 0.0,
        "fs_max": float(fs.max()) if fs.size else 0.0,
        "fill_time_min": float(ft.min()) if ft.size else 0.0,
        "fill_time_max": float(ft.max()) if ft.size else 0.0,
    }
    if plateau.sum() > 1:
        H_plat = result.H_field[plateau]
        T_plat = result.T_meet[plateau]
        fs_plat = result.fs_meet[plateau]
        report["plateau_T_mean"] = float(T_plat.mean())
        report["plateau_T_std"] = float(T_plat.std())
        report["plateau_H_range"] = float(H_plat.max() - H_plat.min())
        report["plateau_H_drop_OK"] = (H_plat.max() - H_plat.min()) > 1e3
        report["plateau_fs_min"] = float(fs_plat.min())
        report["plateau_fs_max"] = float(fs_plat.max())

    out_path = Path(__file__).parent / "results" / "v8_phase_b_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[test_v8_phase_b] wrote {out_path}")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
