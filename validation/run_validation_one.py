"""Run the full analysis pipeline on a single STEP file and report air entrapment."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from core.step_loader import load_step
from core.voxelizer import build_voxel_grid, apply_unit_scale
from core.sdf_analyzer import analyze
from core.materials import get_alloy, get_mold
from core.types import CastingParameters


def main(step_name: str = "Deneme_Ring.STEP") -> None:
    repo_root = Path(__file__).parent.parent
    step_path = repo_root / "data" / step_name
    if not step_path.exists():
        raise FileNotFoundError(f"STEP not found: {step_path}")

    t0 = time.time()
    gravity = (0.0, -1.0, 0.0) if "parca" in step_path.name.lower() else (0.0, 0.0, -1.0)

    bodies = load_step(str(step_path))
    apply_unit_scale(bodies, "mm")

    grid, body_index, origin, dx, bodies = build_voxel_grid(
        bodies, target_dim=160, gravity_vector=gravity
    )

    alloy_key, mold_key = "42CrMo4", "sand"
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

    result = analyze(
        bodies,
        grid,
        body_index,
        origin,
        dx,
        alloy_key=alloy_key,
        mold_key=mold_key,
        base_res=160,
        max_res=600,
        refine_local=False,
        sub_voxel=2,
        thermal_max_time_s=300,
        thermal_downsample=2,
        casting_params=params,
    )

    elapsed = time.time() - t0
    print(f"\n[run_validation_one] completed in {elapsed:.1f} s\n")

    fr = result.flow_result
    print("flow_result:", fr is not None)
    if fr is not None:
        print("  fill_time_s:", fr.fill_time_s)
        print("  Q_m3_s:", fr.Q_m3_s)
        print("  air_entrapment present:", fr.air_entrapment is not None)
        if fr.air_entrapment is not None and fr.air_entrapment.size:
            print("  air_entrapment.shape:", fr.air_entrapment.shape)
            print("  air_entrapment.max():", float(np.max(fr.air_entrapment)))
            print("  air_entrapment.mean():", float(np.mean(fr.air_entrapment)))
            print("  air_entrapment cells > 0:", int(np.sum(fr.air_entrapment > 0)))
        else:
            print("  air_entrapment is None or empty")
        print("  trapped_air_volume_m3 (flow_result):", fr.trapped_air_volume_m3)
        print("  reason:", fr.reason[:200] if fr.reason else "")

    print("\nresult.trapped_air_volume_m3:", result.trapped_air_volume_m3)
    print("result.air_entrapment.max():", float(np.max(result.air_entrapment)) if result.air_entrapment is not None and result.air_entrapment.size else 0.0)
    print("result.air_entrapment.mean():", float(np.mean(result.air_entrapment)) if result.air_entrapment is not None and result.air_entrapment.size else 0.0)
    print("result.air_entrapment cells > 0:", int(np.sum(result.air_entrapment > 0)) if result.air_entrapment is not None and result.air_entrapment.size else 0)

    out = {
        "model": step_path.name,
        "elapsed_s": elapsed,
        "flow_result_present": fr is not None,
        "air_entrapment_present": fr.air_entrapment is not None and fr.air_entrapment.size > 0 if fr else False,
        "trapped_air_volume_m3": float(result.trapped_air_volume_m3) if result.trapped_air_volume_m3 is not None else 0.0,
        "air_max": float(np.max(result.air_entrapment)) if result.air_entrapment is not None and result.air_entrapment.size else 0.0,
        "air_mean": float(np.mean(result.air_entrapment)) if result.air_entrapment is not None and result.air_entrapment.size else 0.0,
    }
    out_path = Path(__file__).parent / "results" / f"{step_name}_air_entrapment.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n[run_validation_one] wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Deneme_Ring.STEP")
