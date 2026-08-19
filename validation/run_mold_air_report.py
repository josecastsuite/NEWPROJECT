"""Compare air-entrapment behavior across mold types for a fixed STEP model."""
import os
import time
import json
from pathlib import Path

import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from core.step_loader import load_step
from core.voxelizer import build_voxel_grid, apply_unit_scale
from core.sdf_analyzer import analyze
from core.types import CastingParameters
from core.materials import MOLDS, make_effective_mold, get_mold, get_alloy


def main():
    repo_root = Path(__file__).parent.parent
    step_path = repo_root / "data" / "Deneme_Ring.STEP"

    bodies = load_step(str(step_path))
    apply_unit_scale(bodies, "mm")
    grid, body_index, origin, dx, bodies = build_voxel_grid(
        bodies, target_dim=80, gravity_vector=(0.0, 0.0, -1.0)
    )

    alloy_key = "42CrMo4"
    alloy = get_alloy(alloy_key)
    params = CastingParameters(
        t_pour_c=alloy.t_pour_c,
        t_liquidus_c=alloy.t_liquidus_c,
        t_solidus_c=alloy.t_solidus_c,
        t_mold_c=25.0,
        t_fill_s=0.0,
        rho_liquid_kg_m3=alloy.rho_kg_m3,
        viscosity_pa_s=alloy.viscosity_pa_s,
        gravity_direction=(0.0, 0.0, -1.0),
        ingate_velocity_m_s=1.5,
        velocity_section_key="SPRUE_THROAT",
    )

    # Real mould materials only (body presets such as chills/sleeves/filters
    # are not moulds and have been moved to body_presets.json).
    selected = [
        "green_sand",
        "silica_sand",
        "zircon_sand",
        "chromite_sand",
        "furan_resin_sand",
        "sodium_silicate_sand",
        "shell_mold",
        "ceramic",
        "investment_ceramic",
        "metal_mold",
        "graphite_mold",
    ]

    results = []
    for mold_key in selected:
        print(f"\n=== {mold_key} ===")
        t0 = time.time()
        try:
            result = analyze(
                bodies,
                grid,
                body_index,
                origin,
                dx,
                alloy_key=alloy_key,
                mold_key=mold_key,
                base_res=80,
                max_res=160,
                refine_local=False,
                sub_voxel=2,
                thermal_max_time_s=0,
                thermal_downsample=2,
                casting_params=params,
            )
            air = result.air_entrapment
            air_max = float(air.max()) if air is not None and air.size else 0.0
            air_mean = float(air.mean()) if air is not None and air.size else 0.0
            cells_gt0 = int(np.sum(air > 0)) if air is not None and air.size else 0
            vol = float(result.trapped_air_volume_m3) if result.trapped_air_volume_m3 is not None else 0.0
            elapsed = time.time() - t0
            mold = make_effective_mold(get_mold(mold_key), casting_params=params)
            results.append({
                "mold_key": mold_key,
                "mold_name": mold.name,
                "mold_type": mold.mold_type,
                "is_sand": bool(mold.is_sand),
                "d50_mm": float(mold.particle_size_mm),
                "afs_grain_size": float(mold.afs_grain_size),
                "permeability_proxy": float(mold.permeability_proxy),
                "air_max": air_max,
                "air_mean": air_mean,
                "cells_gt0": cells_gt0,
                "trapped_air_volume_m3": vol,
                "elapsed_s": elapsed,
            })
            print(f"max={air_max:.3f} mean={air_mean:.4f} cells={cells_gt0} vol={vol:.3e} time={elapsed:.1f}s")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                "mold_key": mold_key,
                "mold_name": getattr(MOLDS.get(mold_key), "name", mold_key),
                "error": str(e),
            })

    out_path = repo_root / "validation" / "mold_air_entrapment_report.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
