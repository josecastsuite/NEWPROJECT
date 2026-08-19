"""Debug: save ft/part_mask and inspect saddle candidates for Deneme_Ring."""
import os
import sys
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
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


def main():
    step_candidates = [
        repo_root / "data" / "Deneme_Ring.STEP",
        Path("/home/ubuntu/attachments/eaa84a66-adda-41ec-aa2d-87f2182243e4/Deneme_Ring.STEP"),
    ]
    step_path = next((p for p in step_candidates if p.exists()), None)
    if step_path is None:
        raise FileNotFoundError("Deneme_Ring.STEP not found")

    bodies = load_step(str(step_path))
    apply_unit_scale(bodies, "mm")

    grid, body_index, origin, dx, bodies = build_voxel_grid(
        bodies, target_dim=60, gravity_vector=(0.0, 0.0, -1.0)
    )

    alloy = get_alloy("AlSi7")
    mold = get_mold("sand")
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
        alloy_key="AlSi7",
        mold_key="sand",
        base_res=60,
        max_res=120,
        refine_local=False,
        sub_voxel=1,
        thermal_max_time_s=300,
        thermal_downsample=2,
        casting_params=params,
        part_voxels_target=0,
    )
    print("elapsed", time.time() - t0)
    print("grid shape", grid.shape, "dx", dx)
    print("saddle count", result.cold_shot_saddles.get("count", 0))
    np.savez(
        repo_root / "validation" / "results" / "phase_d_debug.npz",
        ft=getattr(result, "fill_time_s", np.array([])),
        part_mask=result.is_metal == 1,
        grid=grid,
        origin=origin,
        dx=dx,
        H_field=result.H_field,
    )
    print("saved debug.npz")


if __name__ == "__main__":
    main()
