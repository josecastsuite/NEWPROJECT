"""Quick single-model, single-mold air-entrapment check."""
import os
import time

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import numpy as np

from core.step_loader import load_step
from core.voxelizer import build_voxel_grid, apply_unit_scale
from core.sdf_analyzer import analyze
from core.materials import get_alloy, get_mold
from core.types import CastingParameters


def main(mold_key: str = "metal_mold"):
    model = "data/Model_Knuckle_Dusuk.STEP"
    bodies = load_step(model)
    apply_unit_scale(bodies, 'mm')
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
        ingate_velocity_m_s=0.0,
        velocity_section_key='SPRUE_THROAT',
    )
    t0 = time.time()
    res = analyze(
        bodies, grid, body_index, origin, dx,
        alloy_key='42CrMo4',
        mold_key=mold_key,
        base_res=120,
        max_res=300,
        refine_local=False,
        sub_voxel=1,
        thermal_max_time_s=120,
        thermal_downsample=2,
        casting_params=params,
    )
    print(f"\n[mold={mold_key}] elapsed={time.time()-t0:.1f}s")
    ae = res.air_entrapment
    if ae is not None and ae.size:
        print(f"  air_entrapment.max()={float(np.max(ae))}")
        print(f"  air_entrapment.mean()={float(np.mean(ae))}")
        print(f"  cells > 0.3={int(np.sum(ae > 0.3))}")
        print(f"  trapped_air_volume_m3={res.trapped_air_volume_m3}")
    else:
        print("  no air_entrapment data")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "metal_mold")
