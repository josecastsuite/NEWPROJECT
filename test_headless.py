"""Headless test of core analysis on Knuckle.STEP."""
import os
import sys
import time

# Avoid importing PyQt / pyvistaqt / weasyprint.
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
from core.step_loader import load_step
from core.voxelizer import build_voxel_grid, apply_unit_scale
from core.sdf_analyzer import analyze
from core.gating import analyze_gating
from core.materials import get_alloy, get_mold, chvorinov_c_from_properties
from core.types import CastingParameters, BodyType
from core.reporter import _generate_html


def main(path: str = "data/Knuckle.STEP", out_dir: str = "/tmp/josecast_test"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading {path} ...")
    bodies = load_step(path)
    print(f"Loaded {len(bodies)} bodies")
    for i, b in enumerate(bodies):
        print(f"  Body {i}: {b.name}, vol={b.volume_cm3:.2f} cm3, bbox={b.mesh.bounds}")

    # Try to auto-detect unit
    max_size = max((b.mesh.bounds[1] - b.mesh.bounds[0]).max() for b in bodies)
    print(f"Max bbox size raw = {max_size:.3f} units")

    # Run analysis with mm assumption
    apply_unit_scale(bodies, "mm")

    target_dim = 160
    print(f"Voxelizing at {target_dim} ...")
    grid, origin, dx, bodies = build_voxel_grid(bodies, target_dim=target_dim)
    print(f"  grid shape={grid.shape}, dx={dx:.4f} mm, origin={origin}")

    alloy_key = "42CrMo4"
    mold_key = "sand"
    params = CastingParameters(
        t_pour_c=1600.0,
        t_liquidus_c=1510.0,
        t_solidus_c=1410.0,
        t_mold_c=25.0,
        t_fill_s=0.0,
        rho_liquid_kg_m3=7850.0,
        viscosity_pa_s=0.005,
    )

    alloy = get_alloy(alloy_key)
    mold = get_mold(mold_key)
    print("Chvorinov C from formula:", chvorinov_c_from_properties(alloy, mold))
    print("Alloy/Mold diffusivities:", alloy.diffusivity_mm2_s, mold.diffusivity_mm2_s)

    print("Running analyze (this may take a while) ...")
    t0 = time.time()
    result = analyze(
        bodies,
        grid,
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
    print(f"  analyze done in {time.time()-t0:.1f}s")
    print(f"  hotspots: {len(result.hotspots)}")
    for i, hs in enumerate(result.hotspots):
        print(f"    HS{i}: pos={hs.position_mm}, M={hs.m_value_mm:.2f}, "
              f"t={hs.t_section_mm:.2f}, dist={hs.dist_to_riser_mm:.1f}, "
              f"FD={hs.max_feeding_distance_mm:.1f}, N={hs.niyama_ensemble:.2f}, "
              f"feed_ok={hs.feed_ok}, dir_ok={hs.directional_ok}, heuver_ok={hs.heuvers_ok}, darcy_ok={hs.darcy_ok}")

    print("Running gating ...")
    gate = analyze_gating(result, casting_params=params, bodies=bodies)
    print(f"  gate: {gate}")
    if gate:
        print(f"    fluidity_length_mm={gate.fluidity_length_mm:.1f}")
        print(f"    ingate_velocity_m_s={gate.ingate_velocity_m_s:.2f}")
        print(f"    total_poured_mass_kg={gate.total_poured_mass_kg:.3f}")
        print(f"    pour_yield={gate.pouring_yield:.3f}")
        for k, sf in gate.section_flows.items():
            print(f"    {k}: v={sf.velocity_m_s:.2f}, Re={sf.reynolds:.0f}, Fr={sf.froude:.2f}, A={sf.area_cm2:.2f}")

    print("Generating HTML report ...")
    html_path = os.path.join(out_dir, "test_report.html")
    _generate_html(result, html_path)
    print(f"  saved to {html_path}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/Knuckle.STEP"
    main(path)
