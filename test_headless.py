"""Headless test of core analysis on a STEP file."""
import argparse
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


def _parse_gravity(text: str):
    if text is None:
        return None
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 3:
        raise ValueError("Yerçekimi vektörü x,y,z formatında 3 sayı olmalı.")
    v = np.array(parts, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm == 0:
        raise ValueError("Yerçekimi vektörü sıfır olamaz.")
    return tuple((v / norm).tolist())


def _parse_section_areas(text: str):
    """Parse --section-area KEY=VALUE,... pairs, e.g. RUNNER=9.85,INGATE=4.89."""
    out = {}
    if not text:
        return out
    for item in text.split(","):
        key, value = item.split("=")
        out[key.strip().upper()] = float(value.strip())
    return out


def main():
    parser = argparse.ArgumentParser(description="JoseCast headless test")
    parser.add_argument("path", nargs="?", default="data/Knuckle.STEP", help="STEP file")
    parser.add_argument("--out-dir", default="/tmp/josecast_test", help="HTML output directory")
    parser.add_argument("--gravity", default=None, help="Yerçekimi vektörü x,y,z (örn: 0,-1,0)")
    parser.add_argument("--section-area", default=None, help="Kullanıcı kesit alanları: KEY=cm2,... (RUNNER,INGATE,SPRUE_BASE,SPRUE_THROAT)")
    parser.add_argument("--velocity", default=None, type=float, help="Hedef kesit hızı v (m/s)")
    parser.add_argument("--velocity-section", default="INGATE", help="Hedef hız kesiti (INGATE/RUNNER/SPRUE_BASE/SPRUE_THROAT)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Loading {args.path} ...")
    bodies = load_step(args.path)
    print(f"Loaded {len(bodies)} bodies")
    for i, b in enumerate(bodies):
        print(f"  Body {i}: {b.name}, vol={b.volume_cm3:.2f} cm3, bbox={b.mesh.bounds}")

    max_size = max((b.mesh.bounds[1] - b.mesh.bounds[0]).max() for b in bodies)
    print(f"Max bbox size raw = {max_size:.3f} units")

    apply_unit_scale(bodies, "mm")

    if args.gravity:
        gravity = _parse_gravity(args.gravity)
    else:
        # Parça1 model has the riser above the part in +Y and gating in +X;
        # Knuckle uses the conventional -Z gravity.
        gravity = (0.0, -1.0, 0.0) if "parca" in args.path.lower() else (0.0, 0.0, -1.0)
    print(f"Using gravity vector: {gravity}")

    user_section_areas = _parse_section_areas(args.section_area)
    if user_section_areas:
        print(f"User section areas (cm2): {user_section_areas}")

    target_dim = 160
    print(f"Voxelizing at {target_dim} ...")
    grid, origin, dx, bodies = build_voxel_grid(
        bodies, target_dim=target_dim, gravity_vector=gravity
    )
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
        gravity_vector=gravity,
        ingate_velocity_m_s=float(args.velocity) if args.velocity else 0.0,
        velocity_section_key=args.velocity_section,
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
    gate = analyze_gating(
        result,
        casting_params=params,
        bodies=bodies,
        user_section_areas_cm2=user_section_areas,
    )
    print(f"  gate: {gate}")
    if gate:
        print(f"    fluidity_length_mm={gate.fluidity_length_mm:.1f}")
        print(f"    ingate_velocity_m_s={gate.ingate_velocity_m_s:.2f}")
        print(f"    total_poured_mass_kg={gate.total_poured_mass_kg:.3f}")
        print(f"    pour_yield={gate.pouring_yield:.3f}")
        for k, sf in gate.section_flows.items():
            print(f"    {k}: v={sf.velocity_m_s:.2f}, Re={sf.reynolds:.0f}, Fr={sf.froude:.2f}, A={sf.area_cm2:.2f}")

    print("Generating HTML report ...")
    html_path = os.path.join(args.out_dir, "test_report.html")
    _generate_html(result, html_path)
    print(f"  saved to {html_path}")


if __name__ == "__main__":
    main()
