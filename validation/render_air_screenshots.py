"""Generate off-screen PyVista volume/isosurface screenshots for Deneme_Ring."""
import json
import os
import sys
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import pyvista as pv

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from core.materials import get_alloy, get_mold
from core.sdf_analyzer import analyze
from core.step_loader import load_step
from core.types import CastingParameters
from core.voxelizer import apply_unit_scale, build_voxel_grid


def render_air(result, title: str, out_png: Path):
    grid = pv.ImageData()
    shape = np.asarray(result.air_entrapment.shape)
    grid.dimensions = shape + 1
    grid.origin = result.origin_mm
    grid.spacing = (result.dx_mm, result.dx_mm, result.dx_mm)
    grid.cell_data["air"] = np.asarray(result.air_entrapment).ravel(order="F").astype(np.float64)
    grid = grid.cell_data_to_point_data()

    pl = pv.Plotter(off_screen=True, window_size=(1280, 1024))
    contours = grid.contour(isosurfaces=[0.05, 0.20, 0.50, 0.80], scalars="air")
    if contours.n_points > 0:
        pl.add_mesh(
            contours,
            cmap="inferno",
            clim=[0.0, 1.0],
            smooth_shading=True,
            specular=0.8,
            opacity=0.9,
            show_scalar_bar=True,
            scalar_bar_args={"title": "Hava Sıkışması (α_g)", "vertical": True},
        )
    pl.add_volume(
        grid,
        scalars="air",
        cmap="inferno",
        opacity="sigmoid",
        clim=[0.02, 1.0],
        show_scalar_bar=False,
    )
    pl.add_text(title, position="upper_edge", font_size=14, color="white")
    pl.camera_position = "xz"
    pl.reset_camera()
    pl.screenshot(str(out_png))
    print(f"wrote {out_png}")


def run_and_render(mold_key: str, out_png: Path):
    os.environ.setdefault("JOSECAST_CPP_LBM_MAX_STEPS", "2000")
    os.environ.setdefault("JOSECAST_CPP_LBM_CALLBACK_EVERY_N", "50")
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
    print(f"{mold_key} analyze elapsed {time.time()-t0:.1f}s")
    ae = res.air_entrapment
    if ae is None or ae.size == 0:
        raise RuntimeError(f"no air_entrapment for {mold_key}")
    stats = {
        "mold": mold_key,
        "max": float(np.max(ae)),
        "mean": float(np.mean(ae)),
        "trapped_air_volume_m3": float(res.trapped_air_volume_m3),
    }
    print(stats)
    title = f"{mold_key} - max={stats['max']:.2f} mean={stats['mean']:.4f} vol={stats['trapped_air_volume_m3']:.2e} m³"
    render_air(res, title, out_png)
    return stats


def main():
    out_dir = Path(__file__).parent / "results" / "screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_stats = {}
    for mold_key in ["sand", "ceramic", "metal_mold"]:
        out_png = out_dir / f"Deneme_Ring_{mold_key}_air_entrapment.png"
        try:
            all_stats[mold_key] = run_and_render(mold_key, out_png)
        except Exception as e:
            print(f"ERROR {mold_key}: {e}")
    with open(out_dir / "screenshot_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStats: {all_stats}")


if __name__ == "__main__":
    main()
