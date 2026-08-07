"""Render air-entrapment risk point cloud on top of the part surface."""
import os
from pathlib import Path

import numpy as np
import pyvista as pv

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.step_loader import load_step
from core.voxelizer import build_voxel_grid, apply_unit_scale
from core.sdf_analyzer import analyze
from core.types import CastingParameters, BodyType

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent / "results" / "surface_render"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("Model_Knuckle.STEP", (0.0, 0.0, -1.0)),
    ("Deneme_Ring.STEP", (0.0, 0.0, -1.0)),
]
MOLDS = ["sand", "metal_mold"]


def render(result, png_path: Path):
    shape = result.grid.shape
    dx = float(result.dx_mm)
    origin = np.asarray(result.origin_mm, dtype=np.float64)

    grid = pv.ImageData()
    grid.dimensions = np.array(shape) + 1
    grid.origin = origin
    grid.spacing = (dx, dx, dx)
    grid.cell_data["body"] = result.grid.ravel(order="F").astype(np.float64)
    grid.cell_data["is_metal"] = result.is_metal.ravel(order="F").astype(np.float64)
    grid.cell_data["air"] = result.air_entrapment.ravel(order="F").astype(np.float64)

    # Metal surface (part + gating + riser) rendered translucent so the risk
    # points can be seen in their real geometric context.
    try:
        metal = grid.threshold([0.5, 1.5], scalars="is_metal")
        metal_surf = metal.extract_surface(algorithm='dataset_surface')
    except Exception:
        metal_surf = None

    # Risk points: cell centers where is_metal and air > 0.02.
    centers = grid.cell_centers()
    is_metal = centers.point_data["is_metal"]
    air = centers.point_data["air"]
    mask = (is_metal >= 0.5) & (air >= 0.02)

    if not mask.any():
        print(f"  [render] {png_path.name}: no risk points")
        return

    pts = centers.points[mask]
    vals = air[mask]
    cloud = pv.PolyData(pts)
    cloud["air"] = vals

    lut = pv.LookupTable(cmap="coolwarm", scalar_range=(0.0, 1.0))
    lut.annotations = {0.0: "az riskli", 0.5: "riskli", 1.0: "çok riskli"}

    pl = pv.Plotter(off_screen=True, window_size=(1400, 900))
    pl.set_background("white")
    if metal_surf is not None and metal_surf.n_points:
        pl.add_mesh(metal_surf, color="#d1d5db", opacity=0.18, show_edges=False, smooth_shading=False)
    point_size = max(5, min(12, int(round(dx * 2.5))))
    pl.add_points(
        cloud,
        scalars="air",
        cmap=lut,
        clim=[0.0, 1.0],
        render_points_as_spheres=True,
        point_size=point_size,
        opacity=0.85,
        show_scalar_bar=True,
        scalar_bar_args={
            "title": "Hava sıkışması",
            "n_labels": 0,
            "vertical": False,
            "position_x": 0.12,
            "position_y": 0.02,
            "width": 0.76,
            "height": 0.06,
            "title_font_size": 9,
            "label_font_size": 7,
            "color": "#334155",
        },
    )
    pl.reset_camera()
    pl.camera.azimuth = 120
    pl.camera.elevation = 20
    pl.screenshot(png_path, transparent_background=False)
    print(f"  [render] {png_path.name}: {len(vals)} points, max={vals.max():.3f}")


def run_one(step_path: Path, mold_key: str, gravity: tuple):
    bodies = load_step(str(step_path))
    apply_unit_scale(bodies, "mm")
    grid, body_index, origin, dx, bodies = build_voxel_grid(bodies, target_dim=80, gravity_vector=gravity)

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
        velocity_section_key="SPRUE_THROAT",
        fast_flow=True,
    )
    result = analyze(
        bodies,
        grid,
        body_index,
        origin,
        dx,
        alloy_key="42CrMo4",
        mold_key=mold_key,
        base_res=80,
        max_res=240,
        refine_local=False,
        sub_voxel=1,
        thermal_max_time_s=0.0,
        thermal_downsample=2,
        casting_params=params,
    )
    png = OUT_DIR / f"air_{step_path.stem}_{mold_key}_surface.png"
    render(result, png)


def main():
    for model, gravity in MODELS:
        for mold in MOLDS:
            step = DATA_DIR / model
            if not step.exists():
                continue
            print(f"[surface] {model} / {mold}")
            try:
                run_one(step, mold, gravity)
            except Exception as exc:
                print(f"  ERROR: {exc}")


if __name__ == "__main__":
    main()
