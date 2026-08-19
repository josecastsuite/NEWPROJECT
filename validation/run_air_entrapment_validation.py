"""Validation sweep for air-entrapment surface-cloud model.

Runs analyze() on Model_Knuckle and Deneme_Ring with sand, ceramic and metal
molds, then renders a coolwarm point-cloud screenshot of the risk field.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pyvista as pv

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.step_loader import load_step
from core.voxelizer import build_voxel_grid, apply_unit_scale
from core.sdf_analyzer import analyze
from core.materials import get_alloy
from core.types import CastingParameters, BodyType

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["Model_Knuckle.STEP", "Deneme_Ring.STEP"]
MOLDS = ["sand", "ceramic", "metal_mold"]


def _render_screenshot(result, png_path: Path):
    """Off-screen render the air-entrapment point cloud with categorical scalar bar."""
    shape = result.grid.shape
    dx = float(result.dx_mm)
    origin = np.asarray(result.origin_mm, dtype=np.float64)

    grid = pv.ImageData()
    grid.dimensions = np.array(shape) + 1
    grid.origin = origin
    grid.spacing = (dx, dx, dx)
    grid.cell_data["air_entrapment"] = np.asarray(result.air_entrapment).ravel(order="F")
    grid.cell_data["is_metal"] = result.is_metal.ravel(order="F").astype(np.float64)
    grid = grid.cell_data_to_point_data()

    pts = np.asarray(grid.points)
    is_metal = grid["is_metal"]
    air = grid["air_entrapment"]
    mask = (is_metal >= 0.5) & (air >= 0.02)
    if not mask.any():
        print(f"  [render] {png_path.name}: no risk points above 0.02")
        return False

    selected = pts[mask]
    values = air[mask]
    cloud = pv.PolyData(selected)
    cloud["air"] = values

    lut = pv.LookupTable(cmap="coolwarm", scalar_range=(0.0, 1.0))
    lut.annotations = {0.0: "az riskli", 0.5: "riskli", 1.0: "çok riskli"}

    point_size = max(4, min(10, int(round(dx * 2.0))))

    pl = pv.Plotter(off_screen=True, window_size=(1200, 900))
    pl.set_background("white")
    pl.add_points(
        cloud,
        scalars="air",
        cmap=lut,
        clim=[0.0, 1.0],
        render_points_as_spheres=True,
        point_size=point_size,
        opacity=0.6,
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
    pl.camera.azimuth = 120
    pl.camera.elevation = 15
    pl.reset_camera()
    pl.screenshot(png_path, transparent_background=False)
    return True


def _run_one(step_path: Path, mold_key: str) -> dict:
    t0 = time.time()
    gravity = (0.0, -1.0, 0.0) if "parca" in step_path.name.lower() else (0.0, 0.0, -1.0)

    bodies = load_step(str(step_path))
    apply_unit_scale(bodies, "mm")

    grid, body_index, origin, dx, bodies = build_voxel_grid(
        bodies, target_dim=80, gravity_vector=gravity
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

    air = result.air_entrapment
    info = {
        "model": step_path.name,
        "mold": mold_key,
        "grid_shape": list(grid.shape),
        "dx_mm": float(dx),
        "elapsed_s": time.time() - t0,
        "air_max": float(air.max()) if air is not None and air.size else None,
        "air_min": float(air.min()) if air is not None and air.size else None,
        "trapped_air_volume_m3": float(result.trapped_air_volume_m3) if result is not None else None,
    }
    print(
        f"  [validation] {step_path.name} / {mold_key}: "
        f"air_max={info['air_max']:.4f} vol={info['trapped_air_volume_m3']:.3e} "
        f"({info['elapsed_s']:.1f}s)",
        flush=True,
    )

    png = OUT_DIR / f"air_{step_path.stem}_{mold_key}.png"
    if _render_screenshot(result, png):
        info["screenshot"] = str(png)
    return info


def main():
    rows = []
    for model in MODELS:
        step_path = DATA_DIR / model
        if not step_path.exists():
            print(f"[validation] missing {step_path}, skipping")
            continue
        for mold_key in MOLDS:
            print(f"[validation] {model} / {mold_key} ...", flush=True)
            try:
                rows.append(_run_one(step_path, mold_key))
            except Exception as exc:
                print(f"  ERROR: {exc}")
                rows.append({"model": model, "mold": mold_key, "error": str(exc)})

    out = OUT_DIR / "air_entrapment_validation.json"
    out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"[validation] report: {out}")


if __name__ == "__main__":
    main()
