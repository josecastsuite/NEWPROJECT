"""Render shrinkage porosity risk on the part surface for sand/metal/ceramic molds."""
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
OUT_DIR = Path(__file__).parent / "results" / "porosity_mold"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "Deneme_Ring.STEP"
ALLOY = "42CrMo4"
GRAVITY = (0.0, 0.0, -1.0)


def render(result, png_path: Path, title: str):
    shape = result.grid.shape
    dx = float(result.dx_mm)
    origin = np.asarray(result.origin_mm, dtype=np.float64)
    risk = np.asarray(result.risk)
    part_mask = result.grid == int(BodyType.PART)
    if risk.size == 0 or risk.max() <= 0:
        print(f"  [render] {png_path.name}: no risk")
        return

    grid = pv.ImageData()
    grid.dimensions = np.array(shape) + 1
    grid.origin = origin
    grid.spacing = (dx, dx, dx)
    grid.cell_data["body"] = result.grid.ravel(order="F").astype(np.float64)
    grid.cell_data["risk"] = risk.ravel(order="F").astype(np.float64)

    part_surf = grid.threshold([1, 1], scalars="body").extract_surface(
        algorithm="dataset_surface"
    )

    centers = grid.cell_centers()
    body = centers.point_data["body"]
    r = centers.point_data["risk"]
    mask = (body == int(BodyType.PART)) & (r > 0.02)
    if not mask.any():
        print(f"  [render] {png_path.name}: no points after threshold")
        return

    pts = centers.points[mask]
    vals = r[mask]
    cloud = pv.PolyData(pts)
    cloud["risk"] = vals

    pl = pv.Plotter(off_screen=True, window_size=(1400, 900))
    pl.set_background("white")
    if part_surf.n_points:
        pl.add_mesh(part_surf, color="#d1d5db", opacity=0.15, show_edges=False)
    point_size = max(5, min(12, int(round(dx * 2.5))))
    pl.add_points(
        cloud,
        scalars="risk",
        cmap="coolwarm",
        clim=[0.0, 1.0],
        render_points_as_spheres=True,
        point_size=point_size,
        opacity=0.85,
        show_scalar_bar=True,
        scalar_bar_args={
            "title": title,
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


def run_one(mold_key: str, mold_temp_c: float):
    step_path = DATA_DIR / MODEL
    bodies = load_step(str(step_path))
    apply_unit_scale(bodies, "mm")
    grid, body_index, origin, dx, bodies = build_voxel_grid(
        bodies, target_dim=80, gravity_vector=GRAVITY
    )

    params = CastingParameters(
        t_pour_c=1600.0,
        t_liquidus_c=1510.0,
        t_solidus_c=1410.0,
        t_mold_c=mold_temp_c,
        t_fill_s=0.0,
        rho_liquid_kg_m3=7850.0,
        viscosity_pa_s=0.005,
        gravity_direction=GRAVITY,
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
        alloy_key=ALLOY,
        mold_key=mold_key,
        base_res=80,
        max_res=240,
        refine_local=False,
        sub_voxel=1,
        thermal_max_time_s=600.0,
        thermal_downsample=2,
        casting_params=params,
    )

    png = OUT_DIR / f"porosity_{mold_key}_{int(mold_temp_c)}C.png"
    render(result, png, "Porozite riski")
    return {
        "mold": mold_key,
        "mold_temp_c": mold_temp_c,
        "B": result.chvorinov_c,
        "risk_max": float(result.risk.max()) if result.risk.size else None,
        "risk_mean": float(result.risk.mean()) if result.risk.size else None,
        "screenshot": str(png),
    }


def main():
    rows = []
    for mold_key, t0 in [
        ("sand", 25.0),
        ("metal_mold", 25.0),
        ("ceramic", 900.0),
    ]:
        print(f"[porosity] {mold_key} @ {t0}°C ...")
        try:
            rows.append(run_one(mold_key, t0))
        except Exception as exc:
            print(f"  ERROR: {exc}")
            rows.append({"mold": mold_key, "error": str(exc)})
    import json
    (OUT_DIR / "summary.json").write_text(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
