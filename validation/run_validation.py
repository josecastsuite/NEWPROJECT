"""Batch validation script for JoséCast analyzer.

Runs the full analysis pipeline on every *.STEP file under `data/` and writes a
JSON/Markdown summary plus a short per-model sanity check.  It is intentionally
simple: it does not require PyQt/PyVista and only exercises the core solver +
reporting paths.
"""
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from core.step_loader import load_step
from core.voxelizer import build_voxel_grid, apply_unit_scale
from core.sdf_analyzer import analyze
from core.materials import get_alloy, get_mold, chvorinov_c_from_properties
from core.types import CastingParameters


DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent / "results"
MODELS = sorted(DATA_DIR.glob("*.STEP"))


def _collect(result) -> Dict[str, Any]:
    fr = result.flow_result
    flow: Dict[str, Any] = {}
    if fr is not None:
        vmag_arr = fr.velocity_magnitude
        if vmag_arr is None:
            # Fallback to discrete gating-node velocities (simple hydraulic path).
            vmag_arr = np.array(
                [
                    float(getattr(n, "max_velocity_m_s", 0.0) or n.velocity_m_s)
                    for n in (fr.gating_nodes or [])
                    if getattr(n, "velocity_m_s", 0.0) > 1e-12
                ],
                dtype=np.float64,
            )
        vmag = np.asarray(vmag_arr)
        nonzero = vmag[vmag > 1e-12] if vmag.size else vmag
        flow = {
            "fill_time_s": float(fr.fill_time_s),
            "Q_m3_s": float(fr.Q_m3_s),
            "inlet_area_cm2": float(fr.inlet_area_m2 * 1e4),
            "ingate_contact_velocity_m_s": float(fr.ingate_contact_velocity_m_s),
            "max_velocity_m_s": float(np.max(vmag)) if vmag.size else 0.0,
            "mean_velocity_m_s": float(np.mean(nonzero)) if nonzero.size else 0.0,
            "filter_recommendation": fr.filter_recommendation or "",
            "reason": fr.reason or "",
            "node_velocities": fr.node_velocities,
        }
        if fr.reynolds is not None and fr.reynolds.size:
            re = np.asarray(fr.reynolds)
            flow["max_reynolds"] = float(np.max(re))
            flow["mean_reynolds"] = float(np.mean(re[re > 1e-6])) if np.any(re > 1e-6) else 0.0
        if fr.turbulence_intensity is not None and fr.turbulence_intensity.size:
            ti = np.asarray(fr.turbulence_intensity)
            flow["max_turbulence_intensity_pct"] = float(np.max(ti) * 100.0)

    visible = [hs for hs in result.hotspots if not hs.solved]
    thermal: Dict[str, Any] = {
        "total_hotspots": len(result.hotspots),
        "visible_hotspots": len(visible),
        "max_pore_um": max((hs.pore_size_um for hs in visible), default=0.0),
    }

    proposals = []
    for rp in result.riser_proposals or []:
        proposals.append({
            "shape": rp.shape,
            "diameter_mm": float(rp.diameter_mm),
            "height_mm": float(rp.height_mm),
            "volume_cm3": float(rp.volume_cm3),
            "exothermic": bool(rp.exothermic),
            "infeasible": bool(rp.infeasible),
            "reason": rp.reason,
        })

    return {
        "flow": flow,
        "thermal": thermal,
        "riser_proposals": proposals,
        "gate_exists": result.gate_result is not None,
    }


def _run_one(step_path: Path) -> Dict[str, Any]:
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
    summary = {
        "model": step_path.name,
        "bodies": len(bodies),
        "grid_shape": list(grid.shape),
        "dx_mm": float(dx),
        "elapsed_s": elapsed,
        "chvorinov_c": chvorinov_c_from_properties(get_alloy(alloy_key), get_mold(mold_key)),
    }
    summary.update(_collect(result))
    return summary


def _check(summary: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    if summary.get("elapsed_s", 0.0) > 180.0:
        issues.append(f"analyze > {180.0}s")
    flow = summary.get("flow", {})
    if not flow:
        issues.append("no flow result")
    else:
        if flow.get("fill_time_s", 0.0) <= 0.0:
            issues.append("fill_time <= 0")
        if flow.get("Q_m3_s", 0.0) <= 0.0:
            issues.append("flow rate <= 0")
    thermal = summary.get("thermal", {})
    if thermal.get("visible_hotspots", 0) > 0:
        issues.append(f"{thermal['visible_hotspots']} unresolved hot spot(s)")
    return issues


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for step_path in MODELS:
        print(f"[validation] {step_path.name} ...", flush=True)
        t0 = time.time()
        try:
            row = _run_one(step_path)
        except Exception as exc:
            elapsed = time.time() - t0
            row = {"model": step_path.name, "error": str(exc), "elapsed_s": elapsed}
        row["issues"] = _check(row)
        rows.append(row)
        print(f"  elapsed={row.get('elapsed_s', 'n/a')}s issues={row['issues']}", flush=True)

    json_path = OUT_DIR / "validation_report.json"
    json_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    md_lines = ["# JoséCast Batch Validation Report\n"]
    md_lines.append(f"Models: {len(MODELS)}\n")
    overall_issues = sum(len(r.get("issues", [])) for r in rows)
    md_lines.append(f"Total issues: {overall_issues}\n\n")
    md_lines.append("| Model | Bodies | Grid | dx (mm) | Fill (s) | max Re | Visible HS | Issues |\n")
    md_lines.append("|-------|--------|------|---------|----------|--------|------------|--------|\n")
    for r in rows:
        flow = r.get("flow", {})
        dx_s = f"{r['dx_mm']:.4f}" if "dx_mm" in r else "-"
        ft_s = f"{flow['fill_time_s']:.2f}" if "fill_time_s" in flow else "-"
        re_s = f"{flow['max_reynolds']:.0f}" if "max_reynolds" in flow else "-"
        hs_s = f"{r.get('thermal', {}).get('visible_hotspots', '-')}" if "thermal" in r else "-"
        md_lines.append(
            f"| {r['model']} | {r.get('bodies', '-')} | {r.get('grid_shape', '-')} | "
            f"{dx_s} | {ft_s} | {re_s} | {hs_s} | "
            f"{', '.join(r.get('issues', [])) or 'ok'} |\n"
        )
    md_lines.append("\n## Details\n\n")
    for r in rows:
        md_lines.append(f"### {r['model']}\n")
        md_lines.append(f"```json\n{json.dumps(r, indent=2, default=str)}\n```\n")

    md_path = OUT_DIR / "validation_report.md"
    md_path.write_text("".join(md_lines), encoding="utf-8")
    print(f"[validation] reports: {json_path} {md_path}", flush=True)


if __name__ == "__main__":
    main()
