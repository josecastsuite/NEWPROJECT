"""Geometry-aware gating design engine for JoseCast Analyzer.

This module is intentionally separate from `core/gating.py`.  It produces a
self-contained gating layout (sprue / runner / ingate areas and velocities) from
part geometry, material, and user constraints, using the standard `Q = A·v`
continuity equation with material/wall-thickness dependent velocity targets.

Key principle:
  * Measured/CAD cross-sectional areas (from the 3D viewer) are authoritative.
  * Missing areas are designed from `A = Q / v_target`, where `v_target`
    depends on the gating system, alloy, and wall thickness.
  * The gating system is first guessed from geometry, then the actual system
    can be re-classified from the resulting velocities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.gating_calculator import auto_fill_time, effective_head

G = 9.81  # m/s²


@dataclass
class GatingEngineInput:
    """Inputs for a standalone gating design."""

    # Geometry / mass
    total_metal_volume_m3: float
    total_mass_kg: float
    part_volume_m3: float = 0.0
    part_mass_kg: float = 0.0

    # Casting direction / heights (all in mm)
    part_height_mm: float = 100.0
    total_height_mm: float = 120.0
    max_flow_path_mm: float = 200.0

    # Material
    alloy_key: str = "steel"
    alloy_name: str = ""
    rho_kg_m3: float = 7000.0
    viscosity_pa_s: float = 0.005
    latent_heat_j_kg: float = 2.7e5
    cp_j_kgk: float = 500.0
    t_pour_c: float = 1600.0
    t_liquidus_c: float = 1510.0

    # Wall category: "ince cidarlı", "orta cidarlı", "kalın cidarlı"
    wall_category: Optional[str] = None

    # User constraints
    t_fill_s: Optional[float] = None
    user_gate_velocity_m_s: Optional[float] = None
    user_velocity_section_key: str = "INGATE"
    gating_system: Optional[str] = None  # override auto selection
    max_gates: int = 4
    max_runners: int = 1
    discharge_coeff: float = 0.8

    # Measured areas (cm²) from 3D section picker or CAD bodies
    measured_areas_cm2: Dict[str, float] = field(default_factory=dict)

    # Number of physical ingate/runner bodies if already known
    n_gates: Optional[int] = None
    n_runners: Optional[int] = None

    # Head loss already estimated by caller (m)
    head_loss_m: float = 0.0


@dataclass
class GatingDesign:
    """Result of a standalone gating design."""

    gating_system: str
    choke_section: str
    n_gates: int
    n_runners: int
    t_fill_s: float
    q_m3_s: float
    h_eff_mm: float
    v_choke_m_s: float

    sprue_base_area_cm2: float
    sprue_throat_area_cm2: float
    runner_total_area_cm2: float
    gate_total_area_cm2: float
    gate_each_area_cm2: float

    sprue_velocity_m_s: float
    runner_velocity_m_s: float
    gate_velocity_m_s: float

    reynolds: float
    froude: float
    turbulent: bool

    warnings: List[str] = field(default_factory=list)
    reason: str = ""


# Standard velocity ranges (m/s) from foundry handbooks / Campbell practice.
# They are used to pick a target velocity for each section per system type.
_VELOCITY_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "basınçlı (pressurized)": {
        "sprue": (0.8, 1.2),
        "runner": (1.0, 1.5),
        "gate": (1.8, 2.5),
    },
    "basınçsız (unpressurized)": {
        "sprue": (1.5, 2.0),
        "runner": (0.8, 1.2),
        "gate": (0.4, 0.7),
    },
    "yarı basınçlı (semi-pressurized)": {
        "sprue": (1.0, 1.5),
        "runner": (0.6, 1.0),
        "gate": (0.9, 1.2),
    },
}


def _infer_wall_category(thickness_mm: float) -> str:
    if thickness_mm <= 6.0:
        return "ince cidarlı"
    if thickness_mm <= 25.0:
        return "orta cidarlı"
    return "kalın cidarlı"


def _select_gating_system(wall_category: str, max_flow_path_mm: float) -> Tuple[str, str]:
    """Pick gating system from wall category and flow length."""
    if wall_category == "ince cidarlı" or max_flow_path_mm > 400.0:
        return (
            "basınçlı (pressurized)",
            "İnce cidarlı / uzun akış yolu; meme dar, hız yüksek, çabuk doldurma.",
        )
    if wall_category == "kalın cidarlı":
        return (
            "basınçsız (unpressurized)",
            "Kalın cidarlı / büyük hacim; sprue-taban geniş, türbülansı önleyen düşük hız.",
        )
    return (
        "yarı basınçlı (semi-pressurized)",
        "Orta cidarlı parça; sprue/runner/gate dengeli.",
    )


def _target_velocity(system: str, section: str, alloy_key: str) -> float:
    """Return a mid-range target velocity (m/s) for a section."""
    ranges = _VELOCITY_RANGES.get(
        system, _VELOCITY_RANGES["yarı basınçlı (semi-pressurized)"]
    )
    lo, hi = ranges.get(section, (0.5, 1.0))
    target = (lo + hi) / 2.0
    key = alloy_key.lower()
    if "al" in key or "alum" in key:
        target *= 0.85
    elif "gri" in key or "sfero" in key or "ggg" in key or "pik" in key:
        target *= 0.75
    return float(np.clip(target, lo * 0.5, hi * 1.1))


def _target_range(system: str, section: str) -> Tuple[float, float]:
    ranges = _VELOCITY_RANGES.get(
        system, _VELOCITY_RANGES["yarı basınçlı (semi-pressurized)"]
    )
    return ranges.get(section, (0.5, 1.0))


def _classify_from_velocities(v_sprue: float, v_runner: float, v_gate: float) -> str:
    """Classify the actual gating system from the computed section velocities."""
    # Pressurized: v_sprue < v_runner < v_gate  (gate is smallest area / choke)
    if v_sprue <= v_runner <= v_gate:
        return "basınçlı (pressurized)"
    # Unpressurized: v_sprue > v_runner > v_gate (sprue is smallest area / choke)
    if v_sprue >= v_runner >= v_gate:
        return "basınçsız (unpressurized)"
    # Semi-pressurized: runner is the largest area => lowest velocity
    if v_runner <= v_sprue and v_runner <= v_gate:
        return "yarı basınçlı (semi-pressurized)"
    return "yarı basınçlı (semi-pressurized)"


def _choke_section_for_system(system: str) -> str:
    if system == "basınçsız (unpressurized)":
        return "SPRUE_BASE"
    if system == "basınçlı (pressurized)":
        return "INGATE"
    return "RUNNER"


def _hydraulic_diameter_mm(area_m2: float) -> float:
    """Equivalent circular diameter for a given area."""
    if area_m2 <= 0.0:
        return 1e-6
    return 1000.0 * math.sqrt(4.0 * area_m2 / math.pi)


def _reynolds_froude(
    rho: float, velocity: float, area_m2: float, viscosity: float
) -> Tuple[float, float]:
    D_m = _hydraulic_diameter_mm(area_m2) / 1000.0
    re = rho * velocity * D_m / max(viscosity, 1e-6)
    fr = velocity / math.sqrt(max(G * D_m, 1e-9))
    return float(re), float(fr)


def _fluidity_time_s(
    t_pour_c: float, t_liquidus_c: float, latent_heat_j_kg: float, cp_j_kgk: float
) -> float:
    """Rough time available before a solid shell forms (seconds).

    Uses superheat energy and an assumed 1 MW/m² mold heat flux.
    Only an order-of-magnitude estimate for the fluidity-length check.
    """
    superheat = max(t_pour_c - t_liquidus_c, 0.0)
    if superheat <= 0.0:
        return 0.5
    rho_ref = 7000.0
    energy_j_m3 = rho_ref * (latent_heat_j_kg + cp_j_kgk * superheat)
    q = 1.0e6  # W/m²
    layer_m = 0.001
    t_s = layer_m * energy_j_m3 / q
    return float(np.clip(t_s, 0.2, 20.0))


def calculate_gating_design(inp: GatingEngineInput) -> GatingDesign:
    """Produce a gating layout from geometry and constraints."""

    warnings: List[str] = []

    # 1. Wall category / gating system (initial guess)
    wall_cat = inp.wall_category or "orta cidarlı"
    system = inp.gating_system
    reason = "Kullanıcı tarafından seçildi."
    if system is None:
        system, reason = _select_gating_system(wall_cat, inp.max_flow_path_mm)

    # 2. Effective ferrostatic head
    h_avg_m = (
        max(inp.total_height_mm - 0.5 * inp.part_height_mm, inp.total_height_mm * 0.1)
        / 1000.0
    )
    h_eff_m = effective_head(h_avg_m, inp.total_mass_kg)
    h_eff_m = float(np.clip(h_eff_m - inp.head_loss_m, 0.02, 0.60))

    # 3. Bernoulli choke velocity
    v_choke = inp.discharge_coeff * math.sqrt(2.0 * G * h_eff_m)

    # 4. Fill time
    t_fill = inp.t_fill_s
    if t_fill is None or t_fill <= 0.0:
        t_fill = auto_fill_time(inp.total_mass_kg, inp.alloy_key, inp.alloy_name)
    t_fill = float(np.clip(t_fill, 0.2, 120.0))

    # 5. Measured areas (authoritative) - cm² -> m²
    measured = {
        k.upper(): float(v) for k, v in inp.measured_areas_cm2.items() if v > 0.0
    }
    has_measured = {
        "SPRUE_BASE": "SPRUE_BASE" in measured,
        "SPRUE_THROAT": "SPRUE_THROAT" in measured,
        "RUNNER": "RUNNER" in measured,
        "INGATE": "INGATE" in measured,
    }

    a_gate_total_m2 = measured.get("INGATE", 0.0) / 1e4
    a_runner_total_m2 = measured.get("RUNNER", 0.0) / 1e4
    a_sprue_base_m2 = measured.get("SPRUE_BASE", 0.0) / 1e4
    a_sprue_throat_m2 = measured.get("SPRUE_THROAT", 0.0) / 1e4
    if a_sprue_throat_m2 <= 0.0 and a_sprue_base_m2 > 0.0:
        a_sprue_throat_m2 = a_sprue_base_m2

    # 6. If user gave velocity and a measured area for that section, derive t_fill
    if inp.user_gate_velocity_m_s is not None and inp.user_gate_velocity_m_s > 0.0:
        section = inp.user_velocity_section_key.upper()
        a_user = 0.0
        if section == "INGATE" and a_gate_total_m2 > 0.0:
            a_user = a_gate_total_m2
        elif section == "RUNNER" and a_runner_total_m2 > 0.0:
            a_user = a_runner_total_m2
        elif section in ("SPRUE_BASE", "SPRUE") and a_sprue_base_m2 > 0.0:
            a_user = a_sprue_base_m2
        elif section == "SPRUE_THROAT" and a_sprue_throat_m2 > 0.0:
            a_user = a_sprue_throat_m2

        if a_user > 0.0:
            t_fill = float(
                np.clip(
                    inp.total_metal_volume_m3 / (inp.user_gate_velocity_m_s * a_user),
                    0.2,
                    120.0,
                )
            )
            reason += f" t_fill, kullanıcı hızı ve ölçülen {section} alanından hesaplandı."

    q_m3_s = inp.total_metal_volume_m3 / t_fill

    # 7. Number of gates (based on how far metal must flow before it skins)
    v_gate_target = inp.user_gate_velocity_m_s
    if v_gate_target is None or v_gate_target <= 0.0:
        v_gate_target = _target_velocity(system, "gate", inp.alloy_key)
    else:
        reason += f" Hedef gate hızı: {v_gate_target:.2f} m/s."

    t_fluid = _fluidity_time_s(
        inp.t_pour_c, inp.t_liquidus_c, inp.latent_heat_j_kg, inp.cp_j_kgk
    )
    max_reach_mm = max(v_gate_target * t_fluid * 1000.0 * 0.7, 50.0)

    n_gates = inp.n_gates
    if n_gates is None:
        n_gates = 1
        if inp.max_flow_path_mm > max_reach_mm and max_reach_mm > 0.0:
            n_gates = int(math.ceil(inp.max_flow_path_mm / max_reach_mm))
        n_gates = int(np.clip(n_gates, 1, max(1, inp.max_gates)))

    n_runners = inp.n_runners or 1

    # 8. Compute missing areas from Q/v_target
    v_sprue_target = _target_velocity(system, "sprue", inp.alloy_key)
    v_runner_target = _target_velocity(system, "runner", inp.alloy_key)
    v_gate_target_final = v_gate_target

    if not has_measured["INGATE"]:
        a_gate_total_m2 = q_m3_s / v_gate_target_final
    if not has_measured["RUNNER"]:
        a_runner_total_m2 = q_m3_s / v_runner_target
    if not has_measured["SPRUE_BASE"]:
        a_sprue_base_m2 = q_m3_s / v_sprue_target
    if not has_measured["SPRUE_THROAT"]:
        a_sprue_throat_m2 = a_sprue_base_m2

    # Distribute gate area among gates
    a_gate_each_m2 = a_gate_total_m2 / max(n_gates, 1)

    # 9. Actual velocities from areas
    v_gate = q_m3_s / a_gate_total_m2 if a_gate_total_m2 > 0.0 else 0.0
    v_runner = q_m3_s / a_runner_total_m2 if a_runner_total_m2 > 0.0 else 0.0
    v_sprue = q_m3_s / a_sprue_base_m2 if a_sprue_base_m2 > 0.0 else 0.0

    # 10. Re-classify actual system from velocities and warn on mismatch
    detected_system = _classify_from_velocities(v_sprue, v_runner, v_gate)
    if detected_system != system:
        warnings.append(
            f"Öngörülen sistem '{system}', ölçülen/hızlarla '{detected_system}' çıktı."
        )
        # Use the detected system for range checks, but keep the design geometry.
        check_system = detected_system
    else:
        check_system = system

    # 11. Velocity range warnings
    for section, v in [
        ("sprue", v_sprue),
        ("runner", v_runner),
        ("gate", v_gate),
    ]:
        lo, hi = _target_range(check_system, section)
        if v < lo * 0.8:
            warnings.append(
                f"{section} hızı çok düşük: {v:.2f} m/s (hedef {lo}-{hi})."
            )
        elif v > hi * 1.2:
            warnings.append(
                f"{section} hızı çok yüksek: {v:.2f} m/s (hedef {lo}-{hi})."
            )

    # 12. Reynolds / Froude at the gate (most critical)
    re, fr = _reynolds_froude(
        inp.rho_kg_m3, v_gate, a_gate_each_m2, inp.viscosity_pa_s
    )
    gate_lo, gate_hi = _target_range(check_system, "gate")
    turbulent = re > 20000.0 or v_gate > gate_hi

    if turbulent:
        warnings.append(
            f"Gate akışı türbülanslı (Re={re:.0f}). Hız sınırını düşür veya alanı büyüt."
        )

    # 13. Fluidity length check
    fluidity_length_mm = v_gate * t_fluid * 1000.0
    if fluidity_length_mm < inp.max_flow_path_mm:
        warnings.append(
            f"Akışkanlık uzunluğu ({fluidity_length_mm:.0f} mm) en uzak noktaya "
            f"({inp.max_flow_path_mm:.0f} mm) yetmiyor; meme sayısı veya hız artırılmalı."
        )

    choke = _choke_section_for_system(detected_system)

    return GatingDesign(
        gating_system=detected_system,
        choke_section=choke,
        n_gates=n_gates,
        n_runners=n_runners,
        t_fill_s=t_fill,
        q_m3_s=q_m3_s,
        h_eff_mm=h_eff_m * 1000.0,
        v_choke_m_s=v_choke,
        sprue_base_area_cm2=a_sprue_base_m2 * 1e4,
        sprue_throat_area_cm2=a_sprue_throat_m2 * 1e4,
        runner_total_area_cm2=a_runner_total_m2 * 1e4,
        gate_total_area_cm2=a_gate_total_m2 * 1e4,
        gate_each_area_cm2=a_gate_each_m2 * 1e4,
        sprue_velocity_m_s=v_sprue,
        runner_velocity_m_s=v_runner,
        gate_velocity_m_s=v_gate,
        reynolds=re,
        froude=fr,
        turbulent=turbulent,
        warnings=warnings,
        reason=reason,
    )


def design_from_analysis(
    total_metal_volume_m3: float,
    total_mass_kg: float,
    part_height_mm: float,
    total_height_mm: float,
    max_flow_path_mm: float,
    wall_thickness_mm: float,
    alloy_key: str,
    alloy_name: str = "",
    t_fill_s: Optional[float] = None,
    user_gate_velocity_m_s: Optional[float] = None,
    measured_areas_cm2: Optional[Dict[str, float]] = None,
    **kwargs,
) -> GatingDesign:
    """Convenience wrapper that builds `GatingEngineInput` from common values."""
    inp = GatingEngineInput(
        total_metal_volume_m3=total_metal_volume_m3,
        total_mass_kg=total_mass_kg,
        part_height_mm=part_height_mm,
        total_height_mm=total_height_mm,
        max_flow_path_mm=max_flow_path_mm,
        wall_category=_infer_wall_category(wall_thickness_mm),
        alloy_key=alloy_key,
        alloy_name=alloy_name,
        t_fill_s=t_fill_s,
        user_gate_velocity_m_s=user_gate_velocity_m_s,
        measured_areas_cm2=measured_areas_cm2 or {},
        **kwargs,
    )
    return calculate_gating_design(inp)
