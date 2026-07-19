"""Geometry-aware gating design engine for JoseCast Analyzer.

The engine does **not** blindly trust user-measured cross-sectional areas.  It
calculates a physically consistent design from the casting geometry and the
material, then compares the measured areas (if any) to that design and warns
about mismatches.

Design sequence (Campbell / field-script compatible):
  1. geometry + material -> recommended gating system and target gate speed
  2. fill time            -> Campbell or practical table, user override wins
  3. effective head       -> H_eff with mass-dependent head reduction
  4. sprue base area      -> A_s = W / (rho * Cd * t_fill * sqrt(2gH_eff))
  5. gate total area      -> A_g = Q / v_gate_target  (standard ingate formula)
  6. runner area / ratio  -> chosen so the area order matches the system type
  7. gate count           -> from flow-path length and safe local velocity
  8. measured comparison  -> Q/A for each measured area, warnings if far off
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.gating_calculator import (
    G,
    auto_fill_time,
    calc_campbell_parameters,
    effective_head,
)


@dataclass
class GatingEngineInput:
    """Inputs for a standalone gating design."""

    total_metal_volume_m3: float
    total_mass_kg: float
    part_volume_m3: float = 0.0
    part_mass_kg: float = 0.0

    part_height_mm: float = 100.0
    total_height_mm: float = 120.0
    max_flow_path_mm: float = 200.0
    wall_thickness_mm: float = 20.0

    alloy_key: str = "steel"
    alloy_name: str = ""
    rho_kg_m3: float = 7000.0
    viscosity_pa_s: float = 0.005
    latent_heat_j_kg: float = 2.7e5
    cp_j_kgk: float = 500.0
    t_pour_c: float = 1600.0
    t_liquidus_c: float = 1510.0

    gating_system: Optional[str] = None
    gating_ratio: Optional[Tuple[float, float, float]] = None

    t_fill_s: Optional[float] = None
    user_gate_velocity_m_s: Optional[float] = None
    user_velocity_section_key: str = "INGATE"

    discharge_coeff: float = 0.8
    max_gates: int = 4
    n_gates: Optional[int] = None

    measured_areas_cm2: Dict[str, float] = field(default_factory=dict)

    head_loss_m: float = 0.0


@dataclass
class GatingDesign:
    """Result of a standalone gating design.

    The plain fields (sprue_base_area_cm2, runner_total_area_cm2, ...,
    *_velocity_m_s) are the **engine's own geometry-based recommendation**.
    The measured_* fields are optional cross-checks against user/CAD values.
    """

    gating_system: str
    recommended_gating_system: str
    choke_section: str
    n_gates: int
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

    measured_sprue_base_area_cm2: float = 0.0
    measured_sprue_throat_area_cm2: float = 0.0
    measured_runner_total_area_cm2: float = 0.0
    measured_gate_total_area_cm2: float = 0.0

    measured_sprue_velocity_m_s: float = 0.0
    measured_runner_velocity_m_s: float = 0.0
    measured_gate_velocity_m_s: float = 0.0

    warnings: List[str] = field(default_factory=list)
    reason: str = ""


# Backwards-compatible velocity ranges used by the UI / reporter.
_VELOCITY_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "basınçlı (pressurized)": {
        "sprue": (0.8, 2.0),
        "runner": (1.0, 2.5),
        "gate": (2.0, 4.5),
    },
    "basınçsız (unpressurized)": {
        "sprue": (1.0, 2.0),
        "runner": (0.5, 1.2),
        "gate": (0.2, 0.7),
    },
    "yarı basınçlı (semi-pressurized)": {
        "sprue": (0.8, 1.5),
        "runner": (0.5, 1.0),
        "gate": (0.4, 1.0),
    },
}


def _alloy_kind(key: str) -> str:
    k = (key or "").lower()
    if "gri pik" in k or "sfero" in k or "ggg" in k or "nodular" in k or "gray" in k or "pik" in k:
        return "gray"
    if "al" in k or "alsi" in k:
        return "aluminum"
    if "bronz" in k or "bronze" in k or "cu" in k:
        return "bronze"
    return "steel"


def _wall_class(thickness_mm: float) -> str:
    if thickness_mm < 8.0:
        return "thin"
    if thickness_mm <= 20.0:
        return "medium"
    return "thick"


def _safe_gate_velocity_m_s(alloy_key: str, wall_class: str = "medium") -> float:
    """Target (laminar / Campbell-like) gate velocity for the material + wall."""
    kind = _alloy_kind(alloy_key)
    if kind == "gray":
        return {"thin": 2.5, "medium": 2.0, "thick": 1.5}.get(wall_class, 2.0)
    if kind == "aluminum":
        return {"thin": 0.5, "medium": 0.4, "thick": 0.3}.get(wall_class, 0.4)
    if kind == "bronze":
        return {"thin": 1.0, "medium": 0.8, "thick": 0.6}.get(wall_class, 0.8)
    # steel
    return {"thin": 0.7, "medium": 0.5, "thick": 0.4}.get(wall_class, 0.5)


def _max_gate_velocity_m_s(alloy_key: str) -> float:
    kind = _alloy_kind(alloy_key)
    if kind == "gray":
        return 4.5
    if kind == "aluminum":
        return 0.6
    if kind == "bronze":
        return 1.2
    return 0.85


def _max_runner_velocity_m_s(alloy_key: str) -> float:
    # Runner is usually a bit faster than the gate in pressurized systems,
    # but should never exceed the gate max by much.
    return _max_gate_velocity_m_s(alloy_key) * 1.5


def _recommend_system(
    alloy_key: str,
    wall_class: str,
    flow_path_to_height: float,
    user_system: Optional[str] = None,
) -> str:
    if user_system:
        return user_system

    kind = _alloy_kind(alloy_key)

    # Long / thin parts prefer unpressurized (slow, controlled fill).
    long_thin = flow_path_to_height > 2.0 or wall_class == "thin"

    if kind == "gray":
        # Grey/sfero iron can be pressurized for short, heavy sections.
        if wall_class == "thick" and flow_path_to_height < 1.0:
            return "basınçlı (pressurized)"
        if long_thin:
            return "basınçsız (unpressurized)"
        return "yarı basınçlı (semi-pressurized)"

    if kind == "aluminum":
        return "basınçsız (unpressurized)"

    if kind == "bronze":
        return "yarı basınçlı (semi-pressurized)"

    # Steel:
    if long_thin:
        return "basınçsız (unpressurized)"
    if wall_class == "thick" and flow_path_to_height < 1.0:
        return "yarı basınçlı (semi-pressurized)"
    return "yarı basınçlı (semi-pressurized)"


def _classify_from_velocities(v_s: float, v_r: float, v_g: float) -> str:
    """Classify the real system from actual velocities (Q/A).

    Pressurized  -> area shrinks: sprue slowest, gate fastest (v_s < v_r < v_g).
    Unpressurized -> area grows:   sprue fastest, gate slowest (v_s > v_r > v_g).
    """
    if v_s < v_g:
        # metal accelerates through the system -> pressurized-like
        if v_s <= v_r <= v_g:
            return "basınçlı (pressurized)"
        return "yarı basınçlı (semi-pressurized)"
    if v_s > v_g:
        # metal decelerates -> unpressurized-like
        if v_s >= v_r >= v_g:
            return "basınçsız (unpressurized)"
        return "yarı basınçlı (semi-pressurized)"
    return "yarı basınçlı (semi-pressurized)"


def _choke_section_for_system(system: str) -> str:
    if system == "basınçsız (unpressurized)":
        return "SPRUE_BASE"
    if system == "basınçlı (pressurized)":
        return "INGATE"
    return "RUNNER"


def _hydraulic_diameter_m(area_m2: float) -> float:
    if area_m2 <= 0.0:
        return 1e-6
    return math.sqrt(4.0 * area_m2 / math.pi)


def _reynolds_froude(
    rho: float, velocity: float, area_m2: float, viscosity: float
) -> Tuple[float, float]:
    D_m = _hydraulic_diameter_m(area_m2)
    re = rho * velocity * D_m / max(viscosity, 1e-6)
    fr = velocity / math.sqrt(max(G * D_m, 1e-9))
    return float(re), float(fr)


def _resolve_fill_time(
    inp: GatingEngineInput,
    measured: Dict[str, float],
) -> Tuple[float, str]:
    """Choose a fill time from user, measured velocity, Campbell or practical table."""
    if inp.t_fill_s is not None and inp.t_fill_s > 0.0:
        return float(np.clip(inp.t_fill_s, 0.2, 120.0)), "Kullanıcı t_fill."

    if (
        inp.user_gate_velocity_m_s is not None
        and inp.user_gate_velocity_m_s > 0.0
    ):
        section = (inp.user_velocity_section_key or "INGATE").upper()
        a = measured.get(section, 0.0)
        if a <= 0.0:
            a = measured.get("INGATE", 0.0) or measured.get("RUNNER", 0.0) or measured.get("SPRUE_BASE", 0.0)
        if a > 0.0:
            t_fill = inp.total_metal_volume_m3 / (inp.user_gate_velocity_m_s * (a / 1e4))
            return float(np.clip(t_fill, 0.2, 120.0)), (
                f"t_fill, kullanıcı hızı ({inp.user_gate_velocity_m_s:.2f} m/s) "
                f"ve ölçülen {section} alanından hesaplandı."
            )

    part_mass = inp.part_mass_kg if inp.part_mass_kg > 0.0 else inp.total_mass_kg
    superheat = max(inp.t_pour_c - inp.t_liquidus_c, 0.0)

    t_campbell = None
    if inp.wall_thickness_mm > 0.0 and superheat > 0.0:
        cp = calc_campbell_parameters(
            part_mass, inp.rho_kg_m3, inp.wall_thickness_mm, superheat
        )
        t_campbell = float(cp["t_fill"])

    t_auto = auto_fill_time(inp.alloy_name or inp.alloy_key, part_mass)

    if t_campbell is not None:
        t_fill = max(t_campbell, t_auto)
        reason = f"Campbell ({t_campbell:.2f} s) ve pratik ({t_auto:.2f} s) arasından güvenli seçim."
    else:
        t_fill = t_auto
        reason = f"Pratik dolum süresi: {t_auto:.2f} s."

    return float(np.clip(t_fill, 0.2, 120.0)), reason


def _fluidity_time_s(
    t_pour_c: float, t_liquidus_c: float, latent_heat_j_kg: float, cp_j_kgk: float
) -> float:
    superheat = max(t_pour_c - t_liquidus_c, 0.0)
    if superheat <= 0.0:
        return 0.5
    rho_ref = 7000.0
    energy_j_m3 = rho_ref * (latent_heat_j_kg + cp_j_kgk * superheat)
    q = 1.0e6
    layer_m = 0.001
    t_s = layer_m * energy_j_m3 / q
    return float(np.clip(t_s, 0.2, 20.0))


def _area_from_measured_cm2(a_cm2: float) -> float:
    return float(a_cm2) / 1e4 if a_cm2 > 0.0 else 0.0


def calculate_gating_design(inp: GatingEngineInput) -> GatingDesign:
    warnings: List[str] = []

    # Normalise measured areas
    measured: Dict[str, float] = {
        k.upper(): float(v) for k, v in (inp.measured_areas_cm2 or {}).items() if v > 0.0
    }

    # 1. Geometry / material classification
    wall_cat = _wall_class(inp.wall_thickness_mm)
    flow_factor = (
        inp.max_flow_path_mm / max(inp.total_height_mm, inp.part_height_mm, 1.0)
        if max(inp.total_height_mm, inp.part_height_mm, 1.0) > 0.0
        else 1.0
    )
    recommended_system = _recommend_system(
        inp.alloy_key, wall_cat, flow_factor, inp.gating_system
    )
    v_gate_target = _safe_gate_velocity_m_s(inp.alloy_key, wall_cat)
    v_gate_max = _max_gate_velocity_m_s(inp.alloy_key)

    # 2. Fill time
    t_fill, t_fill_reason = _resolve_fill_time(inp, measured)

    # 3. Effective head
    h_avg_m = (
        max(inp.total_height_mm - 0.5 * inp.part_height_mm, inp.total_height_mm * 0.1)
        / 1000.0
    )
    h_eff_m = effective_head(h_avg_m, inp.total_mass_kg) - inp.head_loss_m
    h_eff_m = float(np.clip(h_eff_m, 0.02, 0.60))

    # 4. Choke velocity and total flow
    v_c = math.sqrt(2.0 * G * h_eff_m)  # theoretical Bernoulli velocity
    v_s_design = inp.discharge_coeff * v_c  # actual sprue base velocity with Cd
    Q = inp.total_metal_volume_m3 / max(t_fill, 0.1)

    # 5. Sprue base area from Bernoulli
    As = Q / max(v_s_design, 0.01)

    # 6. Gate total area from standard ingate formula A = Q / v
    Ag = Q / max(v_gate_target, 0.01)

    # 7. Runner area and system from geometry so the area order is physically correct
    if Ag < 0.6 * As:
        # Gate is the smallest section -> pressurized
        system = "basınçlı (pressurized)"
        Ar = max(0.75 * As, 1.1 * Ag)
        if Ar >= As:
            Ar = 0.95 * As
    elif Ag > 1.4 * As:
        # Gate is the largest section -> unpressurized
        if Ag > 2.0 * As:
            system = "basınçsız (unpressurized)"
            Ar = max(2.0 * As, math.sqrt(As * Ag))
            if Ar >= Ag:
                Ar = 0.95 * Ag
        else:
            system = "yarı basınçlı (semi-pressurized)"
            Ar = 2.0 * As  # runner is largest
    else:
        # Gate and sprue are comparable -> semi-pressurized, runner as largest
        system = "yarı basınçlı (semi-pressurized)"
        Ar = 2.0 * As
        if Ar <= max(As, Ag):
            Ar = max(As, Ag) * 1.1

    # Override if the user explicitly selected a system
    if inp.gating_system:
        system = inp.gating_system

    # The geometry-based recommendation is the physically consistent system.
    if not inp.gating_system:
        recommended_system = system

    # 8. Recompute velocities from the chosen areas
    v_s = Q / As
    v_r = Q / Ar
    v_g = Q / Ag

    # 9. Do not allow gate velocity above the absolute material limit
    if v_g > v_gate_max:
        Ag = Q / v_gate_max
        v_g = v_gate_max
        warnings.append(
            f"Hedef meme hızı malzeme üst sınırını ({v_gate_max:.2f} m/s) aştığı için "
            f"A_g büyütüldü; yeni v_g = {v_g:.2f} m/s."
        )
        # Re-evaluate system after Ag changed
        if Ag < 0.6 * As:
            system = "basınçlı (pressurized)"
        elif Ag > 1.4 * As:
            system = "basınçsız (unpressurized)"
        else:
            system = "yarı basınçlı (semi-pressurized)"

    # 10. Number of ingates from flow-path length and safe local velocity
    if inp.n_gates is not None and inp.n_gates > 0:
        n_gates = int(inp.n_gates)
        n_reason = "Kullanıcı meme sayısı."
    else:
        # Distance metal can travel at gate speed during fill
        reach_mm = v_g * t_fill * 1000.0
        if reach_mm <= 0.0:
            n_gates = 1
        else:
            n_gates = max(1, int(math.ceil(inp.max_flow_path_mm / reach_mm)))
        n_gates = int(np.clip(n_gates, 1, max(1, inp.max_gates)))
        n_reason = f"Akış yolu {inp.max_flow_path_mm:.0f} mm / reach {reach_mm:.0f} mm."

    # If a single gate still exceeds the safe local velocity, split while keeping total Ag
    # (A_each = Ag/n, v_each stays the same, but this is a clean practical split).
    # For pressurized / small Ag we allow more gates up to max_gates if needed.
    if n_gates > 1 and v_g > v_gate_target:
        # Splitting does not change total Ag; it only divides the flow geometrically.
        # To truly lower v_g we already adjusted Ag above.  Report the split.
        warnings.append(
            f"Meme hızı hedefi {v_gate_target:.2f} m/s; {n_gates} memeye bölünmesi önerilir."
        )

    Ag_each = Ag / max(n_gates, 1)

    # 11. Detected system from real velocities
    detected_system = _classify_from_velocities(v_s, v_r, v_g)
    if detected_system != system:
        warnings.append(
            f"Önerilen sistem '{system}', hesaplanan hızlara göre '{detected_system}' olarak sınıflandırıldı."
        )

    # 12. Compare measured areas (if any)
    measured_vel: Dict[str, float] = {}
    if measured:
        for key, area_cm2 in measured.items():
            a_m2 = area_cm2 / 1e4
            if a_m2 > 0.0:
                measured_vel[key] = Q / a_m2

        # Check measured gate against design
        if "INGATE" in measured:
            v_meas_g = measured_vel["INGATE"]
            if v_meas_g > v_gate_max:
                warnings.append(
                    f"Ölçülen toplam meme alanı ({measured['INGATE']:.2f} cm²) çok küçük; "
                    f"hız {v_meas_g:.2f} m/s (limit {v_gate_max:.2f} m/s)."
                )

        # Check measured sprue
        if "SPRUE_BASE" in measured:
            v_meas_s = measured_vel["SPRUE_BASE"]
            if not (0.3 * v_s <= v_meas_s <= 1.5 * v_s):
                warnings.append(
                    f"Ölçülen sprue taban alanı ({measured['SPRUE_BASE']:.2f} cm²) "
                    f"Bernoulli tasarımına uymuyor (v={v_meas_s:.2f} m/s, tasarım v={v_s:.2f} m/s)."
                )

        # Check measured runner
        if "RUNNER" in measured:
            v_meas_r = measured_vel["RUNNER"]
            max_r = _max_runner_velocity_m_s(inp.alloy_key)
            if v_meas_r > max_r:
                warnings.append(
                    f"Ölçülen yolluk alanı ({measured['RUNNER']:.2f} cm²) çok küçük; "
                    f"hız {v_meas_r:.2f} m/s (limit {max_r:.2f} m/s)."
                )

    # 13. Reynolds / Froude at each gate
    re, fr = _reynolds_froude(inp.rho_kg_m3, v_g, Ag_each, inp.viscosity_pa_s)
    turbulent = re > 20000.0 or v_g > v_gate_max

    # 14. Fluidity length
    t_fluid = _fluidity_time_s(
        inp.t_pour_c, inp.t_liquidus_c, inp.latent_heat_j_kg, inp.cp_j_kgk
    )
    fluidity_length_mm = v_g * t_fluid * 1000.0
    if fluidity_length_mm < inp.max_flow_path_mm:
        warnings.append(
            f"Akışkanlık uzunluğu ({fluidity_length_mm:.0f} mm) en uzak noktaya "
            f"({inp.max_flow_path_mm:.0f} mm) yetmiyor; meme sayısı / hız artırılmalı."
        )

    # Final ratio and reason
    ratio = (1.0, Ar / As, Ag / As)
    choke_section = _choke_section_for_system(detected_system)

    reason = (
        f"Önerilen sistem: {recommended_system}, geometriden tespit: {detected_system}. "
        f"{t_fill_reason} {n_reason} "
        f"Oran As:Ar:Ag ≈ {ratio[0]:.2f}:{ratio[1]:.2f}:{ratio[2]:.2f}. "
        f"H_eff={h_eff_m*1000:.1f} mm, v_c={v_c:.2f} m/s."
    )

    # Sprue throat (design) is taken as the base area; a separate measured throat
    # is kept for comparison only.
    sprue_throat_design = As

    return GatingDesign(
        gating_system=detected_system,
        recommended_gating_system=recommended_system,
        choke_section=choke_section,
        n_gates=n_gates,
        t_fill_s=t_fill,
        q_m3_s=Q,
        h_eff_mm=h_eff_m * 1000.0,
        v_choke_m_s=v_c,
        sprue_base_area_cm2=As * 1e4,
        sprue_throat_area_cm2=sprue_throat_design * 1e4,
        runner_total_area_cm2=Ar * 1e4,
        gate_total_area_cm2=Ag * 1e4,
        gate_each_area_cm2=Ag_each * 1e4,
        sprue_velocity_m_s=v_s,
        runner_velocity_m_s=v_r,
        gate_velocity_m_s=v_g,
        reynolds=re,
        froude=fr,
        turbulent=turbulent,
        measured_sprue_base_area_cm2=measured.get("SPRUE_BASE", 0.0),
        measured_sprue_throat_area_cm2=measured.get("SPRUE_THROAT", 0.0),
        measured_runner_total_area_cm2=measured.get("RUNNER", 0.0),
        measured_gate_total_area_cm2=measured.get("INGATE", 0.0),
        measured_sprue_velocity_m_s=measured_vel.get("SPRUE_BASE", 0.0),
        measured_runner_velocity_m_s=measured_vel.get("RUNNER", 0.0),
        measured_gate_velocity_m_s=measured_vel.get("INGATE", 0.0),
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
    part_mass_kg = kwargs.pop("part_mass_kg", total_mass_kg)
    inp = GatingEngineInput(
        total_metal_volume_m3=total_metal_volume_m3,
        total_mass_kg=total_mass_kg,
        part_mass_kg=part_mass_kg,
        part_height_mm=part_height_mm,
        total_height_mm=total_height_mm,
        max_flow_path_mm=max_flow_path_mm,
        wall_thickness_mm=wall_thickness_mm,
        alloy_key=alloy_key,
        alloy_name=alloy_name,
        t_fill_s=t_fill_s,
        user_gate_velocity_m_s=user_gate_velocity_m_s,
        measured_areas_cm2=measured_areas_cm2 or {},
        **kwargs,
    )
    return calculate_gating_design(inp)
