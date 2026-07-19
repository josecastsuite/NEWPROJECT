"""Geometry-aware gating design engine for JoseCast Analyzer.

This module wraps the exact working equations from the user's field scripts:
  * gating_calculator_tr.py  -> compute_gating, auto_fill_time, effective_head
  * Filling_time_tr.py       -> calc_campbell_parameters

The engine produces a physically consistent gating layout and a set of
recommendations / warnings.  When the user supplies measured cross-sectional
areas (runner / ingate / sprue) the engine compares them with the theoretical
design and warns about mismatches instead of blindly overriding physics.
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
    compute_gating,
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
    """Result of a standalone gating design."""

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

    warnings: List[str] = field(default_factory=list)
    reason: str = ""


def _default_ratio(alloy_key: str) -> Tuple[float, float, float]:
    key = alloy_key.lower()
    if "gri" in key or "sfero" in key or "ggg" in key or "nodular" in key or "gray" in key:
        return (1.0, 0.75, 0.5)
    if "al" in key or "alsi" in key:
        return (1.0, 2.0, 2.0)
    if "bronz" in key or "bronze" in key:
        return (1.0, 2.0, 2.0)
    if "steel" in key or "çelik" in key or "42" in key:
        return (1.0, 2.0, 1.0)
    return (1.0, 2.0, 1.0)


def _ratio_for_system(system: str) -> Tuple[float, float, float]:
    s = (system or "").lower()
    if "basınçlı" in s or "pressurized" in s:
        return (1.0, 0.75, 0.5)
    if "basınçsız" in s or "unpressurized" in s:
        return (1.0, 2.0, 2.0)
    return (1.0, 2.0, 1.0)


def _safe_gate_velocity_m_s(alloy_key: str) -> float:
    """Target (laminar / Campbell Rule 1) gate velocity."""
    key = alloy_key.lower()
    if "gri" in key or "sfero" in key or "ggg" in key or "nodular" in key or "gray" in key:
        return 2.5
    if "al" in key or "alsi" in key:
        return 0.4
    if "bronz" in key or "bronze" in key:
        return 0.8
    return 0.5


def _hard_gate_velocity_m_s(alloy_key: str) -> float:
    """Absolute maximum gate velocity before sand erosion / turbulence warnings."""
    key = alloy_key.lower()
    if "gri" in key or "sfero" in key or "ggg" in key or "nodular" in key or "gray" in key:
        return 4.5
    if "al" in key or "alsi" in key:
        return 0.6
    if "bronz" in key or "bronze" in key:
        return 1.2
    return 0.85


def _classify_from_velocities(v_s: float, v_r: float, v_g: float) -> str:
    if v_s <= v_r <= v_g:
        return "basınçlı (pressurized)"
    if v_s >= v_r >= v_g:
        return "basınçsız (unpressurized)"
    if v_r <= v_s and v_r <= v_g:
        return "yarı basınçlı (semi-pressurized)"
    return "yarı basınçlı (semi-pressurized)"


def _choke_section_for_system(system: str) -> str:
    if system == "basınçsız (unpressurized)":
        return "SPRUE_BASE"
    if system == "basınçlı (pressurized)":
        return "INGATE"
    return "RUNNER"


def _hydraulic_diameter_mm(area_m2: float) -> float:
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
    superheat = max(t_pour_c - t_liquidus_c, 0.0)
    if superheat <= 0.0:
        return 0.5
    rho_ref = 7000.0
    energy_j_m3 = rho_ref * (latent_heat_j_kg + cp_j_kgk * superheat)
    q = 1.0e6
    layer_m = 0.001
    t_s = layer_m * energy_j_m3 / q
    return float(np.clip(t_s, 0.2, 20.0))


def _resolve_fill_time(
    inp: GatingEngineInput,
    measured: Dict[str, float],
    has_measured: Dict[str, bool],
) -> Tuple[float, str]:
    """Choose a fill time from user, measured velocity, Campbell or practical table."""
    reason = ""

    if inp.t_fill_s is not None and inp.t_fill_s > 0.0:
        return float(np.clip(inp.t_fill_s, 0.2, 120.0)), "Kullanıcı t_fill."

    if (
        inp.user_gate_velocity_m_s is not None
        and inp.user_gate_velocity_m_s > 0.0
        and has_measured.get(inp.user_velocity_section_key.upper(), False)
    ):
        section = inp.user_velocity_section_key.upper()
        a_m2 = 0.0
        if section == "INGATE":
            a_m2 = measured["INGATE"] / 1e4
        elif section == "RUNNER":
            a_m2 = measured["RUNNER"] / 1e4
        elif section in ("SPRUE_BASE", "SPRUE"):
            a_m2 = measured["SPRUE_BASE"] / 1e4
        elif section == "SPRUE_THROAT":
            a_m2 = measured["SPRUE_THROAT"] / 1e4
        if a_m2 > 0.0:
            t_fill = inp.total_metal_volume_m3 / (inp.user_gate_velocity_m_s * a_m2)
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
        # Use the more conservative (longer) of the two estimates.
        t_fill = max(t_campbell, t_auto)
        reason = f"Campbell ({t_campbell:.2f} s) ve pratik ({t_auto:.2f} s) arasından güvenli seçim."
    else:
        t_fill = t_auto
        reason = f"Pratik dolum süresi: {t_auto:.2f} s."

    return float(np.clip(t_fill, 0.2, 120.0)), reason


def _resolve_gating_ratio(
    inp: GatingEngineInput, system: Optional[str]
) -> Tuple[float, float, float]:
    if inp.gating_ratio is not None:
        return inp.gating_ratio
    if system:
        return _ratio_for_system(system)
    return _default_ratio(inp.alloy_key)


def calculate_gating_design(inp: GatingEngineInput) -> GatingDesign:
    warnings: List[str] = []

    measured = {
        k.upper(): float(v) for k, v in inp.measured_areas_cm2.items() if v > 0.0
    }
    has_measured = {
        "SPRUE_BASE": "SPRUE_BASE" in measured,
        "SPRUE_THROAT": "SPRUE_THROAT" in measured,
        "RUNNER": "RUNNER" in measured,
        "INGATE": "INGATE" in measured,
    }

    # 1. Gating system / ratio from material or user override
    system = inp.gating_system
    if system:
        reason = "Kullanıcı tarafından seçilen sistem."
    else:
        system = "yarı basınçlı (semi-pressurized)"
        reason = "Malzeme ve oranlara göre önerilen sistem."

    ratio = _resolve_gating_ratio(inp, inp.gating_system)

    # 2. Fill time
    t_fill, t_fill_reason = _resolve_fill_time(inp, measured, has_measured)

    # 3. Effective ferrostatic head
    h_avg_m = (
        max(inp.total_height_mm - 0.5 * inp.part_height_mm, inp.total_height_mm * 0.1)
        / 1000.0
    )
    h_eff_m = effective_head(h_avg_m, inp.total_mass_kg) - inp.head_loss_m
    h_eff_m = float(np.clip(h_eff_m, 0.02, 0.60))

    # 4. Base design from gating_calculator_tr.py
    n_gates = max(1, inp.n_gates or 1)
    base = compute_gating(
        W_kg=inp.total_mass_kg,
        rho_kgm3=inp.rho_kg_m3,
        H_m=h_eff_m,
        t_fill_s=t_fill,
        Cd=inp.discharge_coeff,
        gating_ratio=ratio,
        n_ingates=n_gates,
    )

    As_m2 = base["As_m2"]
    Ar_total_m2 = base["Ar_total_m2"]
    Ag_total_m2 = base["Ag_total_m2"]
    Ag_each_m2 = base["Ag_each_m2"]
    Vc_ms = base["Vc_ms"]

    # 5. Apply measured areas carefully
    if has_measured["INGATE"]:
        Ag_total_m2 = measured["INGATE"] / 1e4
        # Each measured area is the TOTAL ingate area; per-gate will be split later.
    if has_measured["RUNNER"]:
        Ar_total_m2 = measured["RUNNER"] / 1e4
    if has_measured["SPRUE_BASE"]:
        # Use measured sprue only if it is physically plausible.
        measured_As = measured["SPRUE_BASE"] / 1e4
        v_s_measured = (inp.total_mass_kg / (inp.rho_kg_m3 * t_fill)) / measured_As if measured_As > 0.0 else 0.0
        if 0.3 * Vc_ms <= v_s_measured <= 1.5 * Vc_ms:
            As_m2 = measured_As
        else:
            warnings.append(
                f"Ölçülen sprue taban alanı ({measured['SPRUE_BASE']:.2f} cm²) "
                f"Bernoulli hızına uymuyor; tasarım alanı ({As_m2*1e4:.2f} cm²) kullanıldı."
            )

    sprue_throat_m2 = measured.get("SPRUE_THROAT", As_m2 * 1e4) / 1e4
    if not has_measured["SPRUE_THROAT"]:
        sprue_throat_m2 = As_m2

    Q = inp.total_mass_kg / (inp.rho_kg_m3 * t_fill)

    # 6. Actual velocities from continuity Q = A·v
    v_s = Q / As_m2 if As_m2 > 0.0 else Vc_ms
    v_r = Q / Ar_total_m2 if Ar_total_m2 > 0.0 else 0.0
    v_g = Q / Ag_total_m2 if Ag_total_m2 > 0.0 else 0.0

    # 7. Smart ingate number / safe local gate velocity
    v_gate_target = _safe_gate_velocity_m_s(inp.alloy_key)
    v_gate_max = _hard_gate_velocity_m_s(inp.alloy_key)

    if v_g > v_gate_target:
        # To reach target velocity while keeping total flow Q,
        # total gate area must become Q / v_target.
        n_gates = max(n_gates, int(math.ceil(v_g / v_gate_target)))
        n_gates = int(np.clip(n_gates, 1, max(1, inp.max_gates)))
        Ag_total_m2 = Q / v_gate_target
        v_g = v_gate_target
        warnings.append(
            f"Tek memede hız {v_g:.2f} m/s (hedef ≤ {v_gate_target:.2f} m/s) olduğu için "
            f"{n_gates} adet meme önerildi."
        )
    elif v_g < v_gate_target * 0.2:
        warnings.append(
            f"Meme hızı çok düşük ({v_g:.2f} m/s); soğuk birleşme (cold shut) riski."
        )

    Ag_each_m2 = Ag_total_m2 / max(n_gates, 1)

    # 8. Recompute runner / sprue velocities after possible gate changes
    v_r = Q / Ar_total_m2 if Ar_total_m2 > 0.0 else 0.0
    v_s = Q / As_m2 if As_m2 > 0.0 else Vc_ms

    # 9. Gating system classification from actual velocities
    detected_system = _classify_from_velocities(v_s, v_r, v_g)
    choke_section = _choke_section_for_system(detected_system)

    if detected_system != system:
        warnings.append(
            f"Önerilen sistem '{system}', hesaplanan hızlara göre '{detected_system}' olarak sınıflandırıldı."
        )

    # 10. Section-specific warnings
    if has_measured["SPRUE_BASE"]:
        if v_s < 0.3 * Vc_ms:
            warnings.append(
                "Ölçülen / kullanılan sprue taban alanı büyük; sprue hızı düşük, hava emilimi (aspiration) riski."
            )
        elif v_s > 1.5 * Vc_ms:
            warnings.append(
                "Ölçülen / kullanılan sprue taban alanı küçük; türbülans ve kum aşınması riski."
            )

    if v_r > 2.0:
        warnings.append(f"Runner hızı yüksek ({v_r:.2f} m/s); aşınma ve türbülans riski.")
    elif v_r > 0.0 and v_r < 0.2:
        warnings.append(f"Runner hızı çok düşük ({v_r:.2f} m/s); soğuk birleşme riski.")

    if v_g > v_gate_max:
        warnings.append(
            f"Meme hızı üst sınırı ({v_gate_max:.2f} m/s) aşıyor: {v_g:.2f} m/s. "
            "Meme sayısını veya toplam alanı artır."
        )

    # 11. Reynolds / Froude at the gate
    re, fr = _reynolds_froude(
        inp.rho_kg_m3, v_g, Ag_each_m2, inp.viscosity_pa_s
    )
    turbulent = re > 20000.0 or v_g > v_gate_max

    # 12. Fluidity length
    t_fluid = _fluidity_time_s(
        inp.t_pour_c, inp.t_liquidus_c, inp.latent_heat_j_kg, inp.cp_j_kgk
    )
    fluidity_length_mm = v_g * t_fluid * 1000.0
    if fluidity_length_mm < inp.max_flow_path_mm:
        warnings.append(
            f"Akışkanlık uzunluğu ({fluidity_length_mm:.0f} mm) en uzak noktaya "
            f"({inp.max_flow_path_mm:.0f} mm) yetmiyor; meme sayısı / hız artırılmalı."
        )

    full_reason = f"{reason} {t_fill_reason} Oran As:Ar:Ag = {ratio[0]}:{ratio[1]}:{ratio[2]}."

    return GatingDesign(
        gating_system=detected_system,
        recommended_gating_system=system,
        choke_section=choke_section,
        n_gates=n_gates,
        t_fill_s=t_fill,
        q_m3_s=Q,
        h_eff_mm=h_eff_m * 1000.0,
        v_choke_m_s=Vc_ms,
        sprue_base_area_cm2=As_m2 * 1e4,
        sprue_throat_area_cm2=sprue_throat_m2 * 1e4,
        runner_total_area_cm2=Ar_total_m2 * 1e4,
        gate_total_area_cm2=Ag_total_m2 * 1e4,
        gate_each_area_cm2=Ag_each_m2 * 1e4,
        sprue_velocity_m_s=v_s,
        runner_velocity_m_s=v_r,
        gate_velocity_m_s=v_g,
        reynolds=re,
        froude=fr,
        turbulent=turbulent,
        warnings=warnings,
        reason=full_reason,
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


# Kept for backwards compatibility with callers that want UI target ranges.
# The design itself does NOT use these tables; it uses the field formulas above.
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
