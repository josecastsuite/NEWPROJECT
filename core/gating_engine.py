"""Geometry-aware gating design engine for JoseCast Analyzer.

The engine is built around the principle: once the flow rate `Q` and fill time
`t_fill` are fixed, every section area is `A = Q / v`.  The selected section
(user velocity / measured area or the choke section) anchors the design; the
remaining sections are derived from a geometry-aware gating-system ratio.

Design sequence (Campbell / field-script compatible):
  1. geometry + material -> recommended gating system and target gate speed
  2. fill time / flow rate -> user velocity+measured area, or Campbell/practical
  3. effective head       -> H_eff with mass-dependent head reduction
  4. anchor section area  -> A_anchor = Q / v_anchor
  5. derive As, Ar, Ag    -> from the geometry-aware system ratio
  6. enforce velocity limits and physical ordering
  7. gate count           -> from flow-path length and local gate velocity
  8. measured comparison  -> Q/A for each measured area, warnings if far off

Key property: when a user gives a velocity for a measured section, the engine
produces exactly that velocity for that section and builds the rest around it
with simple proportion.  No hidden double bookkeeping.
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
from core.materials import get_alloy


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

    # v9.2: extended geometry features for intelligent system/ratio recommendation
    wall_thickness_min_mm: float = 0.0
    wall_thickness_max_mm: float = 0.0
    surface_to_volume_ratio_1_mm: float = 0.0
    hotspot_count: int = 0
    max_hotspot_m_mm: float = 0.0
    pore_risk_max: float = 0.0

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


def _section_velocity_limit(system: str, alloy_key: str, section: str) -> float:
    """Material + system aware velocity ceiling for a gating section."""
    sys_max = _VELOCITY_RANGES.get(system, _VELOCITY_RANGES["yarı basınçlı (semi-pressurized)"]).get(
        section, (0.0, 2.0)
    )[1]
    if section == "gate":
        return min(sys_max, _max_gate_velocity_m_s(alloy_key))
    if section == "runner":
        return min(sys_max, _max_runner_velocity_m_s(alloy_key))
    return sys_max


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _lerp(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    """Linear ramp from (x0,y0) to (x1,y1); clamped outside.

    Works for either ascending or descending x range.
    """
    if x0 == x1:
        return (y0 + y1) / 2.0
    t = (x - x0) / (x1 - x0)
    t = max(0.0, min(1.0, t))
    return y0 + t * (y1 - y0)


def _score_systems(inp: GatingEngineInput, features: Dict[str, float]) -> Tuple[str, Dict[str, float], str]:
    """Score each gating system 0-100 based on geometry/material features.

    Returns (recommended_system, scores_dict, reason_string).
    """
    kind = _alloy_kind(inp.alloy_key)
    t_avg = features["t_avg"]
    t_min = features["t_min"]
    t_max = features["t_max"]
    sv = features["surface_to_volume_ratio"]
    slenderness = features["slenderness"]
    flow_ratio = features["flow_ratio"]
    head_ratio = features["head_ratio"]
    hotspots = features["hotspot_count"]
    max_hot_m = features["max_hotspot_m_mm"]
    pore_risk = features["pore_risk_max"]
    thickness_var = t_max / max(t_min, 1.0)

    # --- Unpressurized score ---
    u = 0.0
    u += _lerp(t_avg, 30.0, 0.0, 8.0, 20.0)
    u += _lerp(t_min, 15.0, 0.0, 4.0, 15.0)
    u += _lerp(slenderness, 1.0, 0.0, 5.0, 22.0)
    u += _lerp(flow_ratio, 0.0, 0.0, 0.8, 25.0)
    if flow_ratio > 1.0:
        u -= 50.0 * _lerp(flow_ratio, 1.0, 0.0, 2.0, 1.0)
    u += _lerp(head_ratio, 0.6, 0.0, 0.1, 12.0)
    u += _clamp(25.0 * (sv - 0.05), 0.0, 12.0)
    if kind == "aluminum":
        u += 15.0
    elif kind == "bronze":
        u += 5.0
    elif kind == "gray":
        u -= 5.0
    u += _clamp(3.0 * hotspots, 0.0, 12.0)
    u += _clamp(12.0 * pore_risk, 0.0, 10.0)
    u = _clamp(u, 0.0, 100.0)

    # --- Pressurized score ---
    p = 10.0
    p += _lerp(t_avg, 10.0, 0.0, 30.0, 25.0)
    p += _lerp(t_min, 5.0, 0.0, 15.0, 12.0)
    p += _lerp(slenderness, 4.0, 0.0, 1.0, 25.0)
    p += _lerp(head_ratio, 0.05, 0.0, 0.6, 15.0)
    # flow ratio: pressurized is favoured when the flow path is a modest
    # fraction of fluidity length (compact parts); penalty when it is very long.
    p += _lerp(abs(flow_ratio - 0.3), 0.0, 10.0, 0.8, 0.0)
    if flow_ratio > 0.9:
        p += 8.0 * _lerp(flow_ratio, 1.0, 0.0, 2.0, 1.0)
    p += _clamp(0.5 * max_hot_m, 0.0, 10.0)
    p += _clamp(2.0 * max(0, 4 - hotspots), 0.0, 8.0)
    if kind == "steel":
        p += 5.0
    elif kind == "gray":
        p += 18.0
    elif kind == "bronze":
        p += 2.0
    elif kind == "aluminum":
        p -= 15.0
    p = _clamp(p, 0.0, 100.0)

    # --- Semi-pressurized score ---
    s = 15.0
    s += _lerp(abs(slenderness - 2.0), 3.0, 0.0, 0.0, 15.0)
    s += _lerp(thickness_var, 1.0, 0.0, 5.0, 22.0)
    s += _lerp(abs(flow_ratio - 0.6), 1.0, 0.0, 0.0, 18.0)
    s += _lerp(abs(head_ratio - 0.3), 0.6, 0.0, 0.0, 10.0)
    s += _clamp(2.0 * hotspots, 0.0, 10.0)
    s += _clamp(18.0 * min(pore_risk, 1.0 - pore_risk), 0.0, 10.0)
    if kind == "bronze":
        s += 5.0
    elif kind == "gray":
        s += 5.0
    elif kind == "aluminum":
        s += 3.0
    s = _clamp(s, 0.0, 100.0)

    # Material multipliers (final fine-tuning)
    mult = {
        "basınçlı (pressurized)": {"steel": 1.10, "gray": 1.25, "bronze": 0.95, "aluminum": 0.50},
        "basınçsız (unpressurized)": {"steel": 0.95, "gray": 0.85, "bronze": 1.00, "aluminum": 1.25},
        "yarı basınçlı (semi-pressurized)": {"steel": 1.05, "gray": 1.15, "bronze": 1.25, "aluminum": 1.00},
    }
    labels = {
        "basınçlı (pressurized)": "P",
        "basınçsız (unpressurized)": "U",
        "yarı basınçlı (semi-pressurized)": "S",
    }
    scores: Dict[str, float] = {}
    for sys, base in [("basınçlı (pressurized)", p),
                      ("basınçsız (unpressurized)", u),
                      ("yarı basınçlı (semi-pressurized)", s)]:
        m = mult[sys].get(kind, 1.0)
        scores[sys] = _clamp(base * m, 0.0, 100.0)

    # Hard overrides for physically impossible combinations
    if flow_ratio > 1.2:
        scores["basınçsız (unpressurized)"] = _clamp(
            scores["basınçsız (unpressurized)"] - 60.0, 0.0, 100.0
        )
    if flow_ratio > 0.9:
        scores["basınçlı (pressurized)"] += 8.0
        scores["yarı basınçlı (semi-pressurized)"] += 5.0

    # Decide.  If the winner is close to semi, prefer the safer semi choice.
    top = max(scores, key=scores.get)
    semi = "yarı basınçlı (semi-pressurized)"
    if top == "basınçsız (unpressurized)" and flow_ratio > 1.0 and scores[semi] >= scores[top] - 15.0:
        top = semi

    reason = (
        f"Puanlar: U={scores['basınçsız (unpressurized)']:.0f}, "
        f"P={scores['basınçlı (pressurized)']:.0f}, S={scores[semi]:.0f} "
        f"-> {labels[top]}. "
        f"L_flow/LF={flow_ratio:.2f}, slenderness={slenderness:.2f}, "
        f"t_avg={t_avg:.1f} mm, t_var={thickness_var:.2f}."
    )
    return top, scores, reason


def _recommend_system(
    inp: GatingEngineInput,
    features: Dict[str, float],
    v_g_est: float,
    t_fluid_s: float,
) -> Tuple[str, Dict[str, float], str]:
    """Pick the best gating system from geometry/material features."""
    if inp.gating_system:
        # User override: still compute scores for reporting.
        _, scores, _ = _score_systems(inp, features)
        return inp.gating_system, scores, f"Kullanıcı sistem: {inp.gating_system}."
    return _score_systems(inp, features)


def _classify_from_velocities(
    v_s: float,
    v_r: float,
    v_g: float,
    v_distributor: Optional[float] = None,
    v_curufluk: Optional[float] = None,
) -> str:
    """Classify the real system from actual velocities (Q/A).

    Pressurized  -> area shrinks: sprue slowest, gate fastest (v_s < v_r < v_g).
    Unpressurized -> area grows:   sprue fastest, gate slowest (v_s > v_r > v_g).
    When a distributor and/or curufluk is present they sit between runner and
    gate, so the ordering must be monotonic through that chain.
    """
    # Build the measured section velocity list in flow order.
    velocities = [v_s, v_r]
    if v_distributor is not None and v_distributor > 0.0:
        velocities.append(v_distributor)
    if v_curufluk is not None and v_curufluk > 0.0:
        velocities.append(v_curufluk)
    velocities.append(v_g)

    # Pressurized: velocities strictly increase downstream (areas shrink).
    if all(velocities[i] <= velocities[i + 1] for i in range(len(velocities) - 1)):
        return "basınçlı (pressurized)"
    # Unpressurized: velocities strictly decrease downstream (areas grow).
    if all(velocities[i] >= velocities[i + 1] for i in range(len(velocities) - 1)):
        return "basınçsız (unpressurized)"
    return "yarı basınçlı (semi-pressurized)"


def _choke_section_for_system(system: str) -> str:
    if system == "basınçsız (unpressurized)":
        return "SPRUE_BASE"
    if system == "basınçlı (pressurized)":
        return "INGATE"
    return "RUNNER"


def _derive_system_ratio(
    system: str,
    features: Dict[str, float],
    alloy_key: str,
) -> Tuple[float, float, float]:
    """Return a geometry-aware (As, Ar, Ag) ratio with As = 1."""
    flow_ratio = features["flow_ratio"]
    slenderness = features["slenderness"]
    t_avg = features["t_avg"]
    t_min = features["t_min"]
    t_max = features["t_max"]
    sv = features["surface_to_volume_ratio"]
    thickness_var = t_max / max(t_min, 1.0)

    if "basınçlı" in system and "yarı" not in system:
        r_r_base = 0.75
        r_g_base = 0.5
    elif "basınçsız" in system:
        r_r_base = 2.5
        r_g_base = 2.5
    else:
        r_r_base = 1.6
        r_g_base = 0.8

    # Flow-path factor: longer relative to fluidity -> enlarge runner/gate area
    if "basınçlı" in system and "yarı" not in system:
        k_flow = _clamp(1.0 + 0.2 * max(0.0, flow_ratio - 0.3), 0.9, 1.5)
    elif "basınçsız" in system:
        k_flow = _clamp(1.0 + 0.5 * max(0.0, flow_ratio - 0.3), 1.0, 2.2)
    else:
        k_flow = _clamp(1.0 + 0.4 * max(0.0, flow_ratio - 0.4), 0.9, 1.8)

    # Slenderness factor for flat/long parts
    k_slender = _clamp(1.0 + 0.15 * max(0.0, slenderness - 2.0), 1.0, 1.5)

    # Wall thickness factor: thick sections can tolerate a smaller ratio spread
    k_wall = _clamp(1.0 + 0.25 * math.log(max(t_avg, 1.0) / 10.0), 0.7, 1.5)

    # Surface-to-volume factor: thin/flat parts need more generous runner/gate
    k_surface = _clamp(1.0 + 0.3 * max(0.0, sv - 0.10), 1.0, 1.4)

    # Thickness variation factor: non-uniform parts need more gate area
    k_gate = _clamp(1.0 + 0.15 * (thickness_var - 1.0), 0.9, 1.6)

    r_r = r_r_base * k_flow * k_slender * k_wall * k_surface
    r_g = r_g_base * k_flow * k_slender * k_wall * k_surface * k_gate

    # Clamp into physically valid regions for the chosen system
    if "basınçlı" in system and "yarı" not in system:
        r_r = _clamp(r_r, 0.4, 1.0)
        r_g = _clamp(r_g, 0.3, min(r_r, 0.9))
    elif "basınçsız" in system:
        r_r = _clamp(r_r, 1.0, 4.0)
        r_g = _clamp(r_g, max(r_r, 1.0), 4.0)
    else:
        # semi: runner wider than sprue, gate narrower than sprue
        r_r = _clamp(r_r, 1.1, 3.0)
        r_g = _clamp(r_g, 0.4, 0.95)

    return (1.0, float(r_r), float(r_g))


def _derive_design_anchor(
    system: str,
    v_s_bernoulli: float,
    v_gate_target: float,
    v_gate_max: float,
) -> Tuple[str, float]:
    """Return (anchor_section, anchor_velocity_m_s) for the chosen system."""
    if "basınçlı" in system and "yarı" not in system:
        return "INGATE", min(v_gate_target, v_gate_max)
    if "basınçsız" in system:
        # Sprue is the choke, but cap Bernoulli speed by the material/system limit.
        v_s_max = _section_velocity_limit(system, "", "sprue")
        return "SPRUE_BASE", min(v_s_bernoulli, v_s_max)
    # Semi-pressurized: choke is runner, target velocity between sprue and gate.
    v_anchor = 0.5 * (v_s_bernoulli + v_gate_target)
    v_r_max = _section_velocity_limit(system, "", "runner")
    return "RUNNER", _clamp(v_anchor, 0.3, v_r_max)


def _derive_areas_from_anchor(
    anchor_key: str,
    A_anchor_m2: float,
    ratio: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Given one area (anchor), return (As, Ar, Ag) using the system ratio.

    Ratio is (As, Ar, Ag) with As = 1.0.
    """
    r_s, r_r, r_g = ratio
    r_s = max(r_s, 1e-6)
    r_r = max(r_r, 1e-6)
    r_g = max(r_g, 1e-6)

    if anchor_key in ("INGATE",):
        Ag = A_anchor_m2
        As = Ag / r_g * r_s
        Ar = As / r_s * r_r
    elif anchor_key in ("RUNNER",):
        Ar = A_anchor_m2
        As = Ar / r_r * r_s
        Ag = As / r_s * r_g
    elif anchor_key in ("SPRUE_BASE", "SPRUE_THROAT"):
        As = A_anchor_m2
        Ar = As / r_s * r_r
        Ag = As / r_s * r_g
    else:
        As = A_anchor_m2
        Ar = As / r_s * r_r
        Ag = As / r_s * r_g

    return max(As, 1e-9), max(Ar, 1e-9), max(Ag, 1e-9)


def _enforce_velocity_limits(
    system: str,
    As: float,
    Ar: float,
    Ag: float,
    Q: float,
    inp: GatingEngineInput,
) -> Tuple[float, float, float]:
    """Adjust areas so all section velocities respect material/system limits
    and the velocity ordering matches the chosen gating system.
    """
    v_s_lim = _section_velocity_limit(system, inp.alloy_key, "sprue")
    v_r_lim = _section_velocity_limit(system, inp.alloy_key, "runner")
    v_g_lim = _section_velocity_limit(system, inp.alloy_key, "gate")

    def _v(area: float) -> float:
        return Q / max(area, 1e-9)

    v_s = _v(As)
    v_r = _v(Ar)
    v_g = _v(Ag)

    if "basınçlı" in system and "yarı" not in system:
        # v_s < v_r < v_g; keep all below their limits with small separation.
        v_g2 = min(v_g, v_g_lim)
        v_r2 = min(v_r, v_r_lim, v_g2 * 0.95)
        if v_r2 < 0.05:
            v_r2 = v_g2
        v_s2 = min(v_s, v_s_lim, v_r2 * 0.95)
        if v_s2 < 0.05:
            v_s2 = v_r2
        As = Q / v_s2
        Ar = Q / v_r2
        Ag = Q / v_g2
    elif "basınçsız" in system:
        # v_s > v_r > v_g; keep all below their limits with small separation.
        v_g2 = min(v_g, v_g_lim)
        v_r2 = min(v_r_lim, max(min(v_r, v_r_lim), v_g2 * 1.05))
        v_s2 = min(v_s_lim, max(min(v_s, v_s_lim), v_r2 * 1.05))
        if v_s2 < 0.05:
            v_s2 = v_r2
        As = Q / v_s2
        Ar = Q / v_r2
        Ag = Q / v_g2
    else:
        # semi: v_r < v_s < v_g
        v_g2 = min(v_g, v_g_lim)
        v_s2 = min(v_s, v_s_lim, v_g2 * 0.95)
        v_r2 = min(v_r, v_r_lim, v_s2 * 0.95)
        if v_s2 < 0.05:
            v_s2 = v_g2
        if v_r2 < 0.05:
            v_r2 = v_s2
        As = Q / v_s2
        Ar = Q / v_r2
        Ag = Q / v_g2

    return max(As, 1e-9), max(Ar, 1e-9), max(Ag, 1e-9)


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


def _auto_fill_time(inp: GatingEngineInput) -> Tuple[float, str]:
    """Fill time ignoring any user velocity; Campbell + practical table."""
    if inp.t_fill_s is not None and inp.t_fill_s > 0.0:
        return float(np.clip(inp.t_fill_s, 0.2, 120.0)), "Kullanıcı t_fill."

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


def _compute_n_gates(
    inp: GatingEngineInput,
    v_g: float,
    t_fluid_s: float,
    features: Dict[str, float],
    system: str,
) -> Tuple[int, str]:
    """Number of ingates from flow-path length, fluidity and part complexity."""
    if inp.n_gates is not None and inp.n_gates > 0:
        return int(inp.n_gates), "Kullanıcı meme sayısı."

    max_path = max(inp.max_flow_path_mm, 1.0)
    LF_mm = max(v_g * t_fluid_s * 1000.0, 1.0)
    flow_ratio = max_path / LF_mm

    n_flow = int(math.ceil(max_path / (0.7 * LF_mm)))
    n_hot = max(1, (inp.hotspot_count + 1) // 2)
    n_geom = 2 if features["thickness_var"] > 3.0 else 1

    # Feeding-distance based spacing: gates should be close enough that the
    # largest known hot spot can be fed from one of them. FD = feed_k1 * 2 * M_hot.
    n_feed = 1
    feed_distance_mm = 0.0
    if inp.max_hotspot_m_mm > 0.0:
        alloy = get_alloy(inp.alloy_key)
        feed_distance_mm = alloy.feed_k1 * 2.0 * inp.max_hotspot_m_mm
        n_feed = int(math.ceil(max_path / max(0.7 * feed_distance_mm, 1.0)))

    n_gates = max(n_flow, n_hot, n_geom, n_feed)

    # Long unpressurized paths need more gates to avoid premature freezing
    if "basınçsız" in system and flow_ratio > 0.8:
        n_gates = max(n_gates, int(math.ceil(max_path / (0.55 * LF_mm))))

    n_gates = int(np.clip(n_gates, 1, max(1, inp.max_gates)))
    fd_str = f"besleme mesafesi {feed_distance_mm:.0f} mm, " if feed_distance_mm > 0.0 else ""
    reason = (
        f"Akış yolu {max_path:.0f} mm / akışkanlık {LF_mm:.0f} mm, "
        f"{fd_str}"
        f"{inp.hotspot_count} hot-spot, kalınlık varyasyonu={features['thickness_var']:.2f}."
    )
    return n_gates, reason


def calculate_gating_design(inp: GatingEngineInput) -> GatingDesign:
    warnings: List[str] = []

    # Normalise measured areas
    measured: Dict[str, float] = {
        k.upper(): float(v) for k, v in (inp.measured_areas_cm2 or {}).items() if v > 0.0
    }

    # 1. Geometry / material features
    wall_cat = _wall_class(inp.wall_thickness_mm)
    t_avg = max(inp.wall_thickness_mm, 1.0)
    t_min = max(inp.wall_thickness_min_mm, 1.0) if inp.wall_thickness_min_mm > 0.0 else t_avg * 0.5
    t_max = max(inp.wall_thickness_max_mm, 1.0) if inp.wall_thickness_max_mm > 0.0 else t_avg * 1.2

    V_mm3 = inp.part_volume_m3 * 1e9
    if V_mm3 > 0.0 and inp.surface_to_volume_ratio_1_mm > 0.0:
        A_mm2 = V_mm3 * inp.surface_to_volume_ratio_1_mm
        D_bulk = 6.0 * V_mm3 / A_mm2
    else:
        A_mm2 = 0.0
        D_bulk = 2.0 * max(inp.max_flow_path_mm, 1.0)

    slenderness = max(inp.max_flow_path_mm, 1.0) / max(D_bulk, 1.0)
    H_eff_mm = max(inp.total_height_mm - 0.5 * inp.part_height_mm, inp.total_height_mm * 0.1)
    h_eff_m = float(np.clip(max(H_eff_mm / 1000.0 - inp.head_loss_m, 0.02), 0.02, 0.60))

    v_c = math.sqrt(2.0 * G * h_eff_m)
    v_s_bernoulli = inp.discharge_coeff * v_c
    v_gate_target = _safe_gate_velocity_m_s(inp.alloy_key, wall_cat)
    v_gate_max = _max_gate_velocity_m_s(inp.alloy_key)
    t_fluid = _fluidity_time_s(
        inp.t_pour_c, inp.t_liquidus_c, inp.latent_heat_j_kg, inp.cp_j_kgk
    )

    v_g_est = v_gate_target
    LF_mm = max(v_g_est * t_fluid * 1000.0, 1.0)
    max_path = max(inp.max_flow_path_mm, 1.0)
    flow_ratio = max_path / LF_mm
    head_ratio = H_eff_mm / max_path
    sv_ratio = A_mm2 / V_mm3 if V_mm3 > 0.0 else 0.0
    thickness_var = t_max / max(t_min, 1.0)

    features: Dict[str, float] = {
        "t_avg": t_avg,
        "t_min": t_min,
        "t_max": t_max,
        "surface_to_volume_ratio": sv_ratio,
        "slenderness": slenderness,
        "flow_ratio": flow_ratio,
        "head_ratio": head_ratio,
        "hotspot_count": float(inp.hotspot_count),
        "max_hotspot_m_mm": inp.max_hotspot_m_mm,
        "pore_risk_max": inp.pore_risk_max,
        "thickness_var": thickness_var,
    }

    # 2. Resolve fill time and flow rate Q (user anchor may override later)
    user_v = float(inp.user_gate_velocity_m_s or 0.0)
    user_sec = (inp.user_velocity_section_key or "INGATE").upper()

    A_user_measured_cm2 = measured.get(user_sec, 0.0)
    if A_user_measured_cm2 <= 0.0:
        A_user_measured_cm2 = (
            measured.get("INGATE", 0.0)
            or measured.get("RUNNER", 0.0)
            or measured.get("SPRUE_BASE", 0.0)
            or measured.get("SPRUE_THROAT", 0.0)
            or 0.0
        )
    A_user_measured_m2 = A_user_measured_cm2 / 1e4

    if user_v > 0.0 and A_user_measured_m2 > 0.0:
        Q = user_v * A_user_measured_m2
        t_fill = inp.total_metal_volume_m3 / max(Q, 1e-9)
        t_fill = float(np.clip(t_fill, 0.2, 120.0))
        t_fill_reason = (
            f"Kullanıcı {user_sec} hızı ({user_v:.2f} m/s) ve ölçülen "
            f"{A_user_measured_cm2:.2f} cm²'dan Q={Q*1e3:.2f} L/s, t_fill={t_fill:.2f} s."
        )
        anchor_key = user_sec
        A_anchor_m2 = A_user_measured_m2
        v_anchor = user_v
        anchor_from_user = True
        v_g_est = user_v if user_sec == "INGATE" else v_g_est
    elif user_v > 0.0:
        t_fill, t_fill_reason = _auto_fill_time(inp)
        Q = inp.total_metal_volume_m3 / max(t_fill, 0.1)
        A_anchor_m2 = Q / max(user_v, 0.01)
        anchor_key = user_sec
        v_anchor = user_v
        anchor_from_user = True
        warnings.append(
            f"{user_sec} için ölçülen alan olmadığından A={A_anchor_m2*1e4:.2f} cm² "
            f"(Q/v) olarak tasarlandı."
        )
        v_g_est = user_v if user_sec == "INGATE" else v_g_est
    else:
        t_fill, t_fill_reason = _auto_fill_time(inp)
        Q = inp.total_metal_volume_m3 / max(t_fill, 0.1)
        anchor_key = None
        A_anchor_m2 = 0.0
        v_anchor = 0.0
        anchor_from_user = False

    # Recompute fluidity length estimate with the actual gate estimate
    LF_mm = max(v_g_est * t_fluid * 1000.0, 1.0)
    flow_ratio = max_path / LF_mm
    features["flow_ratio"] = flow_ratio

    # 3. Recommend system from geometry/material
    recommended_system, scores, score_reason = _recommend_system(
        inp, features, v_g_est, t_fluid
    )
    system = inp.gating_system or recommended_system

    # 4. Geometry-aware system ratio and anchor section
    ratio = inp.gating_ratio or _derive_system_ratio(system, features, inp.alloy_key)
    if anchor_from_user:
        # keep user anchor
        pass
    else:
        anchor_key, v_anchor = _derive_design_anchor(
            system, v_s_bernoulli, v_gate_target, v_gate_max
        )
        A_anchor_m2 = Q / max(v_anchor, 0.01)

    As, Ar, Ag = _derive_areas_from_anchor(anchor_key, A_anchor_m2, ratio)

    # 5. Enforce material velocity limits and physical ordering
    As, Ar, Ag = _enforce_velocity_limits(system, As, Ar, Ag, Q, inp)
    v_s = Q / max(As, 1e-9)
    v_r = Q / max(Ar, 1e-9)
    v_g = Q / max(Ag, 1e-9)

    # 6. Number of ingates
    n_gates, n_reason = _compute_n_gates(inp, v_g, t_fluid, features, system)
    Ag_each = Ag / max(n_gates, 1)

    # 7. Detected system from real velocities
    detected_system = _classify_from_velocities(v_s, v_r, v_g)
    if detected_system != system:
        warnings.append(
            f"Önerilen sistem '{system}', hesaplanan hızlara göre '{detected_system}' olarak sınıflandırıldı."
        )

    # 8. Compare measured areas (if any)
    measured_vel: Dict[str, float] = {}
    if measured:
        for key, area_cm2 in measured.items():
            a_m2 = area_cm2 / 1e4
            if a_m2 > 0.0:
                measured_vel[key] = Q / a_m2

        if "INGATE" in measured:
            v_meas_g = measured_vel["INGATE"]
            if v_meas_g > v_gate_max:
                warnings.append(
                    f"Ölçülen toplam meme alanı ({measured['INGATE']:.2f} cm²) çok küçük; "
                    f"hız {v_meas_g:.2f} m/s (limit {v_gate_max:.2f} m/s)."
                )

        design_areas = {
            "SPRUE_BASE": (As * 1e4, v_s),
            "SPRUE_THROAT": (As * 1e4, v_s),
            "RUNNER": (Ar * 1e4, v_r),
            "INGATE": (Ag * 1e4, v_g),
        }
        for key, (area_design_cm2, v_design) in design_areas.items():
            if key not in measured:
                continue
            area_meas = measured[key]
            ratio_meas = area_meas / area_design_cm2 if area_design_cm2 > 0.0 else 0.0
            if ratio_meas < 0.6 or ratio_meas > 1.5:
                v_meas = measured_vel[key]
                warnings.append(
                    f"Ölçülen {key} alanı ({area_meas:.2f} cm²) tasarımdan ({area_design_cm2:.2f} cm²) "
                    f"farklı; oran={ratio_meas:.2f}, v_ölçülen={v_meas:.2f} m/s, v_tasarım={v_design:.2f} m/s."
                )

    # 9. Reynolds / Froude at each gate
    re, fr = _reynolds_froude(inp.rho_kg_m3, v_g, Ag_each, inp.viscosity_pa_s)
    turbulent = re > 20000.0 or v_g > v_gate_max

    # 10. Fluidity length check
    fluidity_length_mm = v_g * t_fluid * 1000.0
    if fluidity_length_mm < max_path:
        warnings.append(
            f"Akışkanlık uzunluğu ({fluidity_length_mm:.0f} mm) en uzak noktaya "
            f"({max_path:.0f} mm) yetmiyor; meme sayısı / hız artırılmalı."
        )

    # Final ratio and reason
    ratio_out = (
        1.0,
        Ar / max(As, 1e-9),
        Ag / max(As, 1e-9),
    )
    choke_section = _choke_section_for_system(system)
    v_choke = {"SPRUE_BASE": v_s, "INGATE": v_g, "RUNNER": v_r}.get(choke_section, v_s)

    reason = (
        f"{score_reason} {t_fill_reason} {n_reason} "
        f"Sistem: {system} (önerilen: {recommended_system}). "
        f"Oran As:Ar:Ag ≈ {ratio_out[0]:.2f}:{ratio_out[1]:.2f}:{ratio_out[2]:.2f}. "
        f"H_eff={h_eff_m*1000:.1f} mm, v_choke={v_choke:.2f} m/s."
    )

    return GatingDesign(
        gating_system=system,
        recommended_gating_system=recommended_system,
        choke_section=choke_section,
        n_gates=n_gates,
        t_fill_s=t_fill,
        q_m3_s=Q,
        h_eff_mm=h_eff_m * 1000.0,
        v_choke_m_s=v_choke,
        sprue_base_area_cm2=As * 1e4,
        sprue_throat_area_cm2=As * 1e4,
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
