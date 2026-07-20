"""Shared types and result containers for JoseCast Analyzer v7.2."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh


@dataclass
class CastingParameters:
    """User-editable process and material overrides for a single run."""
    t_pour_c: float = 1600.0
    t_liquidus_c: float = 1510.0
    t_solidus_c: float = 1410.0
    t_mold_c: float = 25.0
    t_fill_s: float = 10.0
    rho_liquid_kg_m3: float = 7000.0
    viscosity_pa_s: float = 0.006
    # v8.1: user-specified inlet velocity (0 = auto from V_part / t_fill)
    ingate_velocity_m_s: float = 0.0
    # v8.3: which gating section the velocity above refers to
    velocity_section_key: str = "INGATE"
    # v8.7: gravity direction for feeding and gating calculations (default -Z)
    gravity_vector: Tuple[float, float, float] = (0.0, 0.0, -1.0)

    @property
    def superheat_c(self) -> float:
        return max(self.t_pour_c - self.t_liquidus_c, 0.0)


class BodyType(IntEnum):
    """Material / role identifier stored in the voxel grid."""
    EMPTY = 0
    PART = 1
    RISER = 3
    INGATE = 5
    RUNNER = 6
    SPRUE = 7
    CORE = 9
    COOLING_SPRUE = 11
    FILTER = 13
    POURING_BASIN = 15
    SPRUE_THROAT = 17


BODY_TYPE_LABELS = {
    BodyType.PART: "PARÇA",
    BodyType.RISER: "BESLEYİCİ",
    BodyType.INGATE: "MEME",
    BodyType.RUNNER: "YOLLUK",
    BodyType.SPRUE: "DÖKÜM AĞZI",
    BodyType.CORE: "MAÇA",
    BodyType.COOLING_SPRUE: "SOĞUTUCU DÖKÜM AĞZI",
    BodyType.FILTER: "FİLTRE",
    BodyType.POURING_BASIN: "DÖKÜM HAVZASI",
    BodyType.SPRUE_THROAT: "D.AĞZI BOĞAZI",
}

# Body types that contain liquid metal during pouring (part + gating + riser).
# CORE, FILTER and COOLING_SPRUE are not part of the liquid metal domain: the
# latter is a chill insert and should be treated as a heat sink, not as metal.
BODY_CASTING_METAL_TYPES = [
    BodyType.PART,
    BodyType.RISER,
    BodyType.INGATE,
    BodyType.RUNNER,
    BodyType.SPRUE,
    BodyType.SPRUE_THROAT,
    BodyType.POURING_BASIN,
]

# Backwards-compatible alias; cooling sprue and filter are excluded from
# geometric/thermal metal domain.
BODY_METAL_TYPES = BODY_CASTING_METAL_TYPES

# Body types that can act as a feeding source.
BODY_FEEDER_TYPES = [
    BodyType.RISER,
    BodyType.INGATE,
    BodyType.RUNNER,
    BodyType.SPRUE,
    BodyType.SPRUE_THROAT,
    BodyType.POURING_BASIN,
]

# Inserts that accelerate local cooling and must never be treated as feeders.
CHILL_BODY_TYPES = [BodyType.COOLING_SPRUE]


@dataclass
class Body:
    """One solid body extracted from the STEP file."""
    index: int
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    mesh: trimesh.Trimesh
    body_type: BodyType = BodyType.PART
    volume_cm3: float = 0.0
    surface_area_cm2: float = 0.0
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # v9.3: per-body user overrides from the GUI
    section_key: str = ""  # INGATE / RUNNER / SPRUE_BASE / SPRUE_THROAT
    section_area_cm2: float = 0.0
    feeder_type: str = ""  # conventional / exothermic / insulated / chilled / sleeve
    feeder_m_mm: float = 0.0
    feeder_note: str = ""


@dataclass
class HotSpot:
    position_mm: np.ndarray
    m_value_mm: float
    dist_to_riser_mm: float
    feed_ok: bool
    max_feeding_distance_mm: float
    # v7 heavy physics
    niyama_min: float = 0.0
    resistance: float = 0.0
    resistance_ok: bool = True
    local_sdf_max: float = 0.0
    # v7.1
    t_section_mm: float = 0.0
    darcy_resistance: float = 0.0
    directional_ok: bool = True
    min_neck_m_mm: float = 0.0
    # v7.2
    width_mm: float = 0.0
    shape_factor: float = 0.0
    curvature_mean: float = 0.0
    curvature_gaussian: float = 0.0
    m_uncertainty_mm: float = 0.0
    niyama_ensemble: float = 0.0
    niyama_variants: Dict[str, float] = field(default_factory=dict)
    # v8.0
    heuvers_ok: bool = True
    feeding_cost: float = 0.0
    darcy_ok: bool = True
    # v8.8: estimated pore size from Niyama + SDAS + feeding risk
    pore_size_um: float = 0.0
    pore_size_mm: float = 0.0
    pore_size_class: str = ""


@dataclass
class RiserResult:
    body_index: int
    name: str
    volume_cm3: float
    surface_area_cm2: float
    m_value_mm: float
    target_hotspot_m_mm: float
    large_enough: bool
    volume_ratio_ok: bool
    nearest_hotspot_position_mm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gravity_factor: float = 1.0
    effective_m_required: float = 0.0
    # v7.1
    required_volume_cm3: float = 0.0
    # v7.2
    resistance_correction_mm: float = 0.0
    # v9.0
    mass_kg: float = 0.0
    feed_to_part_mass_ratio: float = 0.0
    feed_to_part_volume_ratio: float = 0.0
    # v9.3: user feeder type / modulus and the effective modulus used in checks
    feeder_type: str = ""
    feeder_m_user_mm: float = 0.0
    effective_m_value_mm: float = 0.0


@dataclass
class RiserProposal:
    target_hotspot_index: int
    target_hotspot_position_mm: np.ndarray
    placement_mm: np.ndarray
    reason: str
    m_required_mm: float
    shape: str
    diameter_mm: float
    height_mm: float
    volume_cm3: float
    neck_diameter_mm: float = 0.0
    neck_height_mm: float = 0.0
    # v9.2: proposal metadata
    exothermic: bool = False
    infeasible: bool = False
    warning: str = ""


@dataclass
class SectionFlow:
    """Flow velocity and Re/Fr at one gating section."""
    velocity_m_s: float = 0.0
    area_cm2: float = 0.0
    thickness_mm: float = 0.0
    reynolds: float = 0.0
    froude: float = 0.0
    turbulent: bool = False
    max_velocity_m_s: float = 1.0
    # v8.4: target velocity/area range for the recommended gating system
    target_v_min_m_s: float = 0.0
    target_v_max_m_s: float = 0.0
    target_area_min_cm2: float = 0.0
    target_area_max_cm2: float = 0.0


@dataclass
class GateResult:
    total_ingate_contact_area_cm2: float
    runner_min_area_cm2: float
    sprue_base_area_cm2: float
    required_sprue_area_cm2: float
    campbell_ok: bool
    bernoulli_ok: bool
    ingate_on_thick_region: bool
    ingate_avg_m_mm: float
    ingate_max_m_mm: float
    ingate_thickness_mm: float = 0.0
    runner_thickness_mm: float = 0.0
    # v7.1
    required_runner_area_cm2: float = 0.0
    required_ingate_area_cm2: float = 0.0
    runner_ok: bool = True
    ingate_ok: bool = True
    # v7.2
    elbow_count: int = 0
    head_loss_mm: float = 0.0
    effective_head_mm: float = 0.0
    required_sprue_area_with_losses_cm2: float = 0.0
    # v8.0 flow
    ingate_velocity_m_s: float = 0.0
    ingate_max_velocity_m_s: float = 1.0
    reynolds: float = 0.0
    froude: float = 0.0
    turbulent: bool = False
    # v8.1
    ingate_flow_rate_m3_s: float = 0.0
    ingate_fill_time_s: float = 0.0
    velocity_fill_time_match_ok: bool = True
    required_ingate_area_for_velocity_cm2: float = 0.0
    velocity_area_ok: bool = True
    fluidity_length_mm: float = 0.0
    # v8.3: separate sprue throat (minimum) and sprue base (bottom) areas + per-section flows
    sprue_throat_area_cm2: float = 0.0
    sprue_base_bottom_area_cm2: float = 0.0
    sprue_thickness_mm: float = 0.0
    selected_section_key: str = "INGATE"
    selected_velocity_m_s: float = 0.0
    section_flows: Dict[str, SectionFlow] = field(default_factory=dict)
    # v8.4 gating system classification and wall-thickness recommendation
    effective_gate_section: str = "INGATE"
    detected_gating_system: str = ""
    recommended_gating_system: str = ""
    wall_thickness_category: str = ""
    gating_system_reason: str = ""
    # v8.5/v8.6: fill-time design (gating_calculator_tr.py + Filling_time_tr.py)
    recommended_fill_time_s: float = 0.0
    fill_time_basis: str = ""
    auto_fill_time_s: float = 0.0
    campbell_fill_time_s: float = 0.0
    campbell_fill_time_basis: str = ""
    head_reduction_percent: float = 0.0
    total_poured_mass_kg: float = 0.0
    pouring_yield: float = 0.0
    design_sprue_base_area_cm2: float = 0.0
    design_runner_area_cm2: float = 0.0
    design_gate_total_area_cm2: float = 0.0
    design_gate_each_area_cm2: float = 0.0
    design_sprue_diameter_mm: float = 0.0
    design_gate_diameter_mm: float = 0.0
    design_choke_velocity_m_s: float = 0.0
    design_gating_ratio: Tuple[float, float, float] = field(default_factory=lambda: (1.0, 2.0, 1.0))
    sprue_design_ok: bool = True
    runner_design_ok: bool = True
    gate_design_ok: bool = True
    # v9.0: part and feed metal masses for feeder/part ratio checks.
    part_mass_kg: float = 0.0
    total_riser_mass_kg: float = 0.0
    gating_mass_kg: float = 0.0
    feed_to_part_mass_ratio: float = 0.0
    feed_to_part_volume_ratio: float = 0.0


@dataclass
class RefinementRegion:
    """High-resolution local grid around a single hot spot."""
    hotspot_index: int
    origin_mm: np.ndarray
    dx_mm: float
    grid: np.ndarray  # local mat_id
    sdf: np.ndarray
    niyama: np.ndarray
    risk: np.ndarray


@dataclass
class AnalysisResult:
    grid: np.ndarray  # mat_id
    origin_mm: np.ndarray
    dx_mm: float
    is_metal: np.ndarray
    sdf: np.ndarray
    dist_to_riser: np.ndarray
    risk: np.ndarray
    # v7 physics fields
    solidification_time: np.ndarray
    niyama: np.ndarray
    gradient_magnitude: np.ndarray
    hotspots: List[HotSpot]
    riser_results: List[RiserResult]
    gate_result: Optional[GateResult]
    recommendations: List[str] = field(default_factory=list)
    # v7 adaptive
    local_regions: List[RefinementRegion] = field(default_factory=list)
    # v7.1 material
    alloy_key: str = "42CrMo4"
    mold_key: str = "sand"
    alloy_name: str = "42CrMo4 (Çelik)"
    mold_name: str = "Kum Kalıp"
    chvorinov_c: float = 2.8
    unit_scale: float = 1.0
    # section / histogram
    dominant_m_mm: float = 0.0
    wall_thickness_mm: float = 0.0
    # v7.2 extended physics
    temperature: np.ndarray = field(default_factory=lambda: np.array([]))
    cooling_rate: np.ndarray = field(default_factory=lambda: np.array([]))
    solid_fraction: np.ndarray = field(default_factory=lambda: np.array([]))
    curvature_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    curvature_gaussian: np.ndarray = field(default_factory=lambda: np.array([]))
    subvoxel_sdf: np.ndarray = field(default_factory=lambda: np.array([]))
    shape_factor_global: float = 0.0
    m_mean_mm: float = 0.0
    m_std_mm: float = 0.0
    m_skewness: float = 0.0
    niyama_variants: Dict[str, np.ndarray] = field(default_factory=dict)
    elapsed_s: float = 0.0
    casting_params: Optional[CastingParameters] = None
    # v8.6: global part geometry for riser/modulus calculations
    part_volume_mm3: float = 0.0
    part_surface_area_mm2: float = 0.0
    thermal_divergence: np.ndarray = field(default_factory=lambda: np.array([]))
    riser_proposals: List[RiserProposal] = field(default_factory=list)
    # v8.8: per-voxel estimated pore size (µm) and macro/micro/fine masks
    pore_size_um: np.ndarray = field(default_factory=lambda: np.array([]))
    pore_size_mm: np.ndarray = field(default_factory=lambda: np.array([]))
    pore_size_macro_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    pore_size_micro_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    pore_size_fine_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    # v8.9: default noise filter (top % of computed porosity to display)
    pore_size_noise_percent: float = 3.0
    pore_size_threshold_um: float = 0.0
    # metadata
    bbox_size_mm: np.ndarray = field(default_factory=lambda: np.zeros(3))
