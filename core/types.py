"""Shared types and result containers for JoseCast Analyzer v7.2."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional

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
    # v8.1: user-specified ingate velocity (0 = auto from V_part / t_fill)
    ingate_velocity_m_s: float = 0.0

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


BODY_TYPE_LABELS = {
    BodyType.PART: "PARÇA",
    BodyType.RISER: "BESLEYİCİ",
    BodyType.INGATE: "MEME",
    BodyType.RUNNER: "YOLLUK",
    BodyType.SPRUE: "DÖKÜM AĞZI",
    BodyType.CORE: "MAÇA",
}


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
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))


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
    thermal_divergence: np.ndarray = field(default_factory=lambda: np.array([]))
    # metadata
    bbox_size_mm: np.ndarray = field(default_factory=lambda: np.zeros(3))
