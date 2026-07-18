"""Shared types and result containers for JoseCast Analyzer v7.1."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh


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
    # metadata
    bbox_size_mm: np.ndarray = field(default_factory=lambda: np.zeros(3))
