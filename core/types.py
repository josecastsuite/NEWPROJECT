"""Shared types and result containers for JoseCast Analyzer v5.0."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple

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


@dataclass
class AnalysisResult:
    grid: np.ndarray  # mat_id
    origin_mm: np.ndarray
    dx_mm: float
    is_metal: np.ndarray
    sdf: np.ndarray
    dist_to_riser: np.ndarray
    risk: np.ndarray
    hotspots: List[HotSpot]
    riser_results: List[RiserResult]
    gate_result: Optional[GateResult]
    recommendations: List[str] = field(default_factory=list)
