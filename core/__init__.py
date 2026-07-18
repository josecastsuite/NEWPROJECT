"""JoseCast Analyzer geometric + pseudo-thermal engine v8.0."""

from core.gating import analyze_gating, ingate_contact_area_and_mask
from core.materials import (
    ALLOYS,
    MATERIALS,
    MOLDS,
    Alloy,
    Material,
    MoldMaterial,
    get_alloy,
    get_material,
    get_mold,
)
from core.reporter import generate_report
from core.sdf_analyzer import analyze
from core.step_loader import load_step
from core.types import (
    AnalysisResult,
    Body,
    BodyType,
    CastingParameters,
    GateResult,
    HotSpot,
    RefinementRegion,
    RiserResult,
)
from core.voxelizer import (
    BASE_RES,
    MAX_RES,
    apply_unit_scale,
    build_voxel_grid,
    detect_unit_suggestion,
)

__all__ = [
    "load_step",
    "build_voxel_grid",
    "apply_unit_scale",
    "detect_unit_suggestion",
    "analyze",
    "analyze_gating",
    "ingate_contact_area_and_mask",
    "generate_report",
    "get_alloy",
    "get_mold",
    "get_material",
    "Alloy",
    "MoldMaterial",
    "Material",
    "ALLOYS",
    "MOLDS",
    "MATERIALS",
    "BASE_RES",
    "MAX_RES",
    "Body",
    "BodyType",
    "AnalysisResult",
    "HotSpot",
    "RiserResult",
    "GateResult",
    "RefinementRegion",
    "CastingParameters",
]
