"""JoseCast Analyzer geometric engine v7."""

from core.gating import analyze_gating, ingate_contact_area_and_mask
from core.materials import MATERIALS, Material, get_material
from core.reporter import generate_report
from core.sdf_analyzer import analyze
from core.step_loader import load_step
from core.types import (
    AnalysisResult,
    Body,
    BodyType,
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
    "get_material",
    "Material",
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
]
