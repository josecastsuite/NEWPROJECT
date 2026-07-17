"""JoseCast Analyzer geometric engine."""

from core.gating import analyze_gating, ingate_contact_area_and_mask
from core.reporter import generate_report
from core.sdf_analyzer import analyze
from core.step_loader import load_step
from core.types import (
    AnalysisResult,
    Body,
    BodyType,
    GateResult,
    HotSpot,
    RiserResult,
)
from core.voxelizer import build_voxel_grid

__all__ = [
    "load_step",
    "build_voxel_grid",
    "analyze",
    "analyze_gating",
    "ingate_contact_mask",
    "generate_report",
    "Body",
    "BodyType",
    "AnalysisResult",
    "HotSpot",
    "RiserResult",
    "GateResult",
]
