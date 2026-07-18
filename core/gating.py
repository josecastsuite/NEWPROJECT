"""Ingate / runner / sprue geometric gating calculations - JoseCast v7.1."""

from typing import Optional

import numpy as np
from scipy import ndimage

from core.materials import get_alloy
from core.types import AnalysisResult, BodyType, GateResult


def _neighbor_offsets_6():
    return [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]


def _apply_edge_mask(arr, di, dj, dk):
    if di > 0:
        arr[-1, :, :] = False
    elif di < 0:
        arr[0, :, :] = False
    if dj > 0:
        arr[:, -1, :] = False
    elif dj < 0:
        arr[:, 0, :] = False
    if dk > 0:
        arr[:, :, -1] = False
    elif dk < 0:
        arr[:, :, 0] = False
    return arr


def ingate_contact_area_and_mask(grid: np.ndarray, dx: float) -> tuple:
    """Return (total ingate-part contact face area in mm2, ingate voxels touching part)."""
    ingate = grid == BodyType.INGATE
    part = grid == BodyType.PART
    contact_ingate = np.zeros_like(ingate)
    face_count = 0
    for di, dj, dk in _neighbor_offsets_6():
        rolled = np.roll(part, (di, dj, dk), axis=(0, 1, 2))
        _apply_edge_mask(rolled, di, dj, dk)
        faces = ingate & rolled
        contact_ingate |= faces
        face_count += int(faces.sum())
    return face_count * dx * dx, contact_ingate


def _part_touching_ingate_mask(grid: np.ndarray) -> np.ndarray:
    """Return part voxels that have at least one ingate neighbor."""
    ingate = grid == BodyType.INGATE
    part = grid == BodyType.PART
    touch = np.zeros_like(part)
    for di, dj, dk in _neighbor_offsets_6():
        rolled = np.roll(ingate, (di, dj, dk), axis=(0, 1, 2))
        _apply_edge_mask(rolled, di, dj, dk)
        touch |= rolled & part
    return touch


def _minimum_cross_section_area(mask: np.ndarray, dx: float) -> float:
    """Approximate minimum cross-sectional area of a voxel set using PCA."""
    pts = np.argwhere(mask)
    if len(pts) < 3:
        return 0.0

    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    principal = principal / (np.linalg.norm(principal) + 1e-12)

    proj = centered @ principal
    slices = np.round(proj).astype(int)
    min_area = float("inf")
    for s in np.unique(slices):
        count = np.sum(slices == s)
        area = count * dx * dx
        if area < min_area:
            min_area = area
    return min_area if min_area != float("inf") else 0.0


def _sprue_base_area(sprue_mask: np.ndarray, dx: float) -> float:
    """Area of the lowest-Z slice of the sprue."""
    pts = np.argwhere(sprue_mask)
    if len(pts) == 0:
        return 0.0
    min_z = pts[:, 2].min()
    base_slice = sprue_mask[:, :, int(min_z)]
    count = int(base_slice.sum())
    return count * dx * dx


def _mean_thickness(mask: np.ndarray, dx: float) -> float:
    """Mean wall thickness of a voxel set (2 * internal distance transform)."""
    if not mask.any():
        return 0.0
    edt = ndimage.distance_transform_edt(mask) * dx
    return float(edt[mask].mean()) * 2.0


def analyze_gating(
    result: AnalysisResult,
    fill_time_s: float = 10.0,
    discharge_coeff: float = 0.8,
) -> Optional[GateResult]:
    """Compute gate/sprue/runner checks."""
    grid = result.grid
    sdf = result.sdf
    dx = result.dx_mm
    alloy = get_alloy(result.alloy_key)

    part_mask = grid == BodyType.PART
    ingate = grid == BodyType.INGATE
    runner = grid == BodyType.RUNNER
    sprue = grid == BodyType.SPRUE

    ag_total_mm2, _ = ingate_contact_area_and_mask(grid, dx)
    ag_total_cm2 = ag_total_mm2 / 100.0

    runner_min_area_mm2 = _minimum_cross_section_area(runner, dx)
    runner_min_area_cm2 = runner_min_area_mm2 / 100.0
    runner_thickness_mm = _mean_thickness(runner, dx)

    sprue_base_mm2 = _sprue_base_area(sprue, dx)
    sprue_base_cm2 = sprue_base_mm2 / 100.0

    ingate_thickness_mm = _mean_thickness(ingate, dx)

    # Part weight for Bernoulli
    part_volume_mm3 = float(part_mask.sum()) * dx ** 3
    part_volume_cm3 = part_volume_mm3 / 1000.0
    part_weight_g = part_volume_cm3 * alloy.density_g_cm3

    # Sprue height H in cm; fallback to total metal height
    metal_mask = result.is_metal
    metal_pts = np.argwhere(metal_mask)
    if len(metal_pts) > 0:
        height_mm = float((metal_pts[:, 2].max() - metal_pts[:, 2].min()) * dx)
    else:
        height_mm = 0.0
    height_cm = height_mm / 10.0

    g_cgs = 981.0  # cm/s^2
    if height_cm > 0 and fill_time_s > 0:
        velocity = np.sqrt(2.0 * g_cgs * height_cm)
        as_req_cm2 = part_weight_g / (
            alloy.density_g_cm3 * discharge_coeff * fill_time_s * velocity
        )
    else:
        as_req_cm2 = 0.0

    bernoulli_ok = sprue_base_cm2 >= as_req_cm2

    # Campbell: total ingate area / runner area < 1.5
    if runner_min_area_cm2 > 0:
        gate_to_runner_ratio = ag_total_cm2 / runner_min_area_cm2
        campbell_ok = gate_to_runner_ratio <= 1.5
    else:
        gate_to_runner_ratio = float("inf")
        campbell_ok = False

    required_runner_area_cm2 = ag_total_cm2 / 1.5 if ag_total_cm2 > 0 else 0.0
    runner_ok = runner_min_area_cm2 >= required_runner_area_cm2

    # Ingate location check
    part_sdf = sdf[part_mask]
    max_part_sdf = float(part_sdf.max()) if len(part_sdf) > 0 else 0.0

    part_touch = _part_touching_ingate_mask(grid)
    contact_sdf = sdf[part_touch]
    if len(contact_sdf) > 0:
        ingate_avg_m = float(np.mean(contact_sdf))
        ingate_max_m = float(np.max(contact_sdf))
    else:
        ingate_avg_m = 0.0
        ingate_max_m = 0.0

    threshold = 0.8 * max_part_sdf
    ingate_on_thick = ingate_avg_m > threshold if max_part_sdf > 0 else False

    # A reasonable minimum ingate contact area: based on runner capacity
    required_ingate_area_cm2 = required_runner_area_cm2 / 1.5
    ingate_ok = (ag_total_cm2 >= required_ingate_area_cm2) and (not ingate_on_thick)

    return GateResult(
        total_ingate_contact_area_cm2=ag_total_cm2,
        runner_min_area_cm2=runner_min_area_cm2,
        sprue_base_area_cm2=sprue_base_cm2,
        required_sprue_area_cm2=as_req_cm2,
        campbell_ok=campbell_ok,
        bernoulli_ok=bernoulli_ok,
        ingate_on_thick_region=ingate_on_thick,
        ingate_avg_m_mm=ingate_avg_m,
        ingate_max_m_mm=ingate_max_m,
        ingate_thickness_mm=ingate_thickness_mm,
        runner_thickness_mm=runner_thickness_mm,
        required_runner_area_cm2=required_runner_area_cm2,
        required_ingate_area_cm2=required_ingate_area_cm2,
        runner_ok=runner_ok,
        ingate_ok=ingate_ok,
    )
