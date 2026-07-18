"""Ingate / runner / sprue geometric gating calculations - JoseCast v8.0."""

from dataclasses import replace
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix, csgraph

from core.materials import get_alloy
from core.sdf_analyzer import COST_26, NEIGH_26
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
    """Approximate minimum cross-sectional area of a voxel set using PCA slicing."""
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


def _distance_to_sprue_26(channel_mask: np.ndarray, sprue_mask: np.ndarray, dx: float) -> np.ndarray:
    """26-neighbor Dijkstra distance from every channel voxel to the sprue."""
    dist = np.full(channel_mask.shape, np.inf, dtype=np.float64)
    if not (channel_mask & sprue_mask).any():
        return dist

    idx = np.full(channel_mask.shape, -1, dtype=np.int64)
    vox = np.argwhere(channel_mask)
    n = int(vox.shape[0])
    idx[tuple(vox.T)] = np.arange(n)

    rows, cols, vals = [], [], []
    for (di, dj, dk), c in zip(NEIGH_26, COST_26):
        ni = vox[:, 0] + di
        nj = vox[:, 1] + dj
        nk = vox[:, 2] + dk
        mask = (
            (ni >= 0)
            & (ni < channel_mask.shape[0])
            & (nj >= 0)
            & (nj < channel_mask.shape[1])
            & (nk >= 0)
            & (nk < channel_mask.shape[2])
        )
        if not mask.any():
            continue
        neighbor_idx = idx[ni[mask], nj[mask], nk[mask]]
        source_idx = np.arange(n)[mask]
        valid = neighbor_idx >= 0
        if not valid.any():
            continue
        rows.append(source_idx[valid])
        cols.append(neighbor_idx[valid])
        vals.append(np.full(valid.sum(), c * dx, dtype=np.float32))

    sprue_flat = np.where(sprue_mask[tuple(vox.T)])[0]
    rows.append(np.full(len(sprue_flat), n, dtype=np.int64))
    cols.append(sprue_flat.astype(np.int64))
    vals.append(np.zeros(len(sprue_flat), dtype=np.float32))

    graph = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n + 1, n + 1),
    ).tocsr()
    flat_dist = csgraph.dijkstra(graph, directed=False, indices=n, return_predecessors=False)
    dist[tuple(vox.T)] = flat_dist[:n].astype(np.float64)
    return dist


def _count_elbows_along_path(
    dist: np.ndarray,
    channel_mask: np.ndarray,
    start: Tuple[int, int, int],
    angle_threshold_deg: float = 60.0,
) -> int:
    """Trace from start toward decreasing dist and count sharp direction changes."""
    shape = dist.shape
    current = start
    if not channel_mask[current]:
        return 0
    path = [current]
    visited = {current}
    for _ in range(1000):
        i, j, k = current
        if dist[i, j, k] <= 0:
            break
        best = None
        best_d = dist[i, j, k]
        for di, dj, dk in _neighbor_offsets_6():
            ni, nj, nk = i + di, j + dj, k + dk
            if not (0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]):
                continue
            if not channel_mask[ni, nj, nk]:
                continue
            d = dist[ni, nj, nk]
            if d < best_d:
                best_d = d
                best = (ni, nj, nk)
        if best is None or best in visited:
            break
        visited.add(best)
        path.append(best)
        current = best

    if len(path) < 3:
        return 0
    elbows = 0
    cos_thresh = np.cos(np.deg2rad(angle_threshold_deg))
    for a in range(1, len(path) - 1):
        v1 = np.array(path[a]) - np.array(path[a - 1])
        v2 = np.array(path[a + 1]) - np.array(path[a])
        n1 = v1 / (np.linalg.norm(v1) + 1e-12)
        n2 = v2 / (np.linalg.norm(v2) + 1e-12)
        if np.dot(n1, n2) < cos_thresh:
            elbows += 1
    return elbows


def analyze_gating(
    result: AnalysisResult,
    fill_time_s: Optional[float] = None,
    discharge_coeff: float = 0.8,
    casting_params=None,
) -> Optional[GateResult]:
    """Compute gate/sprue/runner checks including Bernoulli elbow losses and ingate velocity/Re/Fr."""
    from core.types import CastingParameters

    grid = result.grid
    sdf = result.sdf
    dx = result.dx_mm
    alloy = get_alloy(result.alloy_key)
    if casting_params is not None and isinstance(casting_params, CastingParameters):
        fill_time_s = casting_params.t_fill_s
        alloy = replace(
            alloy,
            t_pour_c=casting_params.t_pour_c,
            t_liquidus_c=casting_params.t_liquidus_c,
            t_solidus_c=casting_params.t_solidus_c,
            rho_kg_m3=casting_params.rho_liquid_kg_m3,
            viscosity_pa_s=casting_params.viscosity_pa_s,
        )
    if fill_time_s is None:
        fill_time_s = 10.0

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
    part_volume_mm3 = float(part_mask.sum()) * (dx ** 3)
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
    # Base Bernoulli velocity without losses
    if height_cm > 0 and fill_time_s > 0:
        velocity = np.sqrt(2.0 * g_cgs * height_cm)
        as_req_cm2 = part_weight_g / (
            alloy.density_g_cm3 * discharge_coeff * fill_time_s * velocity
        )
    else:
        as_req_cm2 = 0.0
        velocity = 0.0

    # Bernoulli with elbow losses along runner/sprue channel
    channel_mask = runner | ingate | sprue
    sprue_mask = sprue & channel_mask
    effective_head_cm = height_cm
    elbow_count = 0
    head_loss_cm = 0.0
    required_sprue_area_with_losses_cm2 = as_req_cm2
    if channel_mask.any() and sprue_mask.any() and ingate.any():
        dist_to_sprue = _distance_to_sprue_26(channel_mask, sprue_mask, dx)
        ingate_vox = np.argwhere(ingate)
        # Sample up to 20 ingate voxels to estimate elbow count
        sample = ingate_vox[
            np.linspace(0, len(ingate_vox) - 1, min(20, len(ingate_vox))).astype(int)
        ]
        counts = []
        for v in sample:
            counts.append(_count_elbows_along_path(dist_to_sprue, channel_mask, tuple(v)))
        elbow_count = int(round(np.median(counts))) if counts else 0
        if velocity > 0:
            v_loss_m_s = velocity / 100.0  # cm/s -> m/s for head-loss formula
            h_loss_per_elbow_m = alloy.elbow_loss_k * (v_loss_m_s ** 2) / (2.0 * 9.81)
            head_loss_cm = h_loss_per_elbow_m * 100.0 * elbow_count
            effective_head_cm = max(0.0, height_cm - head_loss_cm)
            if effective_head_cm > 0 and fill_time_s > 0:
                v_eff = np.sqrt(2.0 * g_cgs * effective_head_cm)
                required_sprue_area_with_losses_cm2 = part_weight_g / (
                    alloy.density_g_cm3 * discharge_coeff * fill_time_s * v_eff
                )
            else:
                required_sprue_area_with_losses_cm2 = float("inf")

    bernoulli_ok = sprue_base_cm2 >= as_req_cm2
    bernoulli_with_losses_ok = sprue_base_cm2 >= required_sprue_area_with_losses_cm2

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

    # Prefer Bernoulli check that accounts for losses
    final_sprue_required_cm2 = max(as_req_cm2, required_sprue_area_with_losses_cm2)
    final_bernoulli_ok = sprue_base_cm2 >= final_sprue_required_cm2

    # v8.0: ingate velocity, Reynolds and Froude checks
    ingate_velocity_m_s = 0.0
    reynolds = 0.0
    froude = 0.0
    turbulent = False
    if ag_total_mm2 > 0 and fill_time_s > 0:
        part_volume_mm3 = float(part_mask.sum()) * (dx ** 3)
        Q_m3_s = (part_volume_mm3 / 1e9) / fill_time_s
        ag_m2 = ag_total_mm2 / 1e6
        ingate_velocity_m_s = Q_m3_s / ag_m2 if ag_m2 > 0 else 0.0

    # Alloy-specific maximum ingate velocity (Campbell rule)
    max_ingate_velocity = 0.5 if alloy.key in ("AlSi7",) else 1.0
    # Characteristic hydraulic diameter D ≈ 2*mean_wall_thickness of ingate
    D = max(ingate_thickness_mm / 1000.0, 1e-6)
    g = 9.81
    rho = alloy.rho_kg_m3
    mu = max(alloy.viscosity_pa_s, 1e-6)
    if ingate_velocity_m_s > 0:
        reynolds = rho * ingate_velocity_m_s * D / mu
        froude = ingate_velocity_m_s / np.sqrt(g * D)
        # Turbulence if Re > 20000 or Fr > 1 or above alloy-specific ingate velocity
        turbulent = (reynolds > 20000.0) or (froude > 1.0) or (ingate_velocity_m_s > max_ingate_velocity)

    return GateResult(
        total_ingate_contact_area_cm2=ag_total_cm2,
        runner_min_area_cm2=runner_min_area_cm2,
        sprue_base_area_cm2=sprue_base_cm2,
        required_sprue_area_cm2=final_sprue_required_cm2,
        campbell_ok=campbell_ok,
        bernoulli_ok=final_bernoulli_ok,
        ingate_on_thick_region=ingate_on_thick,
        ingate_avg_m_mm=ingate_avg_m,
        ingate_max_m_mm=ingate_max_m,
        ingate_thickness_mm=ingate_thickness_mm,
        runner_thickness_mm=runner_thickness_mm,
        required_runner_area_cm2=required_runner_area_cm2,
        required_ingate_area_cm2=required_ingate_area_cm2,
        runner_ok=runner_ok,
        ingate_ok=ingate_ok,
        elbow_count=elbow_count,
        head_loss_mm=head_loss_cm * 10.0,
        effective_head_mm=effective_head_cm * 10.0,
        required_sprue_area_with_losses_cm2=required_sprue_area_with_losses_cm2,
        ingate_velocity_m_s=ingate_velocity_m_s,
        ingate_max_velocity_m_s=max_ingate_velocity,
        reynolds=reynolds,
        froude=froude,
        turbulent=turbulent,
    )
