"""Ingate / runner / sprue geometric gating calculations - JoseCast v8.0."""

from dataclasses import replace
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix, csgraph

from core.materials import get_alloy, get_mold, chvorinov_c_from_properties
from core.sdf_analyzer import COST_26, NEIGH_26
from core.types import (
    BODY_FEEDER_TYPES,
    BODY_METAL_TYPES,
    AnalysisResult,
    BodyType,
    GateResult,
    SectionFlow,
)


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


def _gate_source_mask(grid: np.ndarray) -> np.ndarray:
    """Return gating bodies that can feed metal into the part."""
    return np.isin(
        grid,
        [BodyType.INGATE, BodyType.RUNNER, BodyType.SPRUE, BodyType.COOLING_SPRUE, BodyType.POURING_BASIN],
    )


def ingate_contact_area_and_mask(grid: np.ndarray, dx: float) -> tuple:
    """Return (total ingate-part contact face area in mm2, source voxels touching part)."""
    source = _gate_source_mask(grid)
    part = grid == BodyType.PART
    contact_source = np.zeros_like(source)
    face_count = 0
    for di, dj, dk in _neighbor_offsets_6():
        rolled = np.roll(part, (di, dj, dk), axis=(0, 1, 2))
        _apply_edge_mask(rolled, di, dj, dk)
        faces = source & rolled
        contact_source |= faces
        face_count += int(faces.sum())
    return face_count * dx * dx, contact_source


def _part_touching_ingate_mask(grid: np.ndarray) -> np.ndarray:
    """Return part voxels that have at least one gate-source (ingate/runner/sprue) neighbor."""
    source = _gate_source_mask(grid)
    part = grid == BodyType.PART
    touch = np.zeros_like(part)
    for di, dj, dk in _neighbor_offsets_6():
        rolled = np.roll(source, (di, dj, dk), axis=(0, 1, 2))
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
    counts = []
    for s in np.unique(slices):
        counts.append((s, np.sum(slices == s)))
    if not counts:
        return 0.0
    max_count = max(c for _, c in counts)
    min_area = float("inf")
    for s, count in counts:
        # Ignore end slices that contain only a few voxels; they are not a real cross-section.
        if count < max(3, max_count * 0.05):
            continue
        area = count * dx * dx
        if area < min_area:
            min_area = area
    if min_area == float("inf"):
        # Fallback: use the largest slice if all were tiny.
        s, c = max(counts, key=lambda x: x[1])
        min_area = c * dx * dx
    return min_area


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


def _compute_section_flow(
    section_key: str,
    area_cm2: float,
    thickness_mm: float,
    Q_m3_s: float,
    rho: float,
    mu: float,
    g: float,
    max_velocity_m_s: float,
) -> SectionFlow:
    """Velocity, Reynolds, Froude and turbulence flag for one gating section."""
    area_m2 = area_cm2 / 1e4
    if area_m2 > 0 and Q_m3_s > 0:
        velocity = Q_m3_s / area_m2
    else:
        velocity = 0.0
    D = max(thickness_mm / 1000.0, 1e-6)
    reynolds = 0.0
    froude = 0.0
    turbulent = False
    if velocity > 0:
        reynolds = rho * velocity * D / mu
        froude = velocity / np.sqrt(g * D)
        # Ingate also checked against Campbell max velocity; other sections rely on Re/Fr.
        if section_key == "INGATE":
            turbulent = (reynolds > 20000.0) or (froude > 1.0) or (velocity > max_velocity_m_s)
        else:
            turbulent = (reynolds > 20000.0) or (froude > 1.0)
    return SectionFlow(
        velocity_m_s=velocity,
        area_cm2=area_cm2,
        thickness_mm=thickness_mm,
        reynolds=reynolds,
        froude=froude,
        turbulent=turbulent,
        max_velocity_m_s=max_velocity_m_s,
    )


def analyze_gating(
    result: AnalysisResult,
    fill_time_s: Optional[float] = None,
    discharge_coeff: float = 0.8,
    casting_params=None,
) -> Optional[GateResult]:
    """Compute gate/sprue/runner checks including Bernoulli elbow losses and all section velocities/Re/Fr."""
    from core.types import CastingParameters

    grid = result.grid
    sdf = result.sdf
    dx = result.dx_mm
    alloy = get_alloy(result.alloy_key)
    mold = get_mold(result.mold_key)
    velocity_section_key = "INGATE"
    if casting_params is not None and isinstance(casting_params, CastingParameters):
        fill_time_s = casting_params.t_fill_s
        velocity_section_key = getattr(casting_params, "velocity_section_key", "INGATE") or "INGATE"
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
    source = _gate_source_mask(grid)
    has_ingate = ingate.any()

    # Contact/cross-sectional areas of each gating section
    ag_total_mm2, _ = ingate_contact_area_and_mask(grid, dx)
    ag_total_cm2 = ag_total_mm2 / 100.0

    runner_min_area_mm2 = _minimum_cross_section_area(runner, dx)
    runner_min_area_cm2 = runner_min_area_mm2 / 100.0

    sprue_throat_mm2 = _minimum_cross_section_area(sprue, dx) if sprue.any() else 0.0
    sprue_throat_cm2 = sprue_throat_mm2 / 100.0
    sprue_base_bottom_mm2 = _sprue_base_area(sprue, dx) if sprue.any() else 0.0
    sprue_base_bottom_cm2 = sprue_base_bottom_mm2 / 100.0

    # Legacy field: sprue_base_area_cm2 is the bottom area.
    sprue_base_cm2 = sprue_base_bottom_cm2

    runner_thickness_mm = _mean_thickness(runner, dx)
    sprue_thickness_mm = _mean_thickness(sprue, dx)
    ingate_thickness_mm = _mean_thickness(source, dx)

    # Part weight for Bernoulli
    part_volume_mm3 = float(part_mask.sum()) * (dx ** 3)
    part_volume_cm3 = part_volume_mm3 / 1000.0
    part_weight_g = part_volume_cm3 * alloy.density_g_cm3
    part_volume_m3 = part_volume_mm3 / 1e9

    # Section area map for velocity calculations
    section_area_cm2 = {
        "INGATE": ag_total_cm2,
        "RUNNER": runner_min_area_cm2,
        "SPRUE_THROAT": sprue_throat_cm2,
        "SPRUE_BASE": sprue_base_bottom_cm2,
    }
    selected_area_cm2 = section_area_cm2.get(velocity_section_key, ag_total_cm2)
    selected_area_m2 = selected_area_cm2 / 1e4

    # v8.3: user-specified inlet velocity for the selected section (0 = auto Q = V_part / t_fill)
    user_v = 0.0
    if casting_params is not None and isinstance(casting_params, CastingParameters):
        user_v = float(getattr(casting_params, "ingate_velocity_m_s", 0.0))

    Q_m3_s = 0.0
    t_fill_computed_s = fill_time_s
    velocity_fill_time_match_ok = True
    required_selected_area_for_velocity_m2 = 0.0
    velocity_area_ok = True

    if selected_area_m2 > 0 and part_volume_m3 > 0:
        if user_v > 0:
            Q_m3_s = user_v * selected_area_m2
            t_fill_computed_s = part_volume_m3 / Q_m3_s
            if fill_time_s > 0:
                tolerance = 0.15 * fill_time_s
                velocity_fill_time_match_ok = abs(t_fill_computed_s - fill_time_s) <= tolerance
                required_Q_m3_s = part_volume_m3 / fill_time_s
                required_selected_area_for_velocity_m2 = required_Q_m3_s / user_v
                velocity_area_ok = selected_area_m2 >= required_selected_area_for_velocity_m2 * 0.95
        elif fill_time_s > 0:
            Q_m3_s = part_volume_m3 / fill_time_s
            t_fill_computed_s = fill_time_s

    # Compute per-section velocities and Re/Fr
    g_m_s2 = 9.81
    rho = alloy.rho_kg_m3
    mu = max(alloy.viscosity_pa_s, 1e-6)
    max_ingate_velocity = 0.5 if alloy.key in ("AlSi7",) else 1.0

    section_flows: Dict[str, SectionFlow] = {}
    for key, area_cm2 in section_area_cm2.items():
        thickness = {
            "INGATE": ingate_thickness_mm,
            "RUNNER": runner_thickness_mm,
            "SPRUE_THROAT": sprue_thickness_mm,
            "SPRUE_BASE": sprue_thickness_mm,
        }.get(key, ingate_thickness_mm)
        max_v = max_ingate_velocity if key == "INGATE" else 2.0
        section_flows[key] = _compute_section_flow(
            key, area_cm2, thickness, Q_m3_s, rho, mu, g_m_s2, max_v
        )

    ingate_flow = section_flows.get("INGATE", SectionFlow())
    ingate_velocity_m_s = ingate_flow.velocity_m_s
    ingate_flow_rate_m3_s = Q_m3_s
    ingate_fill_time_s = t_fill_computed_s
    reynolds = ingate_flow.reynolds
    froude = ingate_flow.froude
    turbulent = ingate_flow.turbulent

    # Sprue height H in cm; fallback to total metal height
    metal_pts = np.argwhere(result.is_metal)
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
        velocity = 0.0

    # Bernoulli with elbow losses along the gating channel
    channel_mask = np.isin(
        grid,
        [BodyType.INGATE, BodyType.RUNNER, BodyType.SPRUE, BodyType.COOLING_SPRUE, BodyType.POURING_BASIN],
    )
    sprue_mask = sprue & channel_mask
    effective_head_cm = height_cm
    elbow_count = 0
    head_loss_cm = 0.0
    required_sprue_area_with_losses_cm2 = as_req_cm2
    if channel_mask.any() and sprue_mask.any() and ingate.any():
        dist_to_sprue = _distance_to_sprue_26(channel_mask, sprue_mask, dx)
        ingate_vox = np.argwhere(ingate)
        sample = ingate_vox[
            np.linspace(0, len(ingate_vox) - 1, min(20, len(ingate_vox))).astype(int)
        ]
        counts = []
        for v in sample:
            counts.append(_count_elbows_along_path(dist_to_sprue, channel_mask, tuple(v)))
        elbow_count = int(round(np.median(counts))) if counts else 0
        if velocity > 0:
            v_loss_m_s = velocity / 100.0
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

    # Bernoulli: controlling section is the sprue throat (minimum area)
    final_sprue_required_cm2 = max(as_req_cm2, required_sprue_area_with_losses_cm2)
    bernoulli_ok = sprue_throat_cm2 >= final_sprue_required_cm2

    # Campbell: total ingate area / runner area < 1.5
    if runner_min_area_cm2 > 0:
        gate_to_runner_ratio = ag_total_cm2 / runner_min_area_cm2
        campbell_ok = gate_to_runner_ratio <= 1.5
    else:
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

    required_ingate_area_cm2 = required_runner_area_cm2 / 1.5
    ingate_ok = (ag_total_cm2 >= required_ingate_area_cm2) and (not ingate_on_thick)
    if user_v > 0:
        ingate_ok = ingate_ok and velocity_area_ok

    # Fluidity length uses the ingate velocity
    t_stream = max(ingate_thickness_mm, runner_thickness_mm, 2.0 * result.dominant_m_mm, 2.0)
    M_stream = t_stream / 2.0
    C = chvorinov_c_from_properties(alloy, mold)
    t_s_stream = C * M_stream ** 2
    superheat = max(alloy.t_pour_c - alloy.t_liquidus_c, 0.0)
    l_eff = alloy.latent_heat_j_kg + alloy.cp_j_kgk * superheat
    superheat_ratio = max(alloy.cp_j_kgk * superheat / l_eff, 0.1) if l_eff > 0 else 0.1
    t_superheat = t_s_stream * superheat_ratio
    v_metal_m_s = ingate_velocity_m_s
    if v_metal_m_s <= 0 and height_mm > 0:
        v_metal_m_s = np.sqrt(2.0 * 9.81 * (height_mm / 1000.0))
    fluidity_length_mm = v_metal_m_s * t_superheat * 1000.0

    max_dim_mm = float(result.bbox_size_mm.max())
    result.recommendations = [
        r for r in result.recommendations if not r.startswith("Sıvı akışkanlık")
    ]
    if max_dim_mm > fluidity_length_mm:
        result.recommendations.append(
            f"Sıvı akışkanlık uzunluğu Lf = {fluidity_length_mm:.1f} mm, parça boyutu {max_dim_mm:.1f} mm. "
            f"Soğuk birleşme (cold shut) riski - döküm sıcaklığını artırın, kalıp sıcaklığını yükseltmeyin veya giriş hızını artırın."
        )
    else:
        result.recommendations.append(
            f"Akışkanlık uzunluğu Lf = {fluidity_length_mm:.1f} mm, parça boyutu {max_dim_mm:.1f} mm -> yeterli."
        )

    # Build a compact summary of all section velocities for recommendations
    velocity_summary = " | ".join(
        f"{k}: {sf.velocity_m_s:.2f}m/s (Re={sf.reynolds:.0f}, Fr={sf.froude:.2f})"
        for k, sf in section_flows.items()
        if sf.area_cm2 > 0
    )
    if velocity_summary:
        result.recommendations.append(f"Kesit hızları -> {velocity_summary}")

    return GateResult(
        total_ingate_contact_area_cm2=ag_total_cm2,
        runner_min_area_cm2=runner_min_area_cm2,
        sprue_base_area_cm2=sprue_base_cm2,
        required_sprue_area_cm2=final_sprue_required_cm2,
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
        elbow_count=elbow_count,
        head_loss_mm=head_loss_cm * 10.0,
        effective_head_mm=effective_head_cm * 10.0,
        required_sprue_area_with_losses_cm2=required_sprue_area_with_losses_cm2,
        ingate_velocity_m_s=ingate_velocity_m_s,
        ingate_max_velocity_m_s=max_ingate_velocity,
        reynolds=reynolds,
        froude=froude,
        turbulent=turbulent,
        ingate_flow_rate_m3_s=ingate_flow_rate_m3_s,
        ingate_fill_time_s=ingate_fill_time_s,
        velocity_fill_time_match_ok=velocity_fill_time_match_ok,
        required_ingate_area_for_velocity_cm2=required_selected_area_for_velocity_m2 * 1e4,
        velocity_area_ok=velocity_area_ok,
        fluidity_length_mm=fluidity_length_mm,
        sprue_throat_area_cm2=sprue_throat_cm2,
        sprue_base_bottom_area_cm2=sprue_base_bottom_cm2,
        sprue_thickness_mm=sprue_thickness_mm,
        selected_section_key=velocity_section_key,
        selected_velocity_m_s=user_v,
        section_flows=section_flows,
    )
