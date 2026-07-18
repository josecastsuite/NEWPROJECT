"""SDF-based geometric casting analyzer."""

from collections import deque
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from core.types import (
    AnalysisResult,
    Body,
    BodyType,
    HotSpot,
    RiserResult,
)


NEIGH_6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def compute_sdf(is_metal: np.ndarray) -> np.ndarray:
    """Return Signed Distance Field inside metal: distance to nearest non-metal voxel."""
    return ndimage.distance_transform_edt(is_metal, return_distances=True).astype(np.float64)


def find_hotspots(
    sdf: np.ndarray,
    part_mask: np.ndarray,
    dx: float,
    origin_mm: np.ndarray,
    min_size_mm: float = 2.0,
    cluster_radius_mm: float = 10.0,
) -> List[HotSpot]:
    """Detect local maxima of the SDF inside the part and cluster them."""
    size_vox = 5  # voxel window for local maxima
    if size_vox % 2 == 0:
        size_vox += 1

    local_max = (sdf == ndimage.maximum_filter(sdf, size=size_vox, mode="constant"))
    candidates = np.argwhere(local_max & part_mask & (sdf > min_size_mm))
    if len(candidates) == 0:
        return []

    # Cluster with cKDTree using the requested radius.
    radius_vox = max(1.0, cluster_radius_mm / dx)
    tree = cKDTree(candidates.astype(np.float32))
    groups = tree.query_ball_point(candidates.astype(np.float32), r=radius_vox)

    visited = set()
    clusters: List[List[np.ndarray]] = []
    for i, group in enumerate(groups):
        if i in visited:
            continue
        cluster_indices = set(group)
        pending = set(cluster_indices)
        while pending:
            j = pending.pop()
            if j in visited:
                continue
            visited.add(j)
            new = set(groups[j])
            new_to_add = new - cluster_indices
            cluster_indices.update(new)
            pending.update(new_to_add)
        clusters.append([candidates[k] for k in cluster_indices])

    hotspots: List[HotSpot] = []
    for cluster in clusters:
        pts = np.array(cluster)
        # Keep the point with the maximum SDF value as the representative.
        vals = sdf[pts[:, 0], pts[:, 1], pts[:, 2]]
        max_idx = int(np.argmax(vals))
        pos_vox = pts[max_idx]
        m_value = float(vals[max_idx])
        position_mm = origin_mm + pos_vox * dx
        hotspots.append(
            HotSpot(
                position_mm=position_mm,
                m_value_mm=m_value,
                dist_to_riser_mm=np.inf,
                feed_ok=False,
                max_feeding_distance_mm=4.5 * m_value,
            )
        )

    return hotspots


def feeding_distance_bfs(
    is_metal: np.ndarray, riser_mask: np.ndarray, dx: float
) -> np.ndarray:
    """
    Geodesic distance (within metal) from every metal voxel to the nearest riser.
    6-neighbor BFS as described in the JoseCast spec.
    """
    dist = np.full(is_metal.shape, np.inf, dtype=np.float64)
    sources = np.argwhere(riser_mask & is_metal)
    if len(sources) == 0:
        return dist

    q = deque()
    for s in sources:
        dist[s[0], s[1], s[2]] = 0.0
        q.append((s[0], s[1], s[2]))

    shape = is_metal.shape
    while q:
        i, j, k = q.popleft()
        nd = dist[i, j, k] + dx
        for di, dj, dk in NEIGH_6:
            ni, nj, nk = i + di, j + dj, k + dk
            if not (0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]):
                continue
            if not is_metal[ni, nj, nk]:
                continue
            if nd < dist[ni, nj, nk]:
                dist[ni, nj, nk] = nd
                q.append((ni, nj, nk))

    return dist


def _body_mask(grid: np.ndarray, body_index: int) -> np.ndarray:
    """Placeholder; body index is not stored in grid. Kept for future per-body masks."""
    return np.zeros_like(grid, dtype=bool)


def analyze(
    bodies: List[Body],
    grid: np.ndarray,
    origin_mm: np.ndarray,
    dx: float,
    material_factor: float = 4.5,
    riser_volume_factor: float = 0.3,
    riser_m_factor: float = 1.2,
    progress_callback: Optional[callable] = None,
) -> AnalysisResult:
    """Run the full geometric analysis pipeline."""
    if progress_callback:
        progress_callback(55)

    is_metal = np.isin(grid, [BodyType.PART, BodyType.RISER, BodyType.INGATE, BodyType.RUNNER])
    if is_metal.sum() < 1000:
        raise ValueError("Model çok küçük. Çözünürlüğü artırın veya modelin mm biriminde olduğundan emin olun.")

    part_mask = grid == BodyType.PART
    riser_mask = grid == BodyType.RISER
    ingate_mask = grid == BodyType.INGATE
    runner_mask = grid == BodyType.RUNNER
    sprue_mask = grid == BodyType.SPRUE

    # AŞAMA 2: SDF
    sdf_vox = ndimage.distance_transform_edt(is_metal).astype(np.float64)
    sdf = sdf_vox * dx

    if progress_callback:
        progress_callback(65)

    # AŞAMA 3: Hot spot detection
    hotspots = find_hotspots(sdf, part_mask, dx, origin_mm)

    if progress_callback:
        progress_callback(75)

    # AŞAMA 4: Feeding distance
    dist_feed = feeding_distance_bfs(is_metal, riser_mask, dx)

    if progress_callback:
        progress_callback(85)

    # Update hotspot feeding distances
    for hs in hotspots:
        vox = np.round((hs.position_mm - origin_mm) / dx).astype(int)
        if 0 <= vox[0] < grid.shape[0] and 0 <= vox[1] < grid.shape[1] and 0 <= vox[2] < grid.shape[2]:
            hs.dist_to_riser_mm = float(dist_feed[vox[0], vox[1], vox[2]])
            if np.isinf(hs.dist_to_riser_mm):
                hs.feed_ok = False
            else:
                hs.feed_ok = hs.dist_to_riser_mm <= material_factor * hs.m_value_mm
        else:
            hs.feed_ok = False

    if progress_callback:
        progress_callback(90)

    # AŞAMA 5: Riser sufficiency
    riser_results: List[RiserResult] = []
    for body in bodies:
        if body.body_type != BodyType.RISER:
            continue
        # Body-specific mask using approximate region around center and same label
        # We cannot separate individual risers in grid without an extra per-body map.
        # For now use the label values and distance from the body center.
        body_center_vox = np.round((body.center - origin_mm) / dx).astype(int)

        # Find connected riser component closest to body center
        labeled, num = ndimage.label(riser_mask)
        if num == 0:
            continue
        # Pick the component whose centroid is closest to body center
        best_label = 1
        best_dist = np.inf
        for lbl in range(1, num + 1):
            pts = np.argwhere(labeled == lbl)
            centroid = pts.mean(axis=0)
            d = np.linalg.norm(centroid - body_center_vox)
            if d < best_dist:
                best_dist = d
                best_label = lbl

        component_mask = labeled == best_label
        voxel_count = int(component_mask.sum())
        if voxel_count == 0:
            continue

        volume_mm3 = voxel_count * dx ** 3
        volume_cm3 = volume_mm3 / 1000.0

        # Surface voxels via binary dilation difference
        dilated = ndimage.binary_dilation(component_mask, iterations=1)
        surface_mask = dilated & ~component_mask
        # Exclude surface voxels touching other metal as internal surfaces
        surface_voxels = surface_mask.sum()
        surface_mm2 = float(surface_voxels) * dx * dx

        m_riser = volume_mm3 / surface_mm2 if surface_mm2 > 0 else 0.0

        # Nearest hot spot
        riser_centroid_vox = np.array(np.argwhere(component_mask).mean(axis=0))
        nearest_hs = None
        nearest_m = 0.0
        nearest_pos = np.zeros(3)
        if hotspots:
            hs_positions_vox = np.array([(hs.position_mm - origin_mm) / dx for hs in hotspots])
            tree = cKDTree(hs_positions_vox.astype(np.float32))
            d, idx = tree.query(riser_centroid_vox.astype(np.float32), k=1)
            nearest_hs = hotspots[idx]
            nearest_m = nearest_hs.m_value_mm
            nearest_pos = nearest_hs.position_mm

        m_required = riser_m_factor * nearest_m if nearest_m > 0 else 0.0
        large_enough = m_riser >= m_required if m_required > 0 else True

        # Feed region: sphere radius 2*M_hotspot around nearest hot spot
        if nearest_hs is not None:
            radius_mm = 2.0 * nearest_m
            radius_vox = radius_mm / dx
            hs_vox = (nearest_hs.position_mm - origin_mm) / dx
            # Count part voxels inside the sphere
            sphere_kernel = np.ones((3, 3, 3), dtype=bool)
            # Approximation: use binary sphere via distance transform on a local sub-box
            feed_region = _sphere_mask(grid.shape, hs_vox, radius_vox) & part_mask
            feed_volume_mm3 = float(feed_region.sum()) * dx ** 3
            volume_ratio_ok = volume_mm3 >= riser_volume_factor * feed_volume_mm3
        else:
            feed_volume_mm3 = 0.0
            volume_ratio_ok = True

        riser_results.append(
            RiserResult(
                body_index=body.index,
                name=body.name,
                volume_cm3=volume_cm3,
                surface_area_cm2=surface_mm2 / 100.0,
                m_value_mm=m_riser,
                target_hotspot_m_mm=nearest_m,
                large_enough=large_enough,
                volume_ratio_ok=volume_ratio_ok,
                nearest_hotspot_position_mm=nearest_pos,
            )
        )

    # AŞAMA 7: Porosity risk map
    with np.errstate(divide="ignore", invalid="ignore"):
        risk = np.where(
            is_metal,
            sdf / (dist_feed + 5.0),
            0.0,
        )
    risk = np.nan_to_num(risk, nan=0.0, posinf=0.0, neginf=0.0)
    rmin, rmax = risk.min(), risk.max()
    if rmax > rmin:
        risk_norm = (risk - rmin) / (rmax - rmin)
    else:
        risk_norm = risk

    if progress_callback:
        progress_callback(95)

    result = AnalysisResult(
        grid=grid,
        origin_mm=origin_mm,
        dx_mm=dx,
        is_metal=is_metal,
        sdf=sdf,
        dist_to_riser=dist_feed,
        risk=risk_norm,
        hotspots=hotspots,
        riser_results=riser_results,
        gate_result=None,
    )

    result.recommendations = _build_recommendations(result, material_factor)
    return result


def _sphere_mask(shape: Tuple[int, int, int], center: np.ndarray, radius_vox: float) -> np.ndarray:
    """Return a boolean spherical mask inside the grid."""
    z, y, x = np.indices(shape, dtype=np.float64)
    z -= center[0]
    y -= center[1]
    x -= center[2]
    return z * z + y * y + x * x <= (radius_vox * radius_vox)


def _build_recommendations(result: AnalysisResult, material_factor: float) -> List[str]:
    recs: List[str] = []

    if not result.hotspots:
        recs.append("Kritik sıcak nokta (hot spot) tespit edilmedi. Model çok ince veya geometri düzgün okunamamış olabilir.")
        return recs

    for hs in result.hotspots:
        if np.isinf(hs.dist_to_riser_mm):
            recs.append(
                f"Hot spot {np.round(hs.position_mm, 1)} (M={hs.m_value_mm:.1f} mm): "
                f"Hiç besleyiciye ulaşamıyor. Kırmızı bölgeye besleyici ekleyin."
            )
        elif not hs.feed_ok:
            recs.append(
                f"Hot spot {np.round(hs.position_mm, 1)} (M={hs.m_value_mm:.1f} mm): "
                f"besleyici mesafesi {hs.dist_to_riser_mm:.1f} mm > limit {material_factor * hs.m_value_mm:.1f} mm. "
                f"Besleyiciyi kırmızı bölgeye yakın taşıyın."
            )

    for rr in result.riser_results:
        if not rr.large_enough:
            recs.append(
                f"{rr.name}: M_besleyici={rr.m_value_mm:.1f} mm < gerekli {rr.target_hotspot_m_mm * 1.2:.1f} mm. "
                f"Besleyici hacmini artırın."
            )
        if not rr.volume_ratio_ok:
            recs.append(
                f"{rr.name}: hacim yeterliliği düşük (V={rr.volume_cm3:.2f} cm³). "
                f"Hot spot etrafındaki besleme bölgesinin en az %30'u kadar olmalı."
            )

    if not any(not hs.feed_ok for hs in result.hotspots) and all(rr.large_enough for rr in result.riser_results):
        recs.append("Tüm sıcak noktalar besleyici menzili içinde ve besleyici boyutları yeterli görünüyor.")

    return recs
