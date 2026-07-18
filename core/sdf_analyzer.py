"""SDF-based geometric casting analyzer - JoseCast v7 Titan."""

from typing import List, Optional, Tuple

import numpy as np
import trimesh
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from core.materials import Material, get_material
from core.types import (
    AnalysisResult,
    Body,
    BodyType,
    HotSpot,
    RefinementRegion,
    RiserResult,
)


def _make_neighbors_26() -> Tuple[np.ndarray, np.ndarray]:
    """26-neighbor directions and voxel-center Euclidean costs."""
    neigh = []
    costs = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if di == dj == dk == 0:
                    continue
                d = np.sqrt(di * di + dj * dj + dk * dk)
                neigh.append((di, dj, dk))
                costs.append(d)
    return np.array(neigh, dtype=np.int32), np.array(costs, dtype=np.float64)


NEIGH_26, COST_26 = _make_neighbors_26()


def laplacian_smooth(
    field: np.ndarray, iterations: int = 3, sigma: float = 1.0
) -> np.ndarray:
    """Apply mild Gaussian smoothing repeatedly (3 iterasyon) for heavy physics."""
    out = field.copy()
    for _ in range(iterations):
        out = ndimage.gaussian_filter(out, sigma=sigma, mode="nearest")
    return out


def compute_sdf(is_metal: np.ndarray, dx: float) -> np.ndarray:
    """Return Signed Distance Field inside metal: distance to nearest non-metal voxel."""
    return ndimage.distance_transform_edt(is_metal).astype(np.float64) * dx


def compute_niyama(
    sdf: np.ndarray,
    dx: float,
    material: Material,
    smooth: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Chvorinov solidification time and Niyama criterion."""
    if smooth:
        sdf_s = laplacian_smooth(sdf, iterations=3, sigma=1.0)
    else:
        sdf_s = sdf

    gx, gy, gz = np.gradient(sdf_s, dx)
    G = np.sqrt(gx * gx + gy * gy + gz * gz)

    # Chvorinov: t = K * M^2
    with np.errstate(divide="ignore", invalid="ignore"):
        t_solid = material.chvorinov_k * sdf_s * sdf_s
        t_solid = np.nan_to_num(t_solid, nan=0.0, posinf=0.0, neginf=0.0)
        cooling_rate = 1.0 / (t_solid + 1e-6)
        niyama = G / np.sqrt(cooling_rate)
        niyama = np.nan_to_num(niyama, nan=0.0, posinf=0.0, neginf=0.0)
    return niyama, G, t_solid


def find_hotspots(
    sdf: np.ndarray,
    part_mask: np.ndarray,
    dx: float,
    origin_mm: np.ndarray,
    min_size_mm: float = 2.0,
    cluster_eps_mm: float = 10.0,
) -> List[HotSpot]:
    """Detect SDF local maxima inside the part and cluster with DBSCAN."""
    size_vox = 5
    local_max = (sdf == ndimage.maximum_filter(sdf, size=size_vox, mode="constant"))
    candidates = np.argwhere(local_max & part_mask & (sdf > min_size_mm))
    if len(candidates) == 0:
        return []

    eps_vox = max(1.0, cluster_eps_mm / dx)
    clustering = DBSCAN(eps=eps_vox, min_samples=1, metric="euclidean").fit(
        candidates.astype(np.float64)
    )
    labels = clustering.labels_

    hotspots: List[HotSpot] = []
    for lbl in set(labels):
        if lbl == -1:
            continue
        pts = candidates[labels == lbl]
        vals = sdf[pts[:, 0], pts[:, 1], pts[:, 2]]
        idx = int(np.argmax(vals))
        pos_vox = pts[idx]
        m_value = float(vals[idx])
        position_mm = origin_mm + pos_vox * dx
        hotspots.append(
            HotSpot(
                position_mm=position_mm,
                m_value_mm=m_value,
                dist_to_riser_mm=np.inf,
                feed_ok=False,
                max_feeding_distance_mm=0.0,
            )
        )
    return hotspots


def feeding_distance_dijkstra(
    is_metal: np.ndarray, riser_mask: np.ndarray, dx: float
) -> np.ndarray:
    """
    26-neighbor weighted Dijkstra distance within metal to the nearest riser.
    Uses scipy.sparse.csgraph.dijkstra with a virtual source node for speed.
    """
    dist = np.full(is_metal.shape, np.inf, dtype=np.float64)
    if not (is_metal & riser_mask).any():
        return dist

    # Map metal voxels to a flat graph
    idx = np.full(is_metal.shape, -1, dtype=np.int64)
    metal_vox = np.argwhere(is_metal)
    n = int(metal_vox.shape[0])
    idx[tuple(metal_vox.T)] = np.arange(n)

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    vals: List[np.ndarray] = []

    for (di, dj, dk), c in zip(NEIGH_26, COST_26):
        ni = metal_vox[:, 0] + di
        nj = metal_vox[:, 1] + dj
        nk = metal_vox[:, 2] + dk
        mask = (
            (ni >= 0)
            & (ni < is_metal.shape[0])
            & (nj >= 0)
            & (nj < is_metal.shape[1])
            & (nk >= 0)
            & (nk < is_metal.shape[2])
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

    # Virtual source connected to every riser voxel with zero weight
    riser_flat = np.where(riser_mask[tuple(metal_vox.T)])[0]
    if len(riser_flat) == 0:
        return dist
    rows.append(np.full(len(riser_flat), n, dtype=np.int64))
    cols.append(riser_flat.astype(np.int64))
    vals.append(np.zeros(len(riser_flat), dtype=np.float32))

    rows_arr = np.concatenate(rows)
    cols_arr = np.concatenate(cols)
    vals_arr = np.concatenate(vals)

    graph = sparse.coo_matrix(
        (vals_arr, (rows_arr, cols_arr)), shape=(n + 1, n + 1)
    ).tocsr()

    flat_dist = csgraph.dijkstra(
        graph, directed=False, indices=n, return_predecessors=False
    )
    dist[tuple(metal_vox.T)] = flat_dist[:n].astype(np.float64)
    return dist


def trace_feeding_resistance(
    sdf: np.ndarray,
    dist_to_riser: np.ndarray,
    part_mask: np.ndarray,
    start_vox: np.ndarray,
) -> float:
    """
    Walk from hot-spot voxel toward the nearest riser using the distance field.
    R = sum( max(0, (M_hs - M_i) / M_i) ). R > 80 => gate frozen risk.
    """
    if not part_mask[start_vox[0], start_vox[1], start_vox[2]]:
        return 0.0

    m_hs = float(sdf[start_vox[0], start_vox[1], start_vox[2]])
    if m_hs <= 0:
        return 0.0

    shape = sdf.shape
    current = tuple(start_vox)
    visited = {current}
    resistance = 0.0
    max_steps = sdf.shape[0] + sdf.shape[1] + sdf.shape[2]

    for _ in range(max_steps):
        i, j, k = current
        if dist_to_riser[i, j, k] <= 0:
            break

        best = None
        best_d = dist_to_riser[i, j, k]
        for di, dj, dk in [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]:
            ni, nj, nk = i + di, j + dj, k + dk
            if not (0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]):
                continue
            if not part_mask[ni, nj, nk]:
                continue
            d = dist_to_riser[ni, nj, nk]
            if d < best_d:
                best_d = d
                best = (ni, nj, nk)
        if best is None or best in visited:
            break

        m_i = float(sdf[best[0], best[1], best[2]])
        if m_i > 0:
            resistance += max(0.0, (m_hs - m_i) / m_i)
        visited.add(best)
        current = best

    return resistance


def _sphere_mask(
    shape: Tuple[int, int, int], center: np.ndarray, radius_vox: float
) -> np.ndarray:
    """Boolean spherical mask inside the grid."""
    z, y, x = np.indices(shape, dtype=np.float64)
    z -= center[0]
    y -= center[1]
    x -= center[2]
    return z * z + y * y + x * x <= (radius_vox * radius_vox)


def _refine_region(
    bodies: List[Body],
    hotspot: HotSpot,
    hotspot_index: int,
    coarse_origin: np.ndarray,
    coarse_dx: float,
    base_res: int,
    max_res: int,
    progress_callback: Optional[callable] = None,
) -> Optional[RefinementRegion]:
    """
    Create a high-resolution local grid around a hot spot.
    The resolution factor is chosen so the local box does not explode in RAM.
    """
    from core.voxelizer import build_voxel_grid

    m = hotspot.m_value_mm
    # Local box = 8 * M around the hot spot
    half = max(4 * m, 3 * coarse_dx)
    local_min = hotspot.position_mm - half
    local_max = hotspot.position_mm + half
    local_size = local_max - local_min
    max_size = float(local_size.max())

    desired_dx = max_size / max_res
    max_local_dim = 384
    dx_fine = max(desired_dx, max_size / max_local_dim)

    cropped_bodies: List[Body] = []
    for b in bodies:
        bmin = b.mesh.bounds[0]
        bmax = b.mesh.bounds[1]
        if np.any(bmax < local_min - 1.0) or np.any(bmin > local_max + 1.0):
            continue
        box = trimesh.creation.box(
            extents=local_size,
            transform=trimesh.transformations.translation_matrix(hotspot.position_mm),
        )
        try:
            cropped = (
                b.mesh.intersection(box, engine="scad")
                if hasattr(b.mesh, "intersection")
                else b.mesh
            )
        except Exception:
            cropped = b.mesh
        if cropped is None or len(cropped.faces) == 0:
            continue
        cb = Body(
            index=b.index,
            name=b.name,
            vertices=cropped.vertices,
            faces=cropped.faces,
            mesh=cropped,
            body_type=b.body_type,
            volume_cm3=cropped.volume / 1000.0,
            center=cropped.center_mass
            if cropped.is_watertight
            else cropped.centroid,
        )
        cropped_bodies.append(cb)

    if not cropped_bodies:
        return None

    target_dim = max(32, int(max_size / dx_fine))
    grid, origin, dx, _ = build_voxel_grid(
        cropped_bodies,
        target_dim=target_dim,
        progress_callback=progress_callback,
    )
    is_metal = np.isin(
        grid,
        [
            BodyType.PART,
            BodyType.RISER,
            BodyType.INGATE,
            BodyType.RUNNER,
            BodyType.SPRUE,
        ],
    )
    sdf = compute_sdf(is_metal, dx)
    niyama, G, t_solid = compute_niyama(sdf, dx, get_material("steel"), smooth=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        risk = 1.0 / (niyama + 0.2)
        risk = np.nan_to_num(risk, nan=0.0, posinf=0.0, neginf=0.0)
        rmin, rmax = risk.min(), risk.max()
        if rmax > rmin:
            risk = (risk - rmin) / (rmax - rmin)
        else:
            risk = risk * 0.0

    return RefinementRegion(
        hotspot_index=hotspot_index,
        origin_mm=origin,
        dx_mm=dx,
        grid=grid,
        sdf=sdf,
        niyama=niyama,
        risk=risk,
    )


def analyze(
    bodies: List[Body],
    grid: np.ndarray,
    origin_mm: np.ndarray,
    dx: float,
    material_key: str = "steel",
    base_res: int = 160,
    max_res: int = 2040,
    refine_local: bool = True,
    progress_callback: Optional[callable] = None,
) -> AnalysisResult:
    """Run the full v7 geometric analysis pipeline."""
    if progress_callback:
        progress_callback(5)

    material = get_material(material_key)
    bbox_size = np.array(grid.shape) * dx

    is_metal = np.isin(
        grid,
        [BodyType.PART, BodyType.RISER, BodyType.INGATE, BodyType.RUNNER, BodyType.SPRUE],
    )
    if is_metal.sum() < 1000:
        raise ValueError(
            "Model çok küçük. Çözünürlüğü artırın veya modelin mm biriminde olduğundan emin olun."
        )

    part_mask = grid == BodyType.PART
    riser_mask = grid == BodyType.RISER

    # AŞAMA 2/3: SDF + Laplacian smoothing for heavy physics
    sdf = compute_sdf(is_metal, dx)
    if progress_callback:
        progress_callback(25)

    niyama, G, t_solid = compute_niyama(sdf, dx, material, smooth=True)
    if progress_callback:
        progress_callback(45)

    # AŞAMA 3: Hot spot detection with DBSCAN
    hotspots = find_hotspots(sdf, part_mask, dx, origin_mm)
    if progress_callback:
        progress_callback(55)

    # AŞAMA 4: 26-neighbor Dijkstra feeding distance
    dist_feed = feeding_distance_dijkstra(is_metal, riser_mask, dx)
    if progress_callback:
        progress_callback(75)

    # Update hotspot feeding distances, resistance and Niyama minima
    riser_voxels = np.argwhere(riser_mask)
    for hs in hotspots:
        vox = np.round((hs.position_mm - origin_mm) / dx).astype(int)
        if 0 <= vox[0] < grid.shape[0] and 0 <= vox[1] < grid.shape[1] and 0 <= vox[2] < grid.shape[2]:
            hs.dist_to_riser_mm = float(dist_feed[vox[0], vox[1], vox[2]])
            hs.niyama_min = float(niyama[vox[0], vox[1], vox[2]])
            hs.local_sdf_max = float(sdf[vox[0], vox[1], vox[2]])
            hs.resistance = trace_feeding_resistance(sdf, dist_feed, part_mask, vox)

            if len(riser_voxels) > 0:
                riser_positions_mm = riser_voxels * dx + origin_mm
                dz = riser_positions_mm[:, 2] - hs.position_mm[2]
                closest_riser_idx = int(
                    np.argmin(np.linalg.norm(riser_positions_mm - hs.position_mm, axis=1))
                )
                dz_closest = dz[closest_riser_idx]
                hs.resistance *= max(0.7, 1.0 - 0.003 * max(0, -dz_closest))

            hs.resistance_ok = hs.resistance <= 80.0
            allowed = material.feed_factor * hs.m_value_mm
            if len(riser_voxels) > 0:
                hs.gravity_factor = 1.0 + 0.3 * max(
                    0.0, dz_closest / max(hs.dist_to_riser_mm, 1.0)
                )
            else:
                hs.gravity_factor = 1.0
            hs.max_feeding_distance_mm = allowed * hs.gravity_factor
            hs.feed_ok = (not np.isinf(hs.dist_to_riser_mm)) and (
                hs.dist_to_riser_mm <= hs.max_feeding_distance_mm
            )
        else:
            hs.feed_ok = False

    if progress_callback:
        progress_callback(85)

    # AŞAMA 5: Riser sufficiency
    riser_results: List[RiserResult] = []
    labeled, num = ndimage.label(riser_mask)
    for body in bodies:
        if body.body_type != BodyType.RISER:
            continue
        body_center_vox = np.round((body.center - origin_mm) / dx).astype(int)
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

        dilated = ndimage.binary_dilation(component_mask, iterations=1)
        surface_mask = dilated & ~component_mask
        surface_mm2 = float(surface_mask.sum()) * dx * dx
        m_riser = volume_mm3 / surface_mm2 if surface_mm2 > 0 else 0.0

        riser_centroid_vox = np.array(np.argwhere(component_mask).mean(axis=0))
        nearest_hs = None
        nearest_m = 0.0
        nearest_pos = np.zeros(3)
        if hotspots:
            hs_positions_vox = np.array(
                [(hs.position_mm - origin_mm) / dx for hs in hotspots]
            )
            tree = cKDTree(hs_positions_vox.astype(np.float32))
            d, idx = tree.query(riser_centroid_vox.astype(np.float32), k=1)
            nearest_hs = hotspots[idx]
            nearest_m = nearest_hs.m_value_mm
            nearest_pos = nearest_hs.position_mm

        m_required = material.riser_m_factor * nearest_m
        riser_z_mm = float((riser_centroid_vox[2] * dx) + origin_mm[2])
        dz = (riser_z_mm - nearest_pos[2]) if nearest_hs is not None else 0.0
        gravity = max(0.85, 1.0 - 0.005 * max(0, -dz))
        effective_m_required = m_required * gravity
        large_enough = m_riser >= effective_m_required if m_required > 0 else True

        if nearest_hs is not None:
            radius_mm = 2.0 * nearest_m
            radius_vox = radius_mm / dx
            feed_region = (
                _sphere_mask(
                    grid.shape, (nearest_hs.position_mm - origin_mm) / dx, radius_vox
                )
                & part_mask
            )
            feed_volume_mm3 = float(feed_region.sum()) * dx ** 3
            volume_ratio_ok = (
                volume_mm3 >= material.riser_volume_factor * feed_volume_mm3
            )
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
                gravity_factor=gravity,
                effective_m_required=effective_m_required,
            )
        )

    if progress_callback:
        progress_callback(90)

    # Risk map: low Niyama and far feeding distance increase risk
    with np.errstate(divide="ignore", invalid="ignore"):
        niyama_risk = 1.0 / (niyama + 0.2)
        feed_risk = dist_feed / (sdf + 5.0)
        risk = niyama_risk * (1.0 + feed_risk)
        risk = np.nan_to_num(risk, nan=0.0, posinf=0.0, neginf=0.0)
    rmin, rmax = risk.min(), risk.max()
    if rmax > rmin:
        risk_norm = (risk - rmin) / (rmax - rmin)
    else:
        risk_norm = risk

    if progress_callback:
        progress_callback(95)

    # Local refinement around hot spots (optional, can be slow)
    local_regions: List[RefinementRegion] = []
    if refine_local and hotspots:
        for idx, hs in enumerate(hotspots):
            if progress_callback:
                progress_callback(95 + int((idx + 1) / len(hotspots) * 5))
            region = _refine_region(
                bodies,
                hs,
                idx,
                origin_mm,
                dx,
                base_res,
                max_res,
                progress_callback,
            )
            if region is not None:
                local_regions.append(region)

    result = AnalysisResult(
        grid=grid,
        origin_mm=origin_mm,
        dx_mm=dx,
        is_metal=is_metal,
        sdf=sdf,
        dist_to_riser=dist_feed,
        risk=risk_norm,
        solidification_time=t_solid,
        niyama=niyama,
        gradient_magnitude=G,
        hotspots=hotspots,
        riser_results=riser_results,
        gate_result=None,
        local_regions=local_regions,
        material_key=material_key,
        material_name=material.display_name,
        bbox_size_mm=bbox_size,
    )

    result.recommendations = _build_recommendations(result, material)
    return result


def _build_recommendations(result: AnalysisResult, material: Material) -> List[str]:
    recs: List[str] = []

    if not result.hotspots:
        recs.append(
            "Kritik sıcak nokta (hot spot) tespit edilmedi. Model çok ince veya geometri düzgün okunamamış olabilir."
        )
        return recs

    for hs in result.hotspots:
        if np.isinf(hs.dist_to_riser_mm):
            recs.append(
                f"Hot spot (M={hs.m_value_mm:.1f} mm) hiç besleyiciye ulaşamıyor. "
                f"Kırmızı bölgeye besleyici ekleyin."
            )
        elif not hs.feed_ok:
            recs.append(
                f"Hot spot (M={hs.m_value_mm:.1f} mm): besleyici mesafesi {hs.dist_to_riser_mm:.1f} mm "
                f"> limit {hs.max_feeding_distance_mm:.1f} mm. Besleyiciyi kırmızı bölgeye yakın taşıyın."
            )
        if not hs.resistance_ok:
            recs.append(
                f"Hot spot (M={hs.m_value_mm:.1f} mm): besleme direnci {hs.resistance:.1f} > 80. "
                f"Meme/yolluk donma riski yüksek, kesiti büyütün veya kısa yol sağlayın."
            )
        if hs.niyama_min < material.niyama_macro:
            recs.append(
                f"Hot spot (M={hs.m_value_mm:.1f} mm): Niyama {hs.niyama_min:.2f} < {material.niyama_macro} -> "
                f"makro shrinkage / çekinti riski çok yüksek."
            )
        elif hs.niyama_min < material.niyama_shrinkage:
            recs.append(
                f"Hot spot (M={hs.m_value_mm:.1f} mm): Niyama {hs.niyama_min:.2f} < {material.niyama_shrinkage} -> "
                f"shrinkage porozite riski."
            )

    for rr in result.riser_results:
        if not rr.large_enough:
            recs.append(
                f"{rr.name}: M_besleyici={rr.m_value_mm:.1f} mm < gerekli {rr.effective_m_required:.1f} mm. "
                f"Besleyici hacmini artırın."
            )
        if not rr.volume_ratio_ok:
            recs.append(
                f"{rr.name}: hacim yeterliliği düşük (V={rr.volume_cm3:.2f} cm³). "
                f"Hot spot etrafındaki besleme bölgesinin en az %{int(material.riser_volume_factor * 100)}'u kadar olmalı."
            )

    if not any(not hs.feed_ok for hs in result.hotspots) and all(
        rr.large_enough for rr in result.riser_results
    ):
        recs.append(
            "Tüm sıcak noktalar besleyici menzili içinde ve besleyici boyutları yeterli görünüyor."
        )

    return recs
