"""SDF-based geometric casting analyzer - JoseCast v7.1."""

from typing import List, Optional, Tuple

import numpy as np
import trimesh
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from sklearn.cluster import DBSCAN

from core.materials import (
    Alloy,
    MoldMaterial,
    chvorinov_c_from_properties,
    get_alloy,
    get_mold,
)
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


def compute_chvorinov_t(
    sdf: np.ndarray, mold: MoldMaterial, use_formula: bool = False
) -> np.ndarray:
    """Solidification time t_s = C * M^2 (s)."""
    if use_formula:
        C = chvorinov_c_from_properties(Alloy(), mold)
    else:
        C = mold.chvorinov_c
    return C * sdf * sdf


def compute_niyama(
    G: np.ndarray,
    t_s: np.ndarray,
    sdf: np.ndarray,
    alloy: Alloy,
) -> np.ndarray:
    """
    Niyama = G / sqrt(R) where R = 1 / sqrt(t_s).

    With t_s = C * M^2 this becomes G * (C * M^2)^(1/4).  To keep the
    criterion material/mould independent we normalize by C^(1/4), giving
    an equivalent geometric Niyama number of G * sqrt(M).  This still
    follows the user's form while making the 0.775 / 1.5 thresholds usable.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        niyama = G * np.sqrt(np.maximum(sdf, 1e-12))
        niyama = np.nan_to_num(niyama, nan=0.0, posinf=0.0, neginf=0.0)
    return niyama


def _sdf_histogram(sdf: np.ndarray, mask: np.ndarray, dx: float, bins: int = 50):
    """Histogram of SDF values inside mask; return peak M and wall thickness."""
    vals = sdf[mask]
    if len(vals) == 0:
        return np.zeros(bins), np.linspace(0, 1, bins + 1), 0.0
    vmax = float(vals.max())
    hist, edges = np.histogram(vals, bins=bins, range=(0.0, vmax + 1e-6))
    # The raw histogram peak is dominated by surface voxels, so the
    # representative wall-thickness modulus is the median of the interior.
    interior = vals[vals > dx * 2]
    dominant_m = float(np.median(interior)) if len(interior) else float(np.median(vals))
    return hist, edges, dominant_m


def _local_section_thickness(
    sdf: np.ndarray,
    part_mask: np.ndarray,
    center_vox: np.ndarray,
    hotspot_m: float,
    dx: float,
) -> float:
    """Estimate local wall thickness (mm) around a hot spot via SDF median."""
    radius_vox = max(3.0 * hotspot_m / dx, 5.0)
    local_mask = _sphere_mask(part_mask.shape, center_vox, radius_vox) & part_mask
    vals = sdf[local_mask]
    if len(vals) == 0:
        return 2.0 * hotspot_m
    interior = vals[vals > dx * 2]
    m_local = float(np.median(interior)) if len(interior) else float(np.median(vals))
    return 2.0 * max(m_local, hotspot_m * 0.5)


def find_hotspots(
    sdf: np.ndarray,
    part_mask: np.ndarray,
    dx: float,
    origin_mm: np.ndarray,
    use_skeleton: bool = True,
    min_size_mm: float = 2.0,
    cluster_eps_mm: float = 10.0,
) -> List[HotSpot]:
    """Detect SDF local maxima on the medial axis and cluster with DBSCAN."""
    if use_skeleton:
        try:
            skeleton = skeletonize(part_mask)
            search_mask = skeleton & (sdf > min_size_mm)
        except Exception:
            search_mask = part_mask & (sdf > min_size_mm)
    else:
        search_mask = part_mask & (sdf > min_size_mm)

    if not search_mask.any():
        return []

    size_vox = max(1, int(5.0 / dx))
    local_max = (sdf == ndimage.maximum_filter(sdf, size=size_vox, mode="constant"))
    candidates = np.argwhere(local_max & search_mask)
    if len(candidates) == 0:
        candidates = np.argwhere(search_mask)

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


def _trace_path_to_riser(
    dist_to_riser: np.ndarray,
    part_mask: np.ndarray,
    start_vox: np.ndarray,
) -> List[Tuple[int, int, int]]:
    """Walk from start_vox toward decreasing distance-to-riser inside part."""
    shape = dist_to_riser.shape
    current = tuple(start_vox)
    if not part_mask[current]:
        return []
    path = [current]
    visited = {current}
    max_steps = shape[0] + shape[1] + shape[2]

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
        visited.add(best)
        path.append(best)
        current = best

    return path


def trace_feeding_resistance(
    sdf: np.ndarray,
    dist_to_riser: np.ndarray,
    part_mask: np.ndarray,
    start_vox: np.ndarray,
) -> float:
    """
    Geometric feeding resistance: R = sum( max(0, (M_hs - M_i) / M_i) ) along path.
    """
    if not part_mask[start_vox[0], start_vox[1], start_vox[2]]:
        return 0.0
    m_hs = float(sdf[start_vox[0], start_vox[1], start_vox[2]])
    if m_hs <= 0:
        return 0.0

    path = _trace_path_to_riser(dist_to_riser, part_mask, start_vox)
    if not path:
        return 0.0

    resistance = 0.0
    for vox in path[1:]:
        m_i = float(sdf[vox[0], vox[1], vox[2]])
        if m_i > 0:
            resistance += max(0.0, (m_hs - m_i) / m_i)
    return resistance


def _path_darcy_and_directional(
    sdf: np.ndarray,
    t_s: np.ndarray,
    dist_to_riser: np.ndarray,
    part_mask: np.ndarray,
    start_vox: np.ndarray,
    dx: float,
    alloy: Alloy,
    mold: MoldMaterial,
) -> Tuple[float, float, float, bool]:
    """
    Darcy pressure integral, minimum neck M and directional solidification
    along the feeding path.
    Returns (darcy_resistance, min_neck_m, t_s_at_hotspot, directional_ok).
    """
    path = _trace_path_to_riser(dist_to_riser, part_mask, start_vox)
    if not path:
        return 0.0, 0.0, 0.0, True

    m_path = np.array([sdf[v] for v in path])
    t_path = np.array([t_s[v] for v in path])
    t_hs = float(t_path[0])

    min_neck_m = float(m_path.min())

    # Darcy-style pressure drop: proportional to dx / (M^2 * K)
    visc = max(alloy.viscosity_proxy, 1e-6)
    perm = max(mold.permeability_proxy, 1e-6)
    darcy = 0.0
    for i in range(len(path) - 1):
        m_i = max(m_path[i], 0.5)
        darcy += dx / (m_i * m_i * perm) * visc

    directional_ok = True
    if len(t_path) > 2:
        window = min(5, len(t_path) // 2 + 1)
        if window > 1:
            weights = np.ones(window) / window
            t_smooth = np.convolve(t_path, weights, mode="same")
        else:
            t_smooth = t_path
        max_t = float(t_smooth.max())
        if max_t > 0:
            cutoff = int(0.15 * len(t_smooth)) + 1
            tail = t_smooth[cutoff:]
            if len(tail) and float(tail.min()) < 0.55 * max_t:
                directional_ok = False

    return darcy, min_neck_m, t_hs, directional_ok


def _sphere_mask(
    shape: Tuple[int, int, int], center: np.ndarray, radius_vox: float
) -> np.ndarray:
    """Boolean spherical mask inside the grid."""
    z, y, x = np.indices(shape, dtype=np.float64)
    z -= center[0]
    y -= center[1]
    x -= center[2]
    return z * z + y * y + x * x <= (radius_vox * radius_vox)


def _ingate_contact_m(
    grid: np.ndarray, sdf: np.ndarray, part_mask: np.ndarray, dx: float
) -> float:
    """Average SDF (modulus) of part voxels touching an ingate."""
    ingate = grid == BodyType.INGATE
    if not ingate.any():
        return 0.0
    touch = np.zeros_like(part_mask)
    for di, dj, dk in [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]:
        rolled = np.roll(ingate, (di, dj, dk), axis=(0, 1, 2))
        if di > 0:
            rolled[-1, :, :] = False
        elif di < 0:
            rolled[0, :, :] = False
        if dj > 0:
            rolled[:, -1, :] = False
        elif dj < 0:
            rolled[:, 0, :] = False
        if dk > 0:
            rolled[:, :, -1] = False
        elif dk < 0:
            rolled[:, :, 0] = False
        touch |= rolled & part_mask
    vals = sdf[touch]
    if len(vals) == 0:
        return 0.0
    return float(vals.mean())


def _refine_region(
    bodies: List[Body],
    hotspot: HotSpot,
    hotspot_index: int,
    coarse_origin: np.ndarray,
    coarse_dx: float,
    alloy: Alloy,
    mold: MoldMaterial,
    base_res: int,
    max_res: int,
    progress_callback: Optional[callable] = None,
) -> Optional[RefinementRegion]:
    """Create a high-resolution local grid around a hot spot."""
    from core.voxelizer import build_voxel_grid

    m = hotspot.m_value_mm
    half = max(4 * m, 3 * coarse_dx)
    local_min = hotspot.position_mm - half
    local_max = hotspot.position_mm + half
    local_size = local_max - local_min
    max_size = float(local_size.max())

    desired_dx = max_size / max_res
    # Cap local dense grid to 384³ to avoid RAM explosion while still giving
    # the user a 2040-ready resolution dial for future sparse implementations.
    max_local_dim = min(max_res, 384)
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
        [BodyType.PART, BodyType.RISER, BodyType.INGATE, BodyType.RUNNER, BodyType.SPRUE],
    )
    sdf = compute_sdf(is_metal, dx)
    t_s = compute_chvorinov_t(sdf, mold)
    gx, gy, gz = np.gradient(sdf, dx)
    G = np.sqrt(gx * gx + gy * gy + gz * gz)
    niyama = compute_niyama(G, t_s, sdf, alloy)

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
    alloy_key: str = "42CrMo4",
    mold_key: str = "sand",
    base_res: int = 160,
    max_res: int = 2040,
    refine_local: bool = True,
    progress_callback: Optional[callable] = None,
) -> AnalysisResult:
    """Run the full v7.1 geometric analysis pipeline."""
    if progress_callback:
        progress_callback(5)

    alloy = get_alloy(alloy_key)
    mold = get_mold(mold_key)
    chvorinov_c = mold.chvorinov_c
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

    # AŞAMA 2: SDF + histogram
    sdf = compute_sdf(is_metal, dx)
    if progress_callback:
        progress_callback(20)

    _, _, dominant_m = _sdf_histogram(sdf, part_mask, dx, bins=50)
    wall_thickness = 2.0 * dominant_m if dominant_m > 0 else 0.0

    # AŞAMA 3: Chvorinov solidification time + 3-iter Laplacian smoothing
    sdf_s = laplacian_smooth(sdf, iterations=3, sigma=1.0)
    t_s = compute_chvorinov_t(sdf_s, mold)
    if progress_callback:
        progress_callback(40)

    # AŞAMA 4: Niyama
    gx, gy, gz = np.gradient(sdf_s, dx)
    G = np.sqrt(gx * gx + gy * gy + gz * gz)
    niyama = compute_niyama(G, t_s, sdf_s, alloy)
    if progress_callback:
        progress_callback(55)

    # AŞAMA 5: Hot spot detection (medial axis / skeleton + DBSCAN)
    hotspots = find_hotspots(sdf, part_mask, dx, origin_mm, use_skeleton=True)
    if progress_callback:
        progress_callback(65)

    # AŞAMA 6: 26-neighbor Dijkstra feeding distance
    dist_feed = feeding_distance_dijkstra(is_metal, riser_mask, dx)
    if progress_callback:
        progress_callback(75)

    # Update hotspot physics: feeding, resistance, Niyama, Darcy, directional
    riser_voxels = np.argwhere(riser_mask)
    for hs in hotspots:
        vox = np.round((hs.position_mm - origin_mm) / dx).astype(int)
        if 0 <= vox[0] < grid.shape[0] and 0 <= vox[1] < grid.shape[1] and 0 <= vox[2] < grid.shape[2]:
            hs.dist_to_riser_mm = float(dist_feed[vox[0], vox[1], vox[2]])
            hs.niyama_min = float(niyama[vox[0], vox[1], vox[2]])
            hs.local_sdf_max = float(sdf[vox[0], vox[1], vox[2]])
            hs.resistance = trace_feeding_resistance(sdf, dist_feed, part_mask, vox)

            darcy, min_neck_m, t_hs, directional_ok = _path_darcy_and_directional(
                sdf, t_s, dist_feed, part_mask, vox, dx, alloy, mold
            )
            hs.darcy_resistance = darcy
            hs.min_neck_m = min_neck_m
            hs.directional_ok = directional_ok

            # Prefer ingate-contact thickness, otherwise local wall thickness.
            ingate_contact = _ingate_contact_m(grid, sdf, part_mask, dx)
            if ingate_contact > 0:
                t_section = 2.0 * ingate_contact
            else:
                t_section = _local_section_thickness(
                    sdf, part_mask, vox, hs.m_value_mm, dx
                )
            # At minimum, use the local hot-spot modulus so a thick boss is not
            # unfairly penalised by a globally thin wall.
            t_section = max(t_section, 2.0 * hs.m_value_mm * 0.5)
            hs.t_section_mm = t_section

            if len(riser_voxels) > 0:
                riser_positions_mm = riser_voxels * dx + origin_mm
                dz = riser_positions_mm[:, 2] - hs.position_mm[2]
                closest_riser_idx = int(
                    np.argmin(np.linalg.norm(riser_positions_mm - hs.position_mm, axis=1))
                )
                dz_closest = dz[closest_riser_idx]
                hs.resistance *= max(0.7, 1.0 - 0.003 * max(0, -dz_closest))
                hs.gravity_factor = 1.0 + 0.3 * max(0.0, dz_closest / max(hs.dist_to_riser_mm, 1.0))
            else:
                dz_closest = 0.0
                hs.gravity_factor = 1.0

            hs.resistance_ok = hs.resistance <= 80.0 and hs.directional_ok

            allowed = alloy.feed_factor * t_section
            hs.max_feeding_distance_mm = allowed * hs.gravity_factor
            hs.feed_ok = (not np.isinf(hs.dist_to_riser_mm)) and (
                hs.dist_to_riser_mm <= hs.max_feeding_distance_mm
            )
        else:
            hs.feed_ok = False

    if progress_callback:
        progress_callback(85)

    # AŞAMA 7: Riser sufficiency
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

        m_required = alloy.riser_m_factor * nearest_m
        riser_z_mm = float((riser_centroid_vox[2] * dx) + origin_mm[2])
        dz = (riser_z_mm - nearest_pos[2]) if nearest_hs is not None else 0.0
        gravity = max(0.85, 1.0 - 0.005 * max(0, -dz))
        effective_m_required = m_required * gravity
        large_enough = m_riser >= effective_m_required if m_required > 0 else True

        required_volume_cm3 = 0.0
        volume_ratio_ok = True
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
            required_volume_cm3 = alloy.riser_volume_factor * feed_volume_mm3 / 1000.0
            volume_ratio_ok = volume_cm3 >= required_volume_cm3

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
                required_volume_cm3=required_volume_cm3,
            )
        )

    if progress_callback:
        progress_callback(90)

    # AŞAMA 8: Risk map
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

    # AŞAMA 9: Local refinement around hot spots
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
                alloy,
                mold,
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
        solidification_time=t_s,
        niyama=niyama,
        gradient_magnitude=G,
        hotspots=hotspots,
        riser_results=riser_results,
        gate_result=None,
        local_regions=local_regions,
        alloy_key=alloy_key,
        mold_key=mold_key,
        alloy_name=alloy.name,
        mold_name=mold.name,
        chvorinov_c=chvorinov_c,
        unit_scale=1.0,
        dominant_m_mm=dominant_m,
        wall_thickness_mm=wall_thickness,
        bbox_size_mm=bbox_size,
    )

    result.recommendations = _build_recommendations(result, alloy, mold)
    return result


def _build_recommendations(
    result: AnalysisResult, alloy: Alloy, mold: MoldMaterial
) -> List[str]:
    recs: List[str] = []

    recs.append(
        f"Malzeme: {alloy.name} | Kalıp: {mold.name} | Chvorinov C = {result.chvorinov_c:.3f} s/mm² | "
        f"Baskın duvar kalınlığı modülü M = {result.dominant_m_mm:.2f} mm (t ≈ {result.wall_thickness_mm:.2f} mm)"
    )

    if not result.hotspots:
        recs.append(
            "Kritik sıcak nokta (hot spot) tespit edilmedi. Model çok ince veya geometri düzgün okunamamış olabilir."
        )
        return recs

    for hs in result.hotspots:
        t = hs.t_section_mm
        fd = alloy.feed_factor * t
        if np.isinf(hs.dist_to_riser_mm):
            recs.append(
                f"Hot spot (M={hs.m_value_mm:.1f} mm, t={t:.1f} mm) hiç besleyiciye ulaşamıyor. "
                f"FD={fd:.1f} mm. Kırmızı bölgeye besleyici ekleyin."
            )
        elif not hs.feed_ok:
            recs.append(
                f"Hot spot (M={hs.m_value_mm:.1f} mm, t={t:.1f} mm): besleme mesafesi {hs.dist_to_riser_mm:.1f} mm "
                f"> limit {hs.max_feeding_distance_mm:.1f} mm (FD={fd:.1f} mm). Besleyiciyi yakın taşı veya kesiti büyütün."
            )

        if not hs.directional_ok:
            recs.append(
                f"Hot spot (M={hs.m_value_mm:.1f} mm): besleme yolunda soğuk nokta/daralma (boyun M={hs.min_neck_m:.1f} mm). "
                f"Yol boyunca kalınlık azalmamalı."
            )

        if hs.darcy_resistance > 100.0:
            recs.append(
                f"Hot spot (M={hs.m_value_mm:.1f} mm): Darcy besleme direnci yüksek ({hs.darcy_resistance:.1f}). "
                f"Meme/yol kesitini büyütün."
            )

        if hs.niyama_min < alloy.niyama_macro:
            recs.append(
                f"Hot spot (M={hs.m_value_mm:.1f} mm): Niyama {hs.niyama_min:.2f} < {alloy.niyama_macro} -> "
                f"makro shrinkage / çekinti riski çok yüksek."
            )
        elif hs.niyama_min < alloy.niyama_shrinkage:
            recs.append(
                f"Hot spot (M={hs.m_value_mm:.1f} mm): Niyama {hs.niyama_min:.2f} < {alloy.niyama_shrinkage} -> "
                f"mikro gözenek / shrinkage porozite riski."
            )

    for rr in result.riser_results:
        if not rr.large_enough:
            increase = (
                (rr.effective_m_required / max(rr.m_value_mm, 1e-6) - 1.0) * 100.0
            )
            recs.append(
                f"{rr.name}: M_besleyici={rr.m_value_mm:.1f} mm < gerekli {rr.effective_m_required:.1f} mm. "
                f"Besleyici modülünü %{int(increase)} büyütün."
            )
        if not rr.volume_ratio_ok:
            short = rr.required_volume_cm3 - rr.volume_cm3
            recs.append(
                f"{rr.name}: hacim yetersiz (V={rr.volume_cm3:.2f} cm³, gerekli {rr.required_volume_cm3:.2f} cm³). "
                f"En az {short:.2f} cm³ daha hacim ekleyin."
            )

    if not any(not hs.feed_ok for hs in result.hotspots) and all(
        rr.large_enough for rr in result.riser_results
    ):
        recs.append(
            "Tüm sıcak noktalar besleyici menzili içinde ve besleyici boyutları yeterli görünüyor."
        )

    return recs
