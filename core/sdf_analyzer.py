"""SDF-based geometric + pseudo-thermal casting analyzer - JoseCast v8.0."""

from dataclasses import replace
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree
from scipy.special import erf
from skimage.measure import marching_cubes
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
    CastingParameters,
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
NEIGH_6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def laplacian_smooth(
    field: np.ndarray, iterations: int = 3, sigma: float = 1.0
) -> np.ndarray:
    """Mild Gaussian smoothing (3 iteration = heavy physics feel)."""
    out = field.copy()
    for _ in range(iterations):
        out = ndimage.gaussian_filter(out, sigma=sigma, mode="nearest")
    return out


def compute_sdf(is_metal: np.ndarray, dx: float) -> np.ndarray:
    """Binary SDF: distance inside metal to nearest non-metal voxel."""
    return ndimage.distance_transform_edt(is_metal).astype(np.float64) * dx


def compute_subvoxel_sdf(
    is_metal: np.ndarray, dx: float, sub: int = 2
) -> np.ndarray:
    """
    Upsample the binary occupancy with linear interpolation, run EDT on the
    high-resolution grid and downsample to obtain a sub-voxel SDF.
    """
    if sub <= 1:
        return compute_sdf(is_metal, dx)
    zoom = float(sub)
    # Linear interpolation of 0/1 gives partial (0..1) boundary voxels.
    fine = ndimage.zoom(is_metal.astype(np.float64), zoom, order=1, mode="nearest")
    fine = (fine > 0.5).astype(np.uint8)
    fine_sdf = ndimage.distance_transform_edt(fine).astype(np.float64) * (dx / zoom)
    # Downsample by averaging (order=1) to keep smooth sub-voxel values.
    return ndimage.zoom(fine_sdf, 1.0 / zoom, order=1, mode="nearest")


def compute_curvature(sdf: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and Gaussian curvature from the SDF Hessian.
    Mean curvature is approximated by the trace of the Hessian (Laplacian of SDF);
    Gaussian curvature is the determinant of the Hessian.  Both are vectorised
    over the whole grid for speed.
    """
    gz, gy, gx = np.gradient(sdf, dx)
    hzz, hzy, hzx = np.gradient(gz, dx)
    hyz, hyy, hyx = np.gradient(gy, dx)
    hxz, hxy, hxx = np.gradient(gx, dx)

    # Mean curvature = Laplacian of SDF (trace of Hessian)
    mean_curv = hxx + hyy + hzz

    # Determinant of symmetric 3x3 Hessian
    gauss = (
        hxx * (hyy * hzz - hyz * hzy)
        - hxy * (hxy * hzz - hxz * hyz)
        + hxz * (hxy * hzy - hxz * hyy)
    )
    return mean_curv, gauss


def _marching_cubes_surface(
    bodies: List[Body], grid_shape: Tuple[int, int, int], origin: np.ndarray, dx: float
) -> Optional[trimesh.Trimesh]:
    """Create a watertight-ish combined metal surface for distance queries."""
    # Build a high-res label grid (1 = metal, 0 = empty) and run marching cubes.
    label = np.zeros(grid_shape, dtype=np.float64)
    for b in bodies:
        if b.body_type in (BodyType.EMPTY, BodyType.CORE):
            continue
        try:
            vox = trimesh.voxel.creation.voxelize(b.mesh, pitch=dx)
            if vox is None:
                continue
            mat = vox.fill().matrix
            off = (vox.transform[:3, 3] - origin) / dx
            off = np.round(off).astype(int)
            i0 = max(0, off[0])
            i1 = min(grid_shape[0], off[0] + mat.shape[0])
            j0 = max(0, off[1])
            j1 = min(grid_shape[1], off[1] + mat.shape[1])
            k0 = max(0, off[2])
            k1 = min(grid_shape[2], off[2] + mat.shape[2])
            li0 = i0 - off[0]
            lj0 = j0 - off[1]
            lk0 = k0 - off[2]
            region = mat[
                li0 : li0 + (i1 - i0), lj0 : lj0 + (j1 - j0), lk0 : lk0 + (k1 - k0)
            ]
            label[i0:i1, j0:j1, k0:k1][region.astype(bool)] = 1.0
        except Exception:
            continue
    if not label.any():
        return None
    try:
        verts, faces, *_ = marching_cubes(label, level=0.5)
        verts = verts * dx + origin
        if len(faces) == 0:
            return None
        return trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    except Exception:
        return None


def _scheil_fs(T_arr, t_liq, t_sol, k):
    """Vectorised Scheil solid fraction."""
    fs = np.zeros_like(T_arr)
    mask_past = T_arr <= t_sol
    mask_liq = T_arr >= t_liq
    mask_mush = ~(mask_past | mask_liq)
    fs[mask_past] = 1.0
    fs[mask_liq] = 0.0
    if mask_mush.any():
        k = max(k, 1e-6)
        ratio = (t_liq - T_arr[mask_mush]) / (t_liq - t_sol + 1e-9)
        with np.errstate(divide="ignore", invalid="ignore"):
            fs[mask_mush] = 1.0 - np.power(np.clip(ratio, 0.0, 1.0), 1.0 / (k - 1.0))
            fs = np.clip(fs, 0.0, 1.0)
    return fs


def _temperature_from_erf(
    sdf: np.ndarray, t: Union[float, np.ndarray], alloy: Alloy, mold: MoldMaterial
) -> np.ndarray:
    """1-D semi-infinite solution of the Fourier heat equation in the normal direction."""
    alpha = mold.diffusivity_mm2_s
    if alpha <= 0:
        return np.full_like(sdf, alloy.t_pour_c)
    t = np.maximum(np.asarray(t, dtype=np.float64), 1e-9)
    arg = sdf / (2.0 * np.sqrt(alpha * t))
    T = mold.t0_c + (alloy.t_pour_c - mold.t0_c) * erf(arg)
    return np.clip(T, mold.t0_c, alloy.t_pour_c)


def _cooling_rate_from_erf(
    sdf: np.ndarray, t: Union[float, np.ndarray], alloy: Alloy, mold: MoldMaterial
) -> np.ndarray:
    """Time derivative dT/dt of the erf solution (always <= 0 for cooling)."""
    alpha = mold.diffusivity_mm2_s
    if alpha <= 0:
        return np.zeros_like(sdf)
    t = np.maximum(np.asarray(t, dtype=np.float64), 1e-9)
    sqrt_term = np.sqrt(alpha * t)
    arg = sdf / (2.0 * sqrt_term)
    exp = np.exp(-(arg * arg))
    denom = 2.0 * np.sqrt(np.pi * alpha) * (t ** 1.5)
    denom = np.where(denom > 0, denom, 1e-30)
    dTdt = - (alloy.t_pour_c - mold.t0_c) * sdf * exp / denom
    return np.where(sdf > 0, dTdt, 0.0)


def compute_thermal_field(
    grid: np.ndarray,
    is_metal: np.ndarray,
    alloy: Alloy,
    mold: MoldMaterial,
    dx: float,
    n_steps: int = 100,
    progress_callback: Optional[callable] = None,
    sdf: Optional[np.ndarray] = None,
    M_mod: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Analytical 1-D thermal field at the local Chvorinov solidification time:
        T(x,t_s) = T0 + (T_pour - T0) * erf( x / (2 sqrt(alpha*t_s)) )
    x is the signed distance to the nearest surface (SDF). Latent heat enters
    through the Scheil solid fraction. Returns (T, dT/dt, fs, div(∇T)).
    """
    if sdf is None:
        sdf = compute_subvoxel_sdf(is_metal, dx, sub=1)
    C = chvorinov_c_from_properties(alloy, mold)
    if M_mod is None:
        M_field = sdf
    else:
        M_field = M_mod
    t_s_field = np.maximum(compute_chvorinov_t(M_field, C), 1e-9)
    T = _temperature_from_erf(sdf, t_s_field, alloy, mold)
    cooling_rate = -_cooling_rate_from_erf(sdf, t_s_field, alloy, mold)
    solid_fraction = _scheil_fs(
        T, alloy.t_liquidus_c, alloy.t_solidus_c, alloy.partition_coefficient
    )
    thermal_divergence = ndimage.laplace(T) / (dx * dx)
    return T, cooling_rate, solid_fraction, thermal_divergence


def compute_chvorinov_t(M_field: np.ndarray, C: float) -> np.ndarray:
    """
    Chvorinov solidification time: t_s = C * M^2  [s].
    M is the local casting modulus (mm).
    """
    return C * np.maximum(M_field, 0.0) ** 2


def compute_niyama(
    sdf: np.ndarray,
    M_mod: np.ndarray,
    alloy: Alloy,
    mold: MoldMaterial,
    dx: float,
    is_metal: Optional[np.ndarray] = None,
    temperature: Optional[np.ndarray] = None,
    cooling_rate: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Physically-based Niyama criterion N = G / sqrt(R)  [K s^0.5 / mm].
    G is the metal-side temperature gradient required to remove latent +
    superheat, estimated from the Stefan velocity v = M_mod / t_s.  R is the
    Chvorinov cooling rate ΔT_solid / t_s.  The result is weighted by
    f = M_mod / sdf so that bulky, sphere-like regions (f < 1) report lower
    Niyama (higher shrinkage risk) than plates (f ≈ 1).
    """
    C = chvorinov_c_from_properties(alloy, mold)
    t_s = np.maximum(compute_chvorinov_t(M_mod, C), 1e-9)
    # Stefan velocity based on the local shape-corrected modulus [mm/s]
    v_solid = M_mod / t_s
    l_eff = alloy.latent_heat_j_kg + alloy.cp_j_kgk * max(
        alloy.t_pour_c - alloy.t_liquidus_c, 0.0
    )
    # Metal-side gradient required to carry away latent + superheat [K/mm]
    G = np.where(
        sdf > 0,
        alloy.rho_kg_m3 * l_eff * v_solid / (alloy.k_w_mk * 1e6),
        0.0,
    )
    # Cooling rate from Chvorinov [K/s]
    R = np.where(
        sdf > 0,
        (alloy.t_liquidus_c - alloy.t_solidus_c) / t_s,
        0.0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        niyama = G / np.sqrt(np.maximum(R, 1e-12))
    # Shape correction: sphere-like regions (f < 1) get lower Niyama
    shape_factor = M_mod / np.maximum(sdf, 1e-6)
    niyama = niyama * shape_factor
    niyama = np.nan_to_num(niyama, nan=0.0, posinf=0.0, neginf=0.0)
    if is_metal is not None:
        niyama = np.where(is_metal, niyama, 0.0)
        G = np.where(is_metal, G, 0.0)
        R = np.where(is_metal, R, 0.0)
    return G, R, niyama


def compute_niyama_variants(
    niyama: np.ndarray,
    G: np.ndarray,
    R: np.ndarray,
    t_s: np.ndarray,
    alloy: Alloy,
) -> Dict[str, np.ndarray]:
    """
    Four Niyama-related indicators. The physical classical Niyama is kept as-is;
    the others are scaled to a 0..2 range for the report table only.
    """
    eps = 1e-12
    T_ref = (alloy.t_liquidus_c + alloy.t_solidus_c) / 2.0
    raw = {
        "classical": niyama,
        "coarse": G / (
            np.power(np.maximum(R, eps), 0.5)
            * np.sqrt(np.maximum(T_ref, 1.0) / 1000.0)
        ),
        "elbow": G * np.sqrt(np.maximum(t_s, eps)),
        "lcc": G / (R + eps),
    }
    scaled: Dict[str, np.ndarray] = {}
    for key, val in raw.items():
        finite = np.isfinite(val)
        if finite.any():
            p5, p95 = np.percentile(val[finite], [5, 95])
            span = max(p95 - p5, 1e-9)
            scaled[key] = np.clip((val - p5) / span * 2.0, 0.0, 2.0)
        else:
            scaled[key] = np.zeros_like(val)
    return scaled


def compute_niyama_ensemble(niyama: np.ndarray) -> np.ndarray:
    """Return the physical classical Niyama used for decisions."""
    return niyama


def _sdf_histogram(sdf: np.ndarray, mask: np.ndarray, bins: int = 50):
    """Histogram and dominant interior modulus."""
    vals = sdf[mask]
    if len(vals) == 0:
        return np.zeros(bins), np.linspace(0, 1, bins + 1), 0.0
    vmax = float(vals.max())
    hist, edges = np.histogram(vals, bins=bins, range=(0.0, vmax + 1e-6))
    interior = vals[vals > edges[1]]
    dominant_m = float(np.median(interior)) if len(interior) else float(np.median(vals))
    return hist, edges, dominant_m


def _histogram_stats(sdf: np.ndarray, mask: np.ndarray):
    vals = sdf[mask]
    if len(vals) == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    if std > 1e-9:
        skew = float(((vals - mean) ** 3).mean() / (std ** 3))
    else:
        skew = 0.0
    return mean, std, skew


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
    return 2.0 * m_local


def _shape_factor(mask: np.ndarray, dx: float) -> float:
    """SF = V^2 / A^3; sphere gives ~0.0088, plates/rods give smaller."""
    volume = float(mask.sum()) * (dx ** 3)
    dilated = ndimage.binary_dilation(mask, iterations=1)
    surface = dilated & ~mask
    area = float(surface.sum()) * (dx ** 2)
    if area <= 0 or volume <= 0:
        return 0.0
    return (volume ** 2) / (area ** 3)


def find_hotspots(
    sdf: np.ndarray,
    part_mask: np.ndarray,
    dx: float,
    origin_mm: np.ndarray,
    curvature: Optional[np.ndarray] = None,
    use_skeleton: bool = True,
    min_size_mm: float = 2.0,
    cluster_eps_mm: float = 10.0,
) -> List[HotSpot]:
    """Detect M_mod local maxima on the medial axis and cluster with DBSCAN."""
    # Shape-corrected modulus: bulky regions (κ < 0, sphere-like) get larger f
    if curvature is not None:
        shape_factor_field = np.clip(
            1.0 + np.maximum(-curvature * sdf, 0.0), 1.0, 3.0
        )
    else:
        shape_factor_field = np.ones_like(sdf)
    M_mod = sdf / shape_factor_field

    if use_skeleton:
        try:
            skeleton = skeletonize(part_mask)
            search_mask = skeleton & (M_mod > min_size_mm)
            if not search_mask.any():
                search_mask = part_mask & (M_mod > min_size_mm)
        except Exception:
            search_mask = part_mask & (M_mod > min_size_mm)
    else:
        search_mask = part_mask & (M_mod > min_size_mm)

    if not search_mask.any():
        return []

    size_vox = max(1, int(5.0 / dx))
    local_max = M_mod == ndimage.maximum_filter(M_mod, size=size_vox, mode="constant")
    candidates = np.argwhere(local_max & search_mask)
    if len(candidates) == 0:
        candidates = np.argwhere(search_mask)

    # Combine curvature (sharp corners) as additional candidates.
    if curvature is not None:
        high_curv = (np.abs(curvature) > np.percentile(np.abs(curvature[part_mask]), 95)) & part_mask
        extra = np.argwhere(high_curv & (M_mod > min_size_mm))
        if len(extra):
            candidates = np.vstack([candidates, extra]) if len(candidates) else extra

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
        vals = M_mod[pts[:, 0], pts[:, 1], pts[:, 2]]
        idx = int(np.argmax(vals))
        pos_vox = pts[idx]
        m_value = float(vals[idx])
        sdf_val = float(sdf[pos_vox[0], pos_vox[1], pos_vox[2]])
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
    """26-neighbor weighted Dijkstra distance to the nearest riser inside metal."""
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

    graph = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n + 1, n + 1),
    ).tocsr()
    flat_dist = csgraph.dijkstra(graph, directed=False, indices=n, return_predecessors=False)
    dist[tuple(metal_vox.T)] = flat_dist[:n].astype(np.float64)
    return dist


def feeding_cost_dijkstra(
    is_metal: np.ndarray,
    riser_mask: np.ndarray,
    modulus: np.ndarray,
    dx: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    26-neighbor Dijkstra where edge cost = dx / M (section modulus) of the voxel
    being entered.  This yields the lowest-resistance feeding path and a scalar
    feeding-cost field from every metal voxel to the nearest riser.
    Returns (cost_grid, predecessors, metal_vox).
    """
    cost = np.full(is_metal.shape, np.inf, dtype=np.float64)
    pred = np.full(is_metal.shape, -1, dtype=np.int64)
    if not (is_metal & riser_mask).any():
        return cost, pred, np.empty((0, 3), dtype=np.int64)

    idx = np.full(is_metal.shape, -1, dtype=np.int64)
    metal_vox = np.argwhere(is_metal)
    n = int(metal_vox.shape[0])
    idx[tuple(metal_vox.T)] = np.arange(n)

    # Edge cost = Euclidean step factor * dx / max(M_neighbor, 0.1 mm)
    rows, cols, vals = [], [], []
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
        # cost of moving into neighbor
        m_nb = np.clip(modulus[ni[mask][valid], nj[mask][valid], nk[mask][valid]], 0.1, None)
        rows.append(source_idx[valid])
        cols.append(neighbor_idx[valid])
        vals.append((c * dx / m_nb).astype(np.float32))

    riser_flat = np.where(riser_mask[tuple(metal_vox.T)])[0]
    if len(riser_flat) == 0:
        return cost, pred, metal_vox
    rows.append(np.full(len(riser_flat), n, dtype=np.int64))
    cols.append(riser_flat.astype(np.int64))
    vals.append(np.zeros(len(riser_flat), dtype=np.float32))

    graph = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n + 1, n + 1),
    ).tocsr()
    flat_cost, flat_pred = csgraph.dijkstra(
        graph,
        directed=False,
        indices=n,
        return_predecessors=True,
    )
    flat_cost = flat_cost[:n].astype(np.float64)
    cost[tuple(metal_vox.T)] = flat_cost
    # flat_pred is 1D array of length n+1; map valid graph predecessors to flat voxel indices.
    fp = flat_pred[:n]
    pred_values = np.full(n, -1, dtype=np.int64)
    valid = (fp >= 0) & (fp < n)
    if valid.any():
        pv = metal_vox[fp[valid], 0] * is_metal.shape[1] * is_metal.shape[2]
        pv += metal_vox[fp[valid], 1] * is_metal.shape[2]
        pv += metal_vox[fp[valid], 2]
        pred_values[valid] = pv
    pred[tuple(metal_vox.T)] = pred_values
    return cost, pred, metal_vox


def _trace_cost_path(
    start_vox: np.ndarray,
    predecessors: np.ndarray,
    shape: Tuple[int, int, int],
) -> List[Tuple[int, int, int]]:
    """Walk predecessor map from a voxel back to the virtual riser source."""
    def _vox2flat(v):
        return int(v[0] * shape[1] * shape[2] + v[1] * shape[2] + v[2])

    path = []
    flat = _vox2flat(start_vox)
    visited = set()
    while True:
        if flat < 0 or flat in visited:
            break
        visited.add(flat)
        i = flat // (shape[1] * shape[2])
        rem = flat % (shape[1] * shape[2])
        j = rem // shape[2]
        k = rem % shape[2]
        path.append((i, j, k))
        pred = predecessors[i, j, k]
        if pred < 0 or pred == flat:
            break
        flat = pred
    return path


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
        for di, dj, dk in NEIGH_6:
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
    """Geometric feeding resistance along the path to the riser."""
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


def _liquid_fraction_at_time(
    sdf_i: float, t: float, alloy: Alloy, mold: MoldMaterial
) -> float:
    """Scheil liquid fraction at a point with signed distance sdf_i and time t."""
    if t <= 0:
        return 1.0
    T = float(
        _temperature_from_erf(
            np.array([sdf_i]), t, alloy, mold
        )[0]
    )
    if T >= alloy.t_liquidus_c:
        return 1.0
    if T <= alloy.t_solidus_c:
        return 0.0
    fs = _scheil_fs(
        np.array([T]),
        alloy.t_liquidus_c,
        alloy.t_solidus_c,
        alloy.partition_coefficient,
    )[0]
    return max(0.0, min(1.0, 1.0 - float(fs)))


def _path_darcy_and_directional(
    sdf: np.ndarray,
    M_mod: np.ndarray,
    cost_grid: np.ndarray,
    cost_pred: np.ndarray,
    part_mask: np.ndarray,
    start_vox: np.ndarray,
    dx: float,
    alloy: Alloy,
    mold: MoldMaterial,
    feeder_voxels: Optional[np.ndarray] = None,
    temperature: Optional[np.ndarray] = None,
    solid_fraction: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, bool, bool, float, bool]:
    """
    Walk the lowest-resistance feeding path from start_vox to a riser and compute:
      * Darcy pressure drop through the mushy zone (Kozeny-Carman)
      * minimum neck modulus along the part path
      * t_s at the hot spot
      * directional solidification flag
      * Heuver's circle flag
      * total feeding cost
      * darcy_ok flag
    """
    path = _trace_cost_path(start_vox, cost_pred, sdf.shape)
    if not path:
        return 0.0, 0.0, 0.0, True, True, 0.0, True

    C = chvorinov_c_from_properties(alloy, mold)
    m_hot = float(M_mod[start_vox[0], start_vox[1], start_vox[2]])
    t_s_hot = C * m_hot * m_hot
    hot_sdf = float(sdf[start_vox[0], start_vox[1], start_vox[2]])

    # Hydrostatic head from feeders above the hot spot (if any)
    P_head = 0.0
    if feeder_voxels is not None and len(feeder_voxels) > 0:
        feeder_z = feeder_voxels[:, 2] * dx  # mm
        hot_z = start_vox[2] * dx
        dz_mm = np.max(feeder_z - hot_z)
        if dz_mm > 0:
            P_head = alloy.rho_kg_m3 * 9.81 * (dz_mm / 1000.0)

    # Feeding shrinkage demand: shrinkage of the last-liquid pocket at the hot spot
    hot_M = M_mod[start_vox[0], start_vox[1], start_vox[2]]
    V_hotspot_mm3 = (4.0 / 3.0) * np.pi * max(hot_M, 1e-3) ** 3
    V_shrink_mm3 = alloy.shrinkage_factor * V_hotspot_mm3
    Q_mm3_s = V_shrink_mm3 / max(t_s_hot, 1e-9)

    # Darcy pressure drop along the full metal path (part + runner + sprue)
    darcy = 0.0
    mu = max(alloy.viscosity_pa_s, 1e-6)
    d_dend = max(alloy.dendrite_spacing_mm, 0.01)
    f_l_stop = 0.10  # end of mass feeding / interdendritic flow
    feed_stopped = False
    for vox in path[:-1]:
        sdf_i = max(float(sdf[vox[0], vox[1], vox[2]]), 0.5)
        M_i = max(float(M_mod[vox[0], vox[1], vox[2]]), 0.5)
        f_l = _liquid_fraction_at_time(sdf_i, t_s_hot, alloy, mold)
        if f_l <= f_l_stop:
            feed_stopped = True
            break
        # Kozeny-Carman permeability in the mushy zone [mm²]
        fl_c = max(min(f_l, 0.97), 0.05)
        K_mm2 = (d_dend ** 2 / 180.0) * (fl_c ** 3) / ((1.0 - fl_c) ** 2)
        # Cross-section approximated as a disk of the local modulus
        A_mm2 = np.pi * M_i * M_i
        v_mms = Q_mm3_s / max(A_mm2, 1e-6)
        v_ms = v_mms / 1000.0
        darcy += mu * v_ms * (dx / 1000.0) / max(K_mm2 / 1e6, 1e-15)

    # Minimum driving head: 1000 Pa ≈ 0.01 atm / ~13 mm metal head
    darcy_ok = (not feed_stopped) and (darcy < max(P_head, 1000.0))

    # Heuver / directional checks on the PART portion of the path
    part_path = [v for v in path if part_mask[v[0], v[1], v[2]]]
    if len(part_path) < 2:
        part_path = path

    m_part = np.array([M_mod[v[0], v[1], v[2]] for v in part_path])
    min_neck_m = float(m_part.min()) if len(m_part) else m_hot

    # Heuver: modulus must NOT decrease toward the feeder after the first step.
    # The hot spot itself is a local maximum, so the initial drop is expected.
    heuvers_ok = True
    if len(m_part) > 4:
        tol = max(dx * 0.5, 0.1)
        if np.any(np.diff(m_part[1:]) < -tol):
            heuvers_ok = False

    # Directional solidification: T must increase (or stay) toward the feeder.
    directional_ok = True
    if len(part_path) > 4:
        t_path = np.array(
            [float(_temperature_from_erf(np.array([sdf[v[0], v[1], v[2]]]), t_s_hot, alloy, mold)[0])
             for v in part_path]
        )
        # hot spot -> feeder, T should not drop after the first step
        if np.any(np.diff(t_path[1:]) < -1.0):
            directional_ok = False

    feeding_cost = float(cost_grid[start_vox[0], start_vox[1], start_vox[2]])
    return darcy, min_neck_m, t_s_hot, directional_ok, heuvers_ok, feeding_cost, darcy_ok


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
    for di, dj, dk in NEIGH_6:
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
    # v8.2: dense local refine above ~96³ is too heavy for normal RAM budgets;
    # keep the 2040 dial for future sparse octree implementations.
    max_local_dim = min(max_res, 96)
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
            center=cropped.center_mass if cropped.is_watertight else cropped.centroid,
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
    C = chvorinov_c_from_properties(alloy, mold)
    mean_curv, _ = compute_curvature(sdf, dx)
    shape_factor_field = np.clip(
        1.0 + np.maximum(-mean_curv * sdf, 0.0), 1.0, 3.0
    )
    M_mod = sdf / shape_factor_field
    t_s = compute_chvorinov_t(M_mod, C)
    T, R, fs, _ = compute_thermal_field(
        grid, is_metal, alloy, mold, dx, sdf=sdf, M_mod=M_mod
    )
    G, R, niyama = compute_niyama(
        sdf, M_mod, alloy, mold, dx, is_metal=is_metal,
        temperature=T, cooling_rate=R,
    )

    niyama_risk = np.clip(1.0 - niyama / alloy.niyama_shrinkage, 0.0, 1.0)
    FD_field = alloy.feed_k1 * (2.0 * M_mod)
    # no global dist_feed here, use a conservative isolation term from local SDF
    feed_risk = np.clip(sdf / np.maximum(FD_field, 1.0), 0.0, 1.0)
    risk = 1.0 - (1.0 - niyama_risk) * (1.0 - feed_risk)
    risk = np.where(is_metal, risk, 0.0)
    risk = np.nan_to_num(risk, nan=0.0, posinf=0.0, neginf=0.0)

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
    sub_voxel: int = 2,
    n_thermal_steps: int = 100,
    casting_params: Optional[CastingParameters] = None,
    progress_callback: Optional[callable] = None,
) -> AnalysisResult:
    """Run the full v8.0 geometric + pseudo-thermal analysis pipeline."""
    import time

    t_start = time.time()
    if progress_callback:
        progress_callback(5)

    alloy = get_alloy(alloy_key)
    mold = get_mold(mold_key)
    if casting_params is not None:
        # User overrides for this run
        alloy = replace(
            alloy,
            t_pour_c=casting_params.t_pour_c,
            t_liquidus_c=casting_params.t_liquidus_c,
            t_solidus_c=casting_params.t_solidus_c,
            rho_kg_m3=casting_params.rho_liquid_kg_m3,
            viscosity_pa_s=casting_params.viscosity_pa_s,
        )
        mold = replace(mold, t0_c=casting_params.t_mold_c)
    chvorinov_c = chvorinov_c_from_properties(alloy, mold)
    bbox_size = np.array(grid.shape) * dx

    is_metal = np.isin(
        grid,
        [BodyType.PART, BodyType.RISER, BodyType.INGATE, BodyType.RUNNER, BodyType.SPRUE],
    )
    if is_metal.sum() < 1000:
        raise ValueError(
            "Model çok küçük. Çözünürlüğü artırın veya modelin mm biriminde olduğundan emin olun."
        )

    # v8.2 memory guard: very large grids at sub>1 explode RAM; fall back to sub=1.
    if grid.size > 2_000_000 and sub_voxel > 1:
        sub_voxel = 1

    part_mask = grid == BodyType.PART
    riser_mask = grid == BodyType.RISER

    # v8.2: If there is no separate riser, use the gating system (sprue/runner/ingate)
    # as the feeding source for distance/path calculations.
    if riser_mask.any():
        feeder_mask = riser_mask
        no_riser = False
    else:
        feeder_mask = np.isin(
            grid,
            [BodyType.INGATE, BodyType.RUNNER, BodyType.SPRUE],
        )
        no_riser = True

    # AŞAMA 2: SDF (sub-voxel) + histogram + curvature + shape factor
    sdf = compute_subvoxel_sdf(is_metal, dx, sub=sub_voxel)
    if progress_callback:
        progress_callback(18)

    mean_curv, gauss_curv = compute_curvature(sdf, dx)
    # Shape factor from mean curvature: f=1 for plates, f≈2 for cylinders, f≈3 for spheres
    shape_factor_field = np.clip(
        1.0 + np.maximum(-mean_curv * sdf, 0.0), 1.0, 3.0
    )
    M_mod = sdf / shape_factor_field
    if progress_callback:
        progress_callback(25)

    _, _, dominant_m = _sdf_histogram(M_mod, part_mask, bins=50)
    wall_thickness = 2.0 * dominant_m if dominant_m > 0 else 0.0
    m_mean, m_std, m_skew = _histogram_stats(M_mod, part_mask)
    shape_factor_global = _shape_factor(part_mask, dx)
    if progress_callback:
        progress_callback(28)

    # AŞAMA 3: Thermal field at local solidification time (1-D erf + Scheil)
    t_s = compute_chvorinov_t(M_mod, chvorinov_c)
    temperature, cooling_rate, solid_fraction, thermal_divergence = compute_thermal_field(
        grid, is_metal, alloy, mold, dx, n_steps=n_thermal_steps,
        progress_callback=progress_callback, sdf=sdf, M_mod=M_mod,
    )
    if progress_callback:
        progress_callback(50)

    # AŞAMA 4: Niyama from 3-D thermal gradient and local cooling rate
    G, R, niyama = compute_niyama(
        sdf, M_mod, alloy, mold, dx, is_metal=is_metal,
        temperature=temperature, cooling_rate=cooling_rate,
    )
    niyama_variants = compute_niyama_variants(niyama, G, R, t_s, alloy)
    niyama = compute_niyama_ensemble(niyama)
    if progress_callback:
        progress_callback(65)

    # AŞAMA 5: Hot spot detection (medial axis + DBSCAN + curvature)
    max_part_sdf = float(sdf[part_mask].max()) if part_mask.any() else 0.0
    hotspot_min_size_mm = min(2.0, max(0.5, 0.5 * max_part_sdf))
    hotspots = find_hotspots(
        sdf, part_mask, dx, origin_mm, curvature=mean_curv, use_skeleton=True,
        min_size_mm=hotspot_min_size_mm,
    )
    if progress_callback:
        progress_callback(75)

    # AŞAMA 6: 26-neighbor Dijkstra feeding distance and lowest-resistance cost path
    dist_feed = feeding_distance_dijkstra(is_metal, feeder_mask, dx)
    cost_feed, cost_pred, _ = feeding_cost_dijkstra(is_metal, feeder_mask, M_mod, dx)
    if progress_callback:
        progress_callback(82)

    # AŞAMA 7: Hot-spot physics
    feeder_voxels = np.argwhere(feeder_mask)
    for hs in hotspots:
        vox = np.round((hs.position_mm - origin_mm) / dx).astype(int)
        if 0 <= vox[0] < grid.shape[0] and 0 <= vox[1] < grid.shape[1] and 0 <= vox[2] < grid.shape[2]:
            hs.dist_to_riser_mm = float(dist_feed[vox[0], vox[1], vox[2]])
            hs.niyama_min = float(niyama[vox[0], vox[1], vox[2]])
            hs.niyama_variants = {
                k: float(v[vox[0], vox[1], vox[2]]) for k, v in niyama_variants.items()
            }
            hs.niyama_ensemble = float(niyama[vox[0], vox[1], vox[2]])
            hs.local_sdf_max = float(sdf[vox[0], vox[1], vox[2]])
            hs.m_uncertainty_mm = dx / 2.0
            hs.feeding_cost = float(cost_feed[vox[0], vox[1], vox[2]])

            darcy, min_neck_m, t_hs, directional_ok, heuvers_ok, feeding_cost, darcy_ok = _path_darcy_and_directional(
                sdf,
                M_mod,
                cost_feed,
                cost_pred,
                part_mask,
                vox,
                dx,
                alloy,
                mold,
                feeder_voxels=feeder_voxels,
            )
            hs.darcy_resistance = darcy
            hs.min_neck_m_mm = min_neck_m
            hs.directional_ok = directional_ok
            hs.heuvers_ok = heuvers_ok
            hs.darcy_ok = darcy_ok
            hs.curvature_mean = float(mean_curv[vox[0], vox[1], vox[2]])
            hs.curvature_gaussian = float(gauss_curv[vox[0], vox[1], vox[2]])

            # Section thickness = 2 * local modulus (equivalent wall thickness)
            hs.t_section_mm = 2.0 * hs.m_value_mm
            hs.width_mm = 2.0 * hs.local_sdf_max
            hs.shape_factor = _shape_factor(
                _sphere_mask(part_mask.shape, vox, max(3.0 * hs.m_value_mm / dx, 5.0)) & part_mask,
                dx,
            )

            if len(feeder_voxels) > 0:
                feeder_positions_mm = feeder_voxels * dx + origin_mm
                dz = feeder_positions_mm[:, 2] - hs.position_mm[2]
                closest_feeder_idx = int(
                    np.argmin(np.linalg.norm(feeder_positions_mm - hs.position_mm, axis=1))
                )
                dz_closest = dz[closest_feeder_idx]
                hs.gravity_factor = 1.0 + 0.3 * max(
                    0.0, dz_closest / max(hs.dist_to_riser_mm, 1.0)
                )
            else:
                dz_closest = 0.0
                hs.gravity_factor = 1.0

            # FD = feed_k1 * t_section (t_section = 2 * M_mod)
            base_fd = alloy.feed_k1 * hs.t_section_mm
            hs.max_feeding_distance_mm = base_fd * hs.gravity_factor
            feed_cost_ok = hs.feeding_cost < 30.0
            hs.feed_ok = (
                (not np.isinf(hs.dist_to_riser_mm))
                and (hs.dist_to_riser_mm <= hs.max_feeding_distance_mm)
                and hs.directional_ok
                and hs.heuvers_ok
                and feed_cost_ok
                and hs.darcy_ok
            )
        else:
            hs.feed_ok = False

    if progress_callback:
        progress_callback(88)

    # AŞAMA 8: Riser sufficiency with resistance-corrected modulus transfer
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

        volume_mm3 = voxel_count * (dx ** 3)
        volume_cm3 = volume_mm3 / 1000.0

        dilated = ndimage.binary_dilation(component_mask, iterations=1)
        surface_mask = dilated & ~component_mask
        surface_mm2 = float(surface_mask.sum()) * dx * dx
        m_riser = volume_mm3 / surface_mm2 if surface_mm2 > 0 else 0.0

        riser_centroid_vox = np.array(np.argwhere(component_mask).mean(axis=0))
        nearest_hs = None
        nearest_m = 0.0
        nearest_pos = np.zeros(3)
        nearest_resistance = 0.0
        if hotspots:
            hs_positions_vox = np.array(
                [(hs.position_mm - origin_mm) / dx for hs in hotspots]
            )
            tree = cKDTree(hs_positions_vox.astype(np.float32))
            d, idx = tree.query(riser_centroid_vox.astype(np.float32), k=1)
            nearest_hs = hotspots[idx]
            nearest_m = nearest_hs.m_value_mm
            nearest_pos = nearest_hs.position_mm
            nearest_resistance = nearest_hs.darcy_resistance

        m_required = alloy.riser_m_factor * nearest_m
        riser_z_mm = float((riser_centroid_vox[2] * dx) + origin_mm[2])
        dz = (riser_z_mm - nearest_pos[2]) if nearest_hs is not None else 0.0
        gravity = max(0.85, 1.0 - 0.005 * max(0, -dz))
        resistance_correction = alloy.modulus_resistance_mm * nearest_resistance
        effective_m_required = m_required * gravity + resistance_correction
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
            feed_volume_mm3 = float(feed_region.sum()) * (dx ** 3)
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
                resistance_correction_mm=resistance_correction,
            )
        )

    if progress_callback:
        progress_callback(92)

    # AŞAMA 9: Risk map (probabilistic OR of Niyama risk and feeding distance risk)
    with np.errstate(divide="ignore", invalid="ignore"):
        niyama_risk = np.clip(1.0 - niyama / alloy.niyama_shrinkage, 0.0, 1.0)
        FD_field = alloy.feed_k1 * (2.0 * M_mod)
        feed_risk = np.clip(dist_feed / np.maximum(FD_field, 1.0), 0.0, 1.0)
        risk = 1.0 - (1.0 - niyama_risk) * (1.0 - feed_risk)
        risk = np.where(is_metal, risk, 0.0)
        risk = np.nan_to_num(risk, nan=0.0, posinf=0.0, neginf=0.0)
    risk_norm = risk

    if progress_callback:
        progress_callback(95)

    # AŞAMA 10: Local refinement around hot spots
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
        temperature=temperature,
        cooling_rate=cooling_rate,
        solid_fraction=solid_fraction,
        curvature_mean=mean_curv,
        curvature_gaussian=gauss_curv,
        subvoxel_sdf=sdf,
        shape_factor_global=shape_factor_global,
        m_mean_mm=m_mean,
        m_std_mm=m_std,
        m_skewness=m_skew,
        niyama_variants=niyama_variants,
        elapsed_s=time.time() - t_start,
        casting_params=casting_params,
        thermal_divergence=thermal_divergence,
        bbox_size_mm=bbox_size,
    )

    result.recommendations = _build_recommendations(result, alloy, mold)
    return result


def _build_recommendations(
    result: AnalysisResult, alloy: Alloy, mold: MoldMaterial
) -> List[str]:
    recs: List[str] = []

    has_riser = (result.grid == BodyType.RISER).any()
    if not has_riser:
        recs.append(
            "Ayrı besleyici (riser) atanmamış; döküm ağzı / yolluk / meme kaynağından "
            "besleme mesafesi ve yol maliyeti hesaplandı. Soğuk birleşme ve çekinti riski "
            "daha yüksek olabilir; kritik bölgeler için besleyici eklenmesi önerilir."
        )

    recs.append(
        f"Malzeme: {alloy.name} | Kalıp: {mold.name} | Chvorinov C = {result.chvorinov_c:.4f} s/mm² | "
        f"Baskın M = {result.dominant_m_mm:.2f} mm (t ≈ {result.wall_thickness_mm:.2f} mm) | "
        f"Şekil faktörü SF = {result.shape_factor_global:.6f}"
    )
    recs.append(
        f"Modül istatistikleri: ortalama M = {result.m_mean_mm:.2f} mm, std = {result.m_std_mm:.2f} mm, "
        f"çarpıklık = {result.m_skewness:.2f}. "
        + ("Parça duvar kalınlığı dengesiz." if abs(result.m_skewness) > 1.0 else "Kalınlık dağılımı nispeten dengeli.")
    )

    if not result.hotspots:
        recs.append(
            "Kritik sıcak nokta (hot spot) tespit edilmedi. Model çok ince veya geometri düzgün okunamamış olabilir."
        )
        return recs

    for hs in result.hotspots:
        t = hs.t_section_mm
        W = hs.width_mm
        fd = alloy.feed_k1 * t
        unc = hs.m_uncertainty_mm
        recs.append(
            f"Hot spot M = {hs.m_value_mm:.2f} ± {unc:.2f} mm, t = {t:.2f} mm, "
            f"W = {W:.2f} mm, şekil faktörü = {hs.shape_factor:.6f}"
        )
        if hs.dist_to_riser_mm > hs.max_feeding_distance_mm:
            recs.append(
                f"Hot spot: besleme mesafesi {hs.dist_to_riser_mm:.1f} mm > limit {hs.max_feeding_distance_mm:.1f} mm (FD={fd:.1f} mm). "
                f"Besleyiciyi yakın taşı veya kesiti büyütün."
            )
        if not hs.directional_ok:
            recs.append(
                f"Hot spot: yönlü katılaşma bozuk, yolda daralma (boyun M={hs.min_neck_m_mm:.1f} mm). "
                f"Meme/besleyici arasındaki geometriyi kalınlaştırın."
            )
        if not hs.heuvers_ok:
            recs.append(
                "Hot spot: Heuver çemberi kuralı ihlali - besleme yolunda kesit daralıyor, "
                "ara bölge daha ince/sıcak. Meme konumunu/kalınlığını gözden geçirin."
            )
        if not hs.darcy_ok:
            recs.append(
                f"Hot spot: Darcy basınç kaybı ({hs.darcy_resistance:.2f} Pa) mevcut hidrostatik basıncı aşıyor. "
                f"Mushy-zone geçirgenliği yetersiz; meme/yol kesitini büyütün veya kısa yol seçin."
            )

        niy = hs.niyama_ensemble
        if niy < alloy.niyama_macro:
            recs.append(
                f"Hot spot: Niyama {niy:.2f} < {alloy.niyama_macro} -> "
                f"makro shrinkage / çekinti riski çok yüksek."
            )
        elif niy < alloy.niyama_shrinkage:
            recs.append(
                f"Hot spot: Niyama {niy:.2f} < {alloy.niyama_shrinkage} -> "
                f"mikro gözenek / shrinkage porozite riski."
            )

    for rr in result.riser_results:
        if not rr.large_enough:
            increase = (
                (rr.effective_m_required / max(rr.m_value_mm, 1e-6) - 1.0) * 100.0
            )
            recs.append(
                f"{rr.name}: M_besleyici={rr.m_value_mm:.2f} mm < gerekli {rr.effective_m_required:.2f} mm. "
                f"Besleyici modülünü %{int(increase)} büyütün."
            )
        if not rr.volume_ratio_ok:
            short = rr.required_volume_cm3 - rr.volume_cm3
            recs.append(
                f"{rr.name}: hacim yetersiz (V={rr.volume_cm3:.2f} cm³, gerekli {rr.required_volume_cm3:.2f} cm³). "
                f"En az {short:.2f} cm³ daha hacim ekleyin."
            )

    all_feed_ok = all(hs.feed_ok for hs in result.hotspots)
    if all_feed_ok and all(rr.large_enough for rr in result.riser_results):
        if has_riser:
            recs.append(
                "Tüm sıcak noktalar besleyici menzili içinde ve besleyici boyutları yeterli görünüyor."
            )
        else:
            recs.append(
                "Tüm sıcak noktalar gating kaynağı menzili içinde, ancak ayrı besleyici olmadan "
                "shrinkage riski tamamen giderilemeyebilir."
            )

    return recs
