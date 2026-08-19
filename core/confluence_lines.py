"""V8 cold-shut confluence-line extraction from the fill-time field.

Feng & Liao (2021, *China Foundry*) point out that the scalar time field
``T(x,y,z)`` created by a VOF/front-tracking filling solver contains ridges
where two liquid-metal fronts meet.  The meeting locus is a 1-D ridge of
locally *late* fill time and negative second derivative transverse to the
front propagation direction.

This module extracts those ridges, skeletonizes them to 1-D voxel curves,
prunes small/noisy branches with a confluent-scale length filter, and returns
a set of ordered polylines.  Each line carries its own cold-shut risk scalar,
so the UI can render them as thin ``inferno`` tubes instead of painting the
whole part.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.interpolate import splprep, splev
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from skimage.morphology import skeletonize


def _voxel_to_world(coords: np.ndarray, origin: np.ndarray, dx: float) -> np.ndarray:
    """Convert integer voxel indices to world coordinates in mm."""
    return origin + (coords.astype(np.float64) + 0.5) * dx


def _build_adjacency_26(mask: np.ndarray, flat_indices: np.ndarray) -> csr_matrix:
    """Build a 26-neighbour adjacency graph for a set of voxel indices.

    ``mask`` is a boolean 3-D array.  ``flat_indices`` are the ravelled
    positions (in C order, like ``np.ravel_multi_index``) of the nodes.
    Returns an undirected ``csr_matrix`` with unit edge weights.
    """
    shape = mask.shape
    n = flat_indices.size
    # node id -> (i,j,k)
    indices = np.array(np.unravel_index(flat_indices, shape), dtype=np.int64).T
    # neighbour offsets (26-connectivity)
    offsets = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                offsets.append((di, dj, dk))
    offsets = np.array(offsets, dtype=np.int64)
    rows: List[int] = []
    cols: List[int] = []
    for node_id, idx in enumerate(indices):
        nbrs = idx + offsets
        # keep inside volume
        ok = (
            (nbrs[:, 0] >= 0)
            & (nbrs[:, 0] < shape[0])
            & (nbrs[:, 1] >= 0)
            & (nbrs[:, 1] < shape[1])
            & (nbrs[:, 2] >= 0)
            & (nbrs[:, 2] < shape[2])
        )
        nbrs = nbrs[ok]
        if nbrs.size == 0:
            continue
        flat = np.ravel_multi_index(nbrs.T, shape)
        # keep only neighbours that are also in the component set
        keep = mask.ravel()[flat]
        flat = flat[keep]
        for nb_id in flat:
            rows.append(node_id)
            cols.append(int(nb_id))
    data = np.ones(len(rows), dtype=np.float64)
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def _order_component(coords: np.ndarray) -> np.ndarray:
    """Return ``coords`` ordered along the longest path of the 26-graph.

    For tree-like skeletons this gives a sensible polyline.  For cycles it
    breaks the cycle at the most distant pair.
    """
    n = coords.shape[0]
    if n <= 1:
        return coords
    if n == 2:
        return coords
    # build adjacency using fast pairwise Chebyshev check (components are small)
    diff = coords[:, None, :] - coords[None, :, :]
    adj = (np.abs(diff).max(axis=2) <= 1) & (np.abs(diff).sum(axis=2) > 0)
    # sparse graph
    graph = csr_matrix(adj.astype(np.float64))
    # endpoints have degree 1
    degree = np.asarray(graph.sum(axis=0)).ravel()
    endpoints = np.where(degree == 1)[0]
    if endpoints.size >= 2:
        src, dst = int(endpoints[0]), int(endpoints[-1])
    else:
        # cycle or clump: pick the most distant pair
        if n <= 200:
            dists = np.max(np.abs(diff), axis=2)
        else:
            # sample to keep it cheap
            sample = np.random.choice(n, min(n, 200), replace=False)
            dists = np.max(np.abs(coords[sample][:, None, :] - coords[None, :, :]), axis=2)
        i, j = np.unravel_index(int(np.argmax(dists)), dists.shape)
        if endpoints.size == 0:
            src, dst = int(i), int(j)
        else:
            src = int(endpoints[0])
            # pick farthest from src
            dst = int(np.argmax(np.max(np.abs(coords - coords[src]), axis=1)))
    dist_matrix, predecessors = shortest_path(
        graph, directed=False, indices=src, return_predecessors=True
    )
    if not np.isfinite(dist_matrix[dst]):
        # fall back to greedy nearest-neighbour
        return _greedy_order(coords)
    # reconstruct path from dst back to src
    path = [dst]
    node = dst
    while node != src and predecessors[node] != -9999:
        node = int(predecessors[node])
        path.append(node)
    path.reverse()
    return coords[path]


def _greedy_order(coords: np.ndarray) -> np.ndarray:
    """Fallback TSP-nearest-neighbour order."""
    n = coords.shape[0]
    if n <= 1:
        return coords
    remaining = set(range(1, n))
    order = [0]
    current = 0
    while remaining:
        dists = np.sum((coords[list(remaining)] - coords[current]) ** 2, axis=1)
        nxt = list(remaining)[int(np.argmin(dists))]
        order.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return coords[order]


def _smooth_polyline(points: np.ndarray, dx: float) -> np.ndarray:
    """Resample an ordered polyline with a chord-length B-spline.

    The skeleton voxels are axis-aligned and coarse; a lightweight spline
    smooths the line for professional rendering while staying within a
    fraction of a voxel of the original ridge.
    """
    n = points.shape[0]
    if n < 3:
        return points

    # Chord-length parameterisation.
    diffs = np.diff(points, axis=0)
    chord = np.concatenate(([0.0], np.cumsum(np.sqrt(np.sum(diffs * diffs, axis=1)))))
    if chord[-1] <= 0.0:
        return points
    u = chord / chord[-1]

    # Desired resampling: at most half a voxel spacing so the tube looks smooth.
    length_mm = float(chord[-1])
    mean_step = length_mm / max(1, n - 1)
    n_samples = max(64, int(np.ceil(length_mm / max(mean_step * 0.5, dx * 0.25))))

    try:
        # Allow each point to deviate by ~30 % of the mean step in each coordinate.
        tol = 0.3 * mean_step
        s = max(1.0, n) * (tol * tol)
        k = min(3, n - 1)
        tck, _ = splprep(points.T, u=u, s=s, k=k, quiet=2)
        unew = np.linspace(0.0, 1.0, n_samples)
        smooth = np.vstack(splev(unew, tck)).T
    except Exception:
        # Fall back to linear interpolation at half-voxel spacing.
        unew = np.linspace(0.0, 1.0, n_samples)
        smooth = np.vstack([
            np.interp(unew, u, points[:, 0]),
            np.interp(unew, u, points[:, 1]),
            np.interp(unew, u, points[:, 2]),
        ]).T

    return smooth


def _component_risk_and_saddle(
    component_mask: np.ndarray,
    risk: np.ndarray,
    saddle_risk_by_voxel: Dict[Tuple[int, int, int], float],
) -> Tuple[float, bool]:
    """Return the risk for this component and whether it hosts a saddle."""
    coords = np.argwhere(component_mask)
    tuples = [tuple(int(c) for c in coord) for coord in coords]
    max_saddle_risk = 0.0
    has_saddle = False
    for t in tuples:
        r = saddle_risk_by_voxel.get(t)
        if r is not None:
            has_saddle = True
            if r > max_saddle_risk:
                max_saddle_risk = r
    if has_saddle:
        return max_saddle_risk, True
    # no exact saddle; sample the risk field on the ridge
    vals = risk[component_mask]
    return float(np.max(vals)) if vals.size else 0.0, False


def extract_confluence_lines(
    fill_time_s: np.ndarray,
    part_mask: np.ndarray,
    risk: np.ndarray,
    saddles: List[Dict[str, Any]],
    origin_mm: np.ndarray,
    dx: float,
    sigma: float = 1.0,
    lap_percentile: float = 2.0,
    risk_threshold: float = 0.3,
    min_length_mm: float = 2.0,
    max_lines: int = 5,
) -> List[Dict[str, Any]]:
    """Extract ordered cold-shut confluence lines from the fill-time field.

    Parameters
    ----------
    fill_time_s
        3-D scalar time field ``T(x,y,z)`` (seconds since pour start).
    part_mask
        Boolean array marking metal/part voxels.
    risk
        Per-voxel cold-shut risk, used both as a mask and for line colouring.
    saddles
        Diagnostics dicts from ``compute_sfer_risk`` with at least ``i,j,k``
        and ``risk_cs``.
    origin_mm, dx
        Voxel-to-world conversion.
    sigma
        Gaussian smoothing scale (voxels) for ``gaussian_laplace``.
    lap_percentile
        Keep voxels whose Laplacian is in the lowest ``lap_percentile``
        (most negative) inside the part.
    risk_threshold
        Only keep ridges whose local cold-shut risk is at least this.
    min_length_mm
        Confluent-scale filter: drop lines shorter than this.
    max_lines
        Return at most this many lines, sorted by ``risk * length``.

    Returns
    -------
    List of ``{"points": np.ndarray, "risk": float, "length_mm": float}``.
    """
    shape = part_mask.shape
    part_mask = part_mask.astype(bool, copy=False)
    ft = np.asarray(fill_time_s, dtype=np.float64)
    valid = part_mask & np.isfinite(ft) & (ft >= 0.0)
    if not valid.any():
        return []

    # Build a smooth working copy.  Outside the part is set to the maximum
    # fill time so the Laplacian does not explode at the air/metal boundary.
    t_max = float(np.max(ft[valid]))
    work = ft.copy()
    work[~valid] = t_max
    work = np.nan_to_num(work, nan=t_max, posinf=t_max, neginf=0.0)

    lap = ndimage.gaussian_laplace(work, sigma=sigma)
    # Keep only the most negative Laplacian values inside the part; these are
    # the local maxima/ridges of the time field where two fronts collide.
    lap_part = lap[part_mask]
    if lap_part.size == 0:
        return []
    lap_thr = float(np.percentile(lap_part, lap_percentile))
    if lap_thr >= 0:
        # no meaningful negative ridge
        return []

    risk = np.asarray(risk, dtype=np.float64)
    ridge = part_mask & (lap < lap_thr) & (risk >= risk_threshold)
    if not ridge.any():
        return []

    # Thin the ridge volume to a 1-voxel-wide curve network.
    sk = skeletonize(ridge)
    if not sk.any():
        return []

    # 26-connected components = candidate confluence lines.
    structure = ndimage.generate_binary_structure(3, 3)
    labeled, n_comp = ndimage.label(sk, structure=structure)
    if n_comp == 0:
        return []

    # Map saddle voxels to risk.  A small dilation catches saddles that fall
    # one voxel away from the skeleton.
    saddle_risk: Dict[Tuple[int, int, int], float] = {}
    for s in saddles:
        i = int(s.get("i", -1))
        j = int(s.get("j", -1))
        k = int(s.get("k", -1))
        r = float(s.get("risk_cs", 0.0))
        if 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]:
            saddle_risk[(i, j, k)] = max(saddle_risk.get((i, j, k), 0.0), r)
    if saddle_risk:
        saddle_mask = np.zeros(shape, dtype=bool)
        for (i, j, k), _ in saddle_risk.items():
            saddle_mask[i, j, k] = True
        dilated_saddles = ndimage.binary_dilation(saddle_mask, iterations=2)
    else:
        dilated_saddles = np.zeros(shape, dtype=bool)

    lines: List[Dict[str, Any]] = []
    origin_arr = np.asarray(origin_mm, dtype=np.float64)
    for label in range(1, n_comp + 1):
        comp = labeled == label
        coords = np.argwhere(comp)
        if coords.shape[0] < 2:
            continue

        # Risk and saddle association.
        comp_risk, has_saddle = _component_risk_and_saddle(comp, risk, saddle_risk)
        if comp_risk < risk_threshold:
            continue
        # Require either an exact/nearby saddle or a high risk ridge.
        if not has_saddle and not (dilated_saddles & comp).any():
            # keep it if the risk is high enough anyway
            if comp_risk < max(risk_threshold, 0.5):
                continue

        ordered = _order_component(coords)
        if ordered.shape[0] < 2:
            continue

        # Convert to world and smooth the axis-aligned voxel staircase.
        points = _voxel_to_world(ordered, origin_arr, dx)
        points = _smooth_polyline(points, dx)

        # Physical length along the smoothed polyline.
        diffs = np.diff(points, axis=0)
        length_mm = float(np.sum(np.sqrt(np.sum(diffs * diffs, axis=1))))
        if length_mm < min_length_mm:
            continue

        lines.append(
            {
                "points": points,
                "risk": float(comp_risk),
                "length_mm": length_mm,
            }
        )

    # Confluent scale: keep the strongest/main lines.
    if not lines:
        return []
    lines.sort(key=lambda ln: ln["risk"] * ln["length_mm"], reverse=True)
    return lines[:max_lines]
