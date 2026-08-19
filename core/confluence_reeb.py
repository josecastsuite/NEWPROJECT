"""V8 Reeb/watershed saddle detection for cold-shot confluence risk.

Fix5 from V8: 0-D sublevel-set persistence (merge tree) on the fill-time
landscape gives the exact Reeb-graph saddles, with a vectorised watershed
fallback.  No triple loops over the full grid.
"""
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from skimage.morphology.extrema import h_maxima
from skimage.segmentation import watershed


@np.errstate(divide="ignore", invalid="ignore")
def _neighbour_offsets(shape: Tuple[int, int, int]) -> np.ndarray:
    """Flat offsets for the 6 face-connected neighbours."""
    nz, ny, nx = shape
    return np.array([1, -1, nx, -nx, nx * ny, -nx * ny], dtype=np.int64)


def _find(parent: np.ndarray, x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def find_saddles_sublevel(
    ft: np.ndarray,
    part_mask: np.ndarray,
    persistence_thresh_s: float = 0.1,
    max_saddles: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    0-D sublevel-set persistence (merge tree) on ``ft``.

    Voxels are added in increasing fill time; a saddle is recorded whenever the
    new voxel connects two previously separate components.  Persistence is the
    saddle time minus the larger of the two component minima.
    """
    part_mask = part_mask.astype(bool, copy=False)
    ft = np.asarray(ft, dtype=np.float64)
    shape = ft.shape
    N = ft.size

    part_idx = np.flatnonzero(part_mask)
    if part_idx.size == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    ft_part = ft.flat[part_idx]
    order = np.argsort(ft_part, kind="mergesort")
    sorted_idx = part_idx[order]
    sorted_ft = ft_part[order]

    parent = np.full(N, -1, dtype=np.int64)
    min_ft = np.full(N, np.inf, dtype=np.float64)
    comp_size = np.zeros(N, dtype=np.int64)

    nz, ny, nx = shape
    s_nx = 1
    s_ny = nx
    s_nz = nx * ny

    coords_list = []
    vals_list = []
    pers_list = []

    for flat, t in zip(sorted_idx, sorted_ft):
        parent[flat] = flat
        min_ft[flat] = t
        comp_size[flat] = 1

        # 6-connected active neighbours (all directions; a merge can arrive from
        # any side, not just the negative axes).
        nbs = []
        ix = flat % nx
        iy = (flat // nx) % ny
        iz = flat // s_nz
        if ix > 0:
            nbs.append(flat - s_nx)
        if ix < nx - 1:
            nbs.append(flat + s_nx)
        if iy > 0:
            nbs.append(flat - s_ny)
        if iy < ny - 1:
            nbs.append(flat + s_ny)
        if iz > 0:
            nbs.append(flat - s_nz)
        if iz < nz - 1:
            nbs.append(flat + s_nz)

        roots = set()
        r0 = _find(parent, flat)
        for nb in nbs:
            if parent[nb] != -1:
                roots.add(_find(parent, nb))
        roots.discard(r0)

        for r in roots:
            r0 = _find(parent, flat)
            if r0 == r:
                continue
            persistence = t - max(min_ft[r0], min_ft[r])
            if persistence > persistence_thresh_s:
                k = flat // s_nz
                rem = flat % s_nz
                j = rem // nx
                i = rem % nx
                coords_list.append((i, j, k))
                vals_list.append(t)
                pers_list.append(persistence)
            # union by size
            if comp_size[r0] < comp_size[r]:
                r0, r = r, r0
            parent[r] = r0
            comp_size[r0] += comp_size[r]
            min_ft[r0] = min(min_ft[r0], min_ft[r])

    if not coords_list:
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    coords = np.array(coords_list, dtype=np.int64)
    vals = np.array(vals_list, dtype=np.float64)
    pers = np.array(pers_list, dtype=np.float64)

    # keep the most persistent saddles up to the requested cap
    if coords.shape[0] > max_saddles:
        order = np.argsort(-pers)
        order = order[:max_saddles]
        coords = coords[order]
        vals = vals[order]
        pers = pers[order]

    return coords, vals, pers


def _shifted_labels(labels: np.ndarray, axis: int, direction: int) -> np.ndarray:
    """Return labels shifted by one voxel along ``axis`` without wrap-around."""
    nb = np.zeros_like(labels)
    if direction == 1:
        # neighbour is the previous voxel along axis
        slices = [slice(None)] * labels.ndim
        slices[axis] = slice(1, None)
        slices_nb = [slice(None)] * labels.ndim
        slices_nb[axis] = slice(0, -1)
        nb[tuple(slices_nb)] = labels[tuple(slices)]
    else:
        slices = [slice(None)] * labels.ndim
        slices[axis] = slice(0, -1)
        slices_nb = [slice(None)] * labels.ndim
        slices_nb[axis] = slice(1, None)
        nb[tuple(slices_nb)] = labels[tuple(slices)]
    return nb


def find_saddles_watershed(
    ft: np.ndarray,
    part_mask: np.ndarray,
    persistence_thresh_s: float = 0.1,
    sigma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find persistent saddles on the fill-time (``ft``) landscape.

    Returns
    -------
    coords : (N, 3) int
        Saddle voxel coordinates.
    ft_saddle : (N,) float
        Fill time at the saddle.
    persistence : (N,) float
        ft_saddle - max(ft_min_a, ft_min_b) for the two adjacent basins.
    """
    part_mask = part_mask.astype(bool, copy=False)
    ft = np.asarray(ft, dtype=np.float64)

    valid = part_mask & np.isfinite(ft)
    if not valid.any():
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    t_max = float(ft[valid].max())
    ft_work = np.where(valid, ft, t_max + 1.0)
    ft_smooth = ndimage.gaussian_filter(ft_work, sigma=sigma)

    # local minima of the smoothed fill-time field
    min_filtered = ndimage.minimum_filter(ft_smooth, footprint=np.ones((3, 3, 3)))
    minima = (min_filtered == ft_smooth) & valid
    markers, n_labels = ndimage.label(minima)
    if n_labels < 2:
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    # Watershed on the fill-time surface; background outside part_mask is ignored.
    labels = watershed(ft_smooth, markers=markers, mask=valid)

    # Face-connected boundary voxels between two different basins.
    pairs_list = []
    vals_list = []
    coords_list = []
    for axis in range(3):
        for direction in (1, -1):
            nb = _shifted_labels(labels, axis, direction)
            mask = valid & (labels > 0) & (nb > 0) & (labels != nb)
            if not mask.any():
                continue
            a = np.minimum(labels[mask], nb[mask])
            b = np.maximum(labels[mask], nb[mask])
            pairs_list.append(np.column_stack((a, b)))
            vals_list.append(ft_smooth[mask])
            coords_list.append(np.stack(np.nonzero(mask), axis=1))

    if not pairs_list:
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    pairs = np.vstack(pairs_list)
    vals = np.concatenate(vals_list)
    coords = np.vstack(coords_list)

    # Sort by (a, b, val) so the first occurrence of each pair has the minimum ft.
    order = np.lexsort((vals, pairs[:, 1], pairs[:, 0]))
    pairs = pairs[order]
    vals = vals[order]
    coords = coords[order]

    # group boundaries
    diff = np.concatenate(([True], np.any(pairs[1:] != pairs[:-1], axis=1)))
    idx = np.where(diff)[0]

    saddle_pairs = pairs[idx]
    saddle_vals = vals[idx]
    saddle_coords = coords[idx]

    # minimum fill time per basin label
    min_ft = np.full(n_labels + 1, np.inf, dtype=np.float64)
    np.minimum.at(min_ft, labels[valid], ft_smooth[valid])

    max_min = np.maximum(min_ft[saddle_pairs[:, 0]], min_ft[saddle_pairs[:, 1]])
    persistence = saddle_vals - max_min

    keep = persistence > persistence_thresh_s
    return saddle_coords[keep], saddle_vals[keep], persistence[keep]


GATING_BODY_TYPES = (5, 6, 7, 15, 17, 19, 21)  # INGATE, RUNNER, SPRUE, POURING_BASIN, SPRUE_THROAT, DISTRIBUTOR, CURUFLUK


def find_saddles_gate_watershed(
    ft: np.ndarray,
    part_mask: np.ndarray,
    grid: np.ndarray,
    body_index: np.ndarray,
    persistence_thresh_s: float = 0.1,
    max_saddles: int = 1000,
    sigma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Watershed on the fill-time field seeded by the individual gate bodies.

    Each gate body is a distinct source; the part voxels that are 6-neighbour to
    that gate are marked with the body's label.  Watershed then propagates those
    markers through the part, and the boundaries between different labels are the
    confluence lines of the separate filling fronts.  The saddle for each
    boundary segment is the lowest fill-time voxel on that boundary.
    """
    part_mask = part_mask.astype(bool, copy=False)
    ft = np.asarray(ft, dtype=np.float64)
    grid = np.asarray(grid)
    body_index = np.asarray(body_index)
    shape = ft.shape
    valid = part_mask & np.isfinite(ft)
    if not valid.any():
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    gate_mask = np.isin(grid, GATING_BODY_TYPES)
    gate_labels = body_index * gate_mask  # 0 outside gates
    gate_ids = np.unique(gate_labels[gate_mask])
    if gate_ids.size < 2:
        # Fall back to auto minima-based watershed if no separate gates.
        return find_saddles_watershed(
            ft, part_mask, persistence_thresh_s=persistence_thresh_s, sigma=sigma
        )

    # Mark part voxels that touch each gate body.
    markers = np.zeros(shape, dtype=np.int32)
    structure = np.ones((3, 3, 3), dtype=bool)
    for lbl in gate_ids:
        gate_blob = gate_labels == lbl
        if not gate_blob.any():
            continue
        dilated = ndimage.binary_dilation(gate_blob, structure=structure)
        # Part voxels adjacent to this gate become its seed.
        seed = valid & dilated & (markers == 0)
        markers[seed] = int(lbl)

    if int(markers.max()) < 2:
        return find_saddles_watershed(
            ft, part_mask, persistence_thresh_s=persistence_thresh_s, sigma=sigma
        )

    t_max = float(ft[valid].max())
    ft_work = np.where(valid, ft, t_max + 1.0)
    ft_smooth = ndimage.gaussian_filter(ft_work, sigma=sigma)

    labels = watershed(ft_smooth, markers=markers, mask=valid)
    n_labels = int(labels.max())

    # Face-connected boundary voxels between two different labels.
    pairs_list = []
    vals_list = []
    coords_list = []
    for axis in range(3):
        for direction in (1, -1):
            nb = _shifted_labels(labels, axis, direction)
            mask = valid & (labels > 0) & (nb > 0) & (labels != nb)
            if not mask.any():
                continue
            a = np.minimum(labels[mask], nb[mask])
            b = np.maximum(labels[mask], nb[mask])
            pairs_list.append(np.column_stack((a, b)))
            vals_list.append(ft_smooth[mask])
            coords_list.append(np.stack(np.nonzero(mask), axis=1))

    if not pairs_list:
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    pairs = np.vstack(pairs_list)
    vals = np.concatenate(vals_list)
    coords = np.vstack(coords_list)

    # For each (a,b) pair keep the lowest-ft voxel on their shared boundary.
    order = np.lexsort((vals, pairs[:, 1], pairs[:, 0]))
    pairs = pairs[order]
    vals = vals[order]
    coords = coords[order]
    diff = np.concatenate(([True], np.any(pairs[1:] != pairs[:-1], axis=1)))
    idx = np.where(diff)[0]

    saddle_pairs = pairs[idx]
    saddle_vals = vals[idx]
    saddle_coords = coords[idx]

    # Minimum fill time per basin label (at the source).
    min_ft = np.full(n_labels + 1, np.inf, dtype=np.float64)
    np.minimum.at(min_ft, labels[valid], ft_smooth[valid])
    max_min = np.maximum(min_ft[saddle_pairs[:, 0]], min_ft[saddle_pairs[:, 1]])
    persistence = saddle_vals - max_min

    keep = persistence > persistence_thresh_s
    saddle_coords = saddle_coords[keep]
    saddle_vals = saddle_vals[keep]
    persistence = persistence[keep]

    if saddle_coords.shape[0] > max_saddles:
        order = np.argsort(-persistence)
        order = order[:max_saddles]
        saddle_coords = saddle_coords[order]
        saddle_vals = saddle_vals[order]
        persistence = persistence[order]

    return saddle_coords, saddle_vals, persistence


def find_local_maxima(
    ft: np.ndarray,
    part_mask: np.ndarray,
    h_relative: float = 0.05,
    sigma: float = 2.0,
    max_candidates: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Persistent local maxima of the fill-time field inside the part.

    Maxima are the last-filled closure points where the metal front meets
    itself; they are equally valid confluence candidates for SFER.
    """
    part_mask = part_mask.astype(bool, copy=False)
    ft = np.asarray(ft, dtype=np.float64)
    valid = part_mask & np.isfinite(ft)
    if not valid.any():
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    t_min = float(ft[valid].min())
    t_max = float(ft[valid].max())

    # Work inside a tight bounding box for speed/memory on high-resolution grids.
    coords = np.argwhere(valid)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    pad = int(np.ceil(sigma * 3))
    lo = np.maximum(lo - pad, 0)
    hi = np.minimum(hi + pad, np.array(ft.shape))
    slc = tuple(slice(l, h) for l, h in zip(lo, hi))
    ft_crop = ft[slc]
    valid_crop = valid[slc]

    ft_work = np.where(valid_crop, ft_crop, t_min - 1.0)
    ft_smooth = ndimage.gaussian_filter(ft_work, sigma=sigma)

    h = h_relative * (t_max - t_min)
    if h <= 0.0:
        h = 0.001
    maxima = h_maxima(ft_smooth, h)
    maxima = maxima & valid_crop
    if not maxima.any():
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    labeled, n_comp = ndimage.label(maxima)
    coords_list = []
    vals_list = []
    for c in range(1, n_comp + 1):
        patch = labeled == c
        patch_coords = np.argwhere(patch)
        patch_vals = ft_smooth[patch]
        idx = int(np.argmax(patch_vals))
        local = patch_coords[idx]
        coords_list.append(tuple(local + lo))
        vals_list.append(float(ft[tuple(local + lo)]))

    coords = np.array(coords_list, dtype=np.int64)
    vals = np.array(vals_list, dtype=np.float64)
    persistence = vals - t_min

    if coords.shape[0] > max_candidates:
        order = np.argsort(-persistence)
        order = order[:max_candidates]
        coords = coords[order]
        vals = vals[order]
        persistence = persistence[order]

    return coords, vals, persistence


def find_saddles_skeleton(
    ft: np.ndarray,
    part_mask: np.ndarray,
    persistence_thresh_s: float = 0.1,
    ft_percentile: float = 80.0,
    max_saddles: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Skeleton / medial-axis confluence candidates.

    The thinned centre-line of the part is where opposing filling fronts are
    most likely to meet.  Voxels on the skeleton with late fill times (high
    ``ft`` percentile) are returned as confluence candidates.
    """
    part_mask = part_mask.astype(bool, copy=False)
    ft = np.asarray(ft, dtype=np.float64)
    valid = part_mask & np.isfinite(ft)
    if not valid.any():
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    # Crop to the part bounding box for skeleton speed.
    coords_bbox = np.argwhere(valid)
    lo = coords_bbox.min(axis=0)
    hi = coords_bbox.max(axis=0) + 1
    slc = tuple(slice(l, h) for l, h in zip(lo, hi))
    part_crop = part_mask[slc]
    ft_crop = ft[slc]

    sk = skeletonize(part_crop)
    if not sk.any():
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    t_min = float(ft_crop[part_crop].min())
    t_max = float(ft_crop[part_crop].max())
    if t_max <= t_min:
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    thr = np.percentile(ft_crop[part_crop], ft_percentile)
    candidates = sk & (ft_crop >= thr)
    if not candidates.any():
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    # label each connected skeleton segment and pick the highest-ft voxel in it
    labeled, n_comp = ndimage.label(candidates)
    coords_list = []
    vals_list = []
    for c in range(1, n_comp + 1):
        patch = labeled == c
        patch_coords = np.argwhere(patch)
        patch_vals = ft_crop[patch]
        idx = int(np.argmax(patch_vals))
        local = patch_coords[idx]
        coords_list.append(tuple(local + lo))
        vals_list.append(float(ft[tuple(local + lo)]))

    coords = np.array(coords_list, dtype=np.int64)
    vals = np.array(vals_list, dtype=np.float64)
    persistence = vals - t_min

    keep = persistence > persistence_thresh_s
    coords = coords[keep]
    vals = vals[keep]
    persistence = persistence[keep]

    if coords.shape[0] > max_saddles:
        order = np.argsort(-persistence)
        order = order[:max_saddles]
        coords = coords[order]
        vals = vals[order]
        persistence = persistence[order]

    return coords, vals, persistence


def find_saddles_gate_voronoi(
    ft: np.ndarray,
    part_mask: np.ndarray,
    grid: np.ndarray,
    body_index: np.ndarray,
    persistence_thresh_s: float = 0.1,
    max_saddles: int = 1000,
    sampling: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Geometric Voronoi confluence of gate bodies within the part.

    For each part voxel, the nearest gate body (in Euclidean distance) is found
    with ``distance_transform_edt``.  Voxels whose nearest gate changes across a
    face are confluence candidates.  The saddle for each connected boundary
    patch is the part voxel on the patch with the smallest fill time.
    """
    part_mask = part_mask.astype(bool, copy=False)
    ft = np.asarray(ft, dtype=np.float64)
    grid = np.asarray(grid)
    body_index = np.asarray(body_index)
    shape = ft.shape
    valid = part_mask & np.isfinite(ft)
    if not valid.any():
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    gate_mask = np.isin(grid, GATING_BODY_TYPES)
    gate_labels = body_index * gate_mask  # 0 outside gates
    gate_ids = np.unique(gate_labels[gate_mask])
    if gate_ids.size < 2:
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    # Build a marker volume with one label per gate body.
    marker = np.zeros(shape, dtype=np.int32)
    for lbl in gate_ids:
        marker[gate_labels == lbl] = int(lbl)

    if sampling is None:
        sampling = (1.0, 1.0, 1.0)
    _, nearest = ndimage.distance_transform_edt(
        marker == 0, return_indices=True, sampling=sampling
    )
    nearest_label = marker[nearest[0], nearest[1], nearest[2]]

    # Face-connected boundary voxels between two different gate catchments.
    coords_list = []
    vals_list = []
    pers_list = []
    structure = ndimage.generate_binary_structure(3, 1)
    boundaries = np.zeros(shape, dtype=bool)
    for axis in range(3):
        for direction in (1, -1):
            nb = _shifted_labels(nearest_label, axis, direction)
            boundary = valid & (nearest_label > 0) & (nb > 0) & (nearest_label != nb)
            boundaries |= boundary

    if not boundaries.any():
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    # Label connected boundary patches; each patch is one saddle candidate.
    labeled, n_comp = ndimage.label(boundaries, structure=structure)
    min_ft_gate = np.full(int(np.max(nearest_label)) + 1, np.inf, dtype=np.float64)
    np.minimum.at(min_ft_gate, nearest_label[valid], ft[valid])

    for c in range(1, n_comp + 1):
        patch = labeled == c
        if not patch.any():
            continue
        # the two (or more) gate labels surrounding this patch
        dil = ndimage.binary_dilation(patch, structure=structure)
        adj = set(nearest_label[dil & valid]) - {0}
        if len(adj) < 2:
            continue
        # take the two most frequent labels
        counts = {}
        for v in adj:
            counts[v] = int(np.sum(nearest_label[dil & valid] == v))
        top = sorted(counts, key=counts.get, reverse=True)[:2]
        a, b = int(top[0]), int(top[1])

        patch_coords = np.argwhere(patch)
        patch_vals = ft[patch]
        min_idx = int(np.argmin(patch_vals))
        saddle_coord = patch_coords[min_idx]
        saddle_val = float(patch_vals[min_idx])
        persistence = saddle_val - max(min_ft_gate[a], min_ft_gate[b])
        if persistence > persistence_thresh_s:
            coords_list.append(tuple(saddle_coord))
            vals_list.append(saddle_val)
            pers_list.append(persistence)

    if not coords_list:
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    coords = np.array(coords_list, dtype=np.int64)
    vals = np.array(vals_list, dtype=np.float64)
    pers = np.array(pers_list, dtype=np.float64)

    if coords.shape[0] > max_saddles:
        order = np.argsort(-pers)
        order = order[:max_saddles]
        coords = coords[order]
        vals = vals[order]
        pers = pers[order]

    return coords, vals, pers
