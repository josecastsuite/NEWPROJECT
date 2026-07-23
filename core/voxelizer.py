"""Convert a set of Body meshes into a single labelled voxel grid - JoseCast v7."""

import warnings
from typing import List, Optional, Tuple

import numpy as np
import trimesh
from scipy import ndimage

from core.types import Body, BodyType, BODY_METAL_TYPES

BASE_RES = 160
MAX_RES = 2040


UNIT_SCALE = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "inch": 25.4,
}


def apply_unit_scale(bodies: List[Body], unit: str) -> float:
    """Scale all body meshes to millimeters and return the scale factor."""
    scale = UNIT_SCALE.get(unit, 1.0)
    if scale == 1.0:
        for b in bodies:
            b.surface_area_cm2 = b.mesh.area / 100.0
            b.volume_cm3 = b.mesh.volume / 1000.0
        return 1.0
    for b in bodies:
        b.mesh.apply_scale(scale)
        b.vertices = b.mesh.vertices.copy()
        b.center = b.mesh.center_mass if b.mesh.is_watertight else b.mesh.centroid
        b.volume_cm3 = b.mesh.volume / 1000.0
        b.surface_area_cm2 = b.mesh.area / 100.0
    return scale


def detect_unit_suggestion(bodies: List[Body]) -> str:
    """Suggest a unit based on bounding box magnitude."""
    if not bodies:
        return "mm"
    max_size = np.max(
        np.array([b.mesh.bounds[1] - b.mesh.bounds[0] for b in bodies])
    )
    if max_size < 0.05:
        return "m"
    if max_size < 5.0:
        return "cm"
    if max_size > 5000.0:
        return "m"  # probably metres, not mm
    return "mm"


def _global_bbox(bodies: List[Body]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (min, max) of all body vertices in mm."""
    mins = np.vstack([b.mesh.bounds[0] for b in bodies])
    maxs = np.vstack([b.mesh.bounds[1] for b in bodies])
    return mins.min(axis=0), maxs.max(axis=0)


def _bboxes_overlap_or_close(
    min_a: np.ndarray,
    max_a: np.ndarray,
    min_b: np.ndarray,
    max_b: np.ndarray,
    tol: float = 2.0,
) -> bool:
    """True if two bounding boxes overlap or are within ``tol`` of each other."""
    return bool(np.all((max_a + tol) >= min_b) and np.all((max_b + tol) >= min_a))


def _classify_casting_bodies(
    bodies: List[Body],
    gravity_vector: Tuple[float, float, float] = (0.0, 0.0, -1.0),
    tol_mm: float = 2.0,
) -> None:
    """Heuristic body-type assignment for STEP files with multiple solids.

    - The largest solid is the casting (PART).
    - Solids connected to the part that extend opposite to gravity are labelled RISER.
    - Remaining connected solids are ordered by distance from the part along the
      gating chain; the closest is INGATE, the farthest is SPRUE, and the rest
      are RUNNER.
    """
    if not bodies or any(b.body_type != BodyType.PART for b in bodies):
        return

    # Use mesh volume; if mesh is non-watertight fall back to bbox volume.
    def _volume(b: Body) -> float:
        if b.volume_cm3 > 0.0:
            return b.volume_cm3
        size = b.mesh.bounds[1] - b.mesh.bounds[0]
        return float(np.prod(size)) / 1000.0

    part_idx = int(np.argmax([_volume(b) for b in bodies]))
    part = bodies[part_idx]
    part_center = part.center
    part_min = part.mesh.bounds[0]
    part_max = part.mesh.bounds[1]
    part_size = part_max - part_min
    part_min_dim = float(np.min(part_size))

    # Up is opposite to gravity (where a riser sits).
    g = np.asarray(gravity_vector, dtype=np.float64)
    g_norm = float(np.linalg.norm(g)) + 1e-12
    up = -g / g_norm

    n = len(bodies)
    mins = [b.mesh.bounds[0] for b in bodies]
    maxs = [b.mesh.bounds[1] for b in bodies]

    # Build adjacency from bounding-box proximity.
    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _bboxes_overlap_or_close(mins[i], maxs[i], mins[j], maxs[j], tol_mm):
                adj[i].append(j)
                adj[j].append(i)

    # Breadth-first search from the part to find attached auxiliaries.
    visited = [False] * n
    visited[part_idx] = True
    queue = [part_idx]
    connected: List[int] = []
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                queue.append(v)
                if v != part_idx:
                    connected.append(v)

    # Only keep auxiliaries that protrude outside the part's bounding box.
    # Internal bodies (holes, cores, mounting bosses) are left as PART.
    def _protrudes(v: int) -> bool:
        bmin = mins[v]
        bmax = maxs[v]
        return bool(
            np.any(bmin < part_min - tol_mm) or np.any(bmax > part_max + tol_mm)
        )

    connected = [v for v in connected if _protrudes(v)]

    if not connected:
        return

    # Name-based first pass: many STEP assemblies name sprues/runners/gates
    # explicitly; this lets us avoid treating a distributor/curufluk as a runner.
    def _type_from_name(name: str) -> Optional[BodyType]:
        n = name.lower().replace("ı", "i").replace("ğ", "g").replace("ü", "u").replace("ş", "s").replace("ö", "o").replace("ç", "c").replace(" ", "_").replace("-", "_")
        if any(k in n for k in ("curuf", "slag", "trap", "tuzak", "curufluk")):
            return BodyType.CURUFLUK
        if any(k in n for k in ("dagitici", "distributor", "manifold", "meme_dagitici", "meme_dagitici", "subap")):
            return BodyType.DISTRIBUTOR
        if any(k in n for k in ("meme", "ingate", "gate", "cikis", "giris")) and "dagitici" not in n:
            return BodyType.INGATE
        if any(k in n for k in ("yolluk", "runner", "channel")):
            return BodyType.RUNNER
        if any(k in n for k in ("dokum_agzi", "dokumagzi", "sprue", "pouring_cup", "basin", "throat")):
            return BodyType.SPRUE
        if any(k in n for k in ("besleyici", "riser", "feeder", "feed")):
            return BodyType.RISER
        return None

    name_assigned: set = set()
    for v in connected:
        t = _type_from_name(bodies[v].name)
        if t is not None and t != BodyType.PART:
            bodies[v].body_type = t
            name_assigned.add(v)

    # Identify the riser.  First respect any name-based riser, then try the
    # gravity-opposite direction; if that fails, look for a body that is strongly
    # off the main gating axis (e.g. a side/top riser in a horizontal gating system).
    riser_idx: Optional[int] = None
    for v in connected:
        if bodies[v].body_type == BodyType.RISER:
            riser_idx = v
            break

    if riser_idx is None:
        max_up = 0.2 * part_min_dim
        for v in connected:
            if bodies[v].body_type != BodyType.PART:
                continue
            vec = bodies[v].center - part_center
            proj = float(np.dot(vec, up))
            if proj > max_up:
                max_up = proj
                riser_idx = v

    if riser_idx is None:
        # Fallback: the riser is the outlier perpendicular to the dominant
        # direction of the remaining auxiliary bodies.
        gating_candidates = [v for v in connected if bodies[v].body_type == BodyType.PART]
        if len(gating_candidates) >= 2:
            centered = np.vstack([bodies[v].center - part_center for v in gating_candidates])
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            main_axis = eigvecs[:, int(np.argmax(eigvals))]
            main_axis /= float(np.linalg.norm(main_axis)) + 1e-12
            best_score = 0.0
            for v in gating_candidates:
                vec = bodies[v].center - part_center
                proj_main = float(np.dot(vec, main_axis))
                residual = float(np.linalg.norm(vec - proj_main * main_axis))
                # A riser is far from the gating line and not far along it.
                if residual > 0.5 * part_min_dim and residual > 2.0 * abs(proj_main):
                    if residual > best_score:
                        best_score = residual
                        riser_idx = v

    gating = [v for v in connected if v != riser_idx and bodies[v].body_type == BodyType.PART]
    if riser_idx is not None:
        bodies[riser_idx].body_type = BodyType.RISER

    if not gating:
        return

    # Robust gating chain ordering: combine vertical position (metal flows
    # along gravity) and horizontal distance from the part.  A gate is low and
    # close to the part; a sprue is high and far.  This works for both vertical
    # and horizontal gating layouts, and it naturally handles parallel gates.
    gating_centers = np.vstack([bodies[v].center for v in gating])
    centered = gating_centers - part_center
    vert = centered @ up
    v_min, v_max = float(vert.min()), float(vert.max())
    v_range = max(v_max - v_min, 1e-9)
    vert_score = {v: float((vert[i] - v_min) / v_range) for i, v in enumerate(gating)}

    # Horizontal distance from the part centre, measured in the plane
    # perpendicular to gravity.
    horiz = centered - np.outer(vert, up)
    r = np.linalg.norm(horiz, axis=1)
    r_min, r_max = float(r.min()), float(r.max())
    r_range = max(r_max - r_min, 1e-9)
    r_score = {v: float((r[i] - r_min) / r_range) for i, v in enumerate(gating)}

    # Higher score -> farther upstream (sprue).  Lower score -> closer to part
    # and downstream (gate).  70 % weight on height, 30 % on radial distance.
    score = {v: 0.7 * vert_score[v] + 0.3 * r_score[v] for v in gating}
    sorted_by_score = sorted(gating, key=lambda v: score[v])

    gating_volumes = [bodies[v].volume_cm3 for v in sorted_by_score]
    max_gating_vol = max(gating_volumes)
    median_gating_vol = float(np.median(gating_volumes))
    sprue_min_vol = max(0.05 * max_gating_vol, 1.0)
    gate_max_vol = median_gating_vol

    score_min = score[sorted_by_score[0]]
    score_max = score[sorted_by_score[-1]]
    score_range = max(score_max - score_min, 1e-9)
    ingate_threshold = score_min + 0.40 * score_range

    if len(gating) == 1:
        bodies[sorted_by_score[0]].body_type = BodyType.INGATE
        return

    # The farthest upstream body is the sprue only if it is large enough to be
    # the metal entry.  A tiny body at the far end is more likely a remote gate
    # (or there is no distinct sprue in the model).
    sprue_idx = None
    if bodies[sorted_by_score[-1]].volume_cm3 >= sprue_min_vol:
        sprue_idx = sorted_by_score[-1]
        bodies[sprue_idx].body_type = BodyType.SPRUE

    # Gate candidates are the small bodies within the downstream score band.
    ingate_candidates = [
        v
        for v in sorted_by_score
        if score[v] <= ingate_threshold and bodies[v].volume_cm3 <= gate_max_vol
    ]

    if ingate_candidates:
        ingate_indices = ingate_candidates[:]
    elif sprue_idx is not None:
        # No small downstream body found; the closest candidate becomes the gate.
        ingate_indices = [sorted_by_score[0]]
    else:
        # No distinct sprue: the far-end small body is treated as the gate
        # (runner -> remote gate layout).
        ingate_indices = [
            max(
                sorted_by_score,
                key=lambda v: score[v] if bodies[v].volume_cm3 <= gate_max_vol else -1e9,
            )
        ]
        # If every body is larger than the median gate size, fall back to the
        # smallest body overall.
        if bodies[ingate_indices[0]].volume_cm3 > gate_max_vol:
            ingate_indices = [min(sorted_by_score, key=lambda v: bodies[v].volume_cm3)]

    for v in ingate_indices:
        if bodies[v].body_type == BodyType.PART:
            bodies[v].body_type = BodyType.INGATE

    # Build adjacency among the gating bodies so a manifold feeding multiple
    # gates can be recognized as a distributor.
    gating_mins = [mins[v] for v in gating]
    gating_maxs = [maxs[v] for v in gating]
    gating_adj: List[List[int]] = [[] for _ in gating]
    idx_in_gating = {v: i for i, v in enumerate(gating)}
    for i, vi in enumerate(gating):
        for j, vj in enumerate(gating[i + 1 :], start=i + 1):
            if _bboxes_overlap_or_close(
                gating_mins[i], gating_maxs[i], gating_mins[j], gating_maxs[j], tol_mm
            ):
                gating_adj[i].append(j)
                gating_adj[j].append(i)

    # Remaining intermediate bodies are runners by default.
    # A body that lies upstream of two or more ingates is a distributor
    # (manifold), regardless of strict bbox overlap.
    ingate_set = set(ingate_indices)
    for v in sorted_by_score:
        if bodies[v].body_type != BodyType.PART:
            continue
        downstream_ingates = sum(
            1 for ig in ingate_set if score[ig] < score[v]
        )
        if downstream_ingates >= 2:
            bodies[v].body_type = BodyType.DISTRIBUTOR
        else:
            bodies[v].body_type = BodyType.RUNNER

    # Note: curufluk/slag-trap auto-detection is intentionally conservative.
    # Use the body name ("curuf", "slag", "trap", "curufluk") or assign it
    # explicitly in the STEP assembly; otherwise it defaults to RUNNER/DISTRIBUTOR.


def _voxelize_at_dim(
    bodies: List[Body],
    target_dim: int,
    margin: int,
    progress_callback: Optional[callable],
    fix_mesh: bool,
) -> Tuple[np.ndarray, np.ndarray, float, List[Body]]:
    """Single-shot voxelization used by build_voxel_grid."""
    bbox_min, bbox_max = _global_bbox(bodies)
    bbox_size = bbox_max - bbox_min
    dx = float(np.max(bbox_size) / target_dim)
    if dx <= 0:
        raise ValueError("Geçersiz bounding box.")

    grid_shape = np.ceil((bbox_size + 2 * margin * dx) / dx).astype(int)
    origin = bbox_min - margin * dx
    grid = np.zeros(grid_shape, dtype=np.int16)

    repaired_bodies: List[Body] = []
    for idx, body in enumerate(bodies):
        if progress_callback:
            progress_callback(int((idx / len(bodies)) * 50))

        mesh = body.mesh.copy()
        if fix_mesh:
            mesh.fill_holes()
            mesh.merge_vertices()
            mesh.remove_unreferenced_vertices()

        if len(mesh.faces) == 0:
            continue

        voxelized = trimesh.voxel.creation.voxelize(mesh, pitch=dx)
        if voxelized is None:
            continue

        # Solid voxelization
        voxelized = voxelized.fill()
        matrix = voxelized.matrix
        if not np.any(matrix):
            continue

        # Trimesh VoxelGrid origin is in the 4x4 transform
        local_origin = voxelized.transform[:3, 3].astype(np.float64)

        offset = (local_origin - origin) / dx
        offset_i = int(round(offset[0]))
        offset_j = int(round(offset[1]))
        offset_k = int(round(offset[2]))

        mi, mj, mk = matrix.shape
        i0 = max(0, offset_i)
        i1 = min(grid.shape[0], offset_i + mi)
        j0 = max(0, offset_j)
        j1 = min(grid.shape[1], offset_j + mj)
        k0 = max(0, offset_k)
        k1 = min(grid.shape[2], offset_k + mk)

        if i0 >= i1 or j0 >= j1 or k0 >= k1:
            continue

        li0 = i0 - offset_i
        li1 = li0 + (i1 - i0)
        lj0 = j0 - offset_j
        lj1 = lj0 + (j1 - j0)
        lk0 = k0 - offset_k
        lk1 = lk0 + (k1 - k0)

        region = matrix[li0:li1, lj0:lj1, lk0:lk1]
        mask = region.astype(bool)

        # Later body wins on overlap
        grid[i0:i1, j0:j1, k0:k1][mask] = int(body.body_type)

        repaired_bodies.append(body)

    if progress_callback:
        progress_callback(50)

    return grid, origin, dx, repaired_bodies


def build_voxel_grid(
    bodies: List[Body],
    target_dim: int = BASE_RES,
    progress_callback: Optional[callable] = None,
    fix_mesh: bool = True,
    gravity_vector: Tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> Tuple[np.ndarray, np.ndarray, float, List[Body]]:
    """
    Build a global voxel grid.

    The grid is padded with a 4-voxel empty border to avoid boundary clipping
    of SDF/gradient calculations.  The resolution is automatically increased if
    the chosen voxel size exceeds one third of the minimum wall thickness
    (Nyquist criterion for thin-wall feeding paths).

    If every body is still ``BodyType.PART`` (typical for a raw STEP import),
    a heuristic classifier is run first to distinguish casting, riser, sprue,
    runner and ingate solids using ``gravity_vector``.

    Returns
    -------
    grid : np.ndarray[int]
        Material id grid (Nx, Ny, Nz).
    origin_mm : np.ndarray
        World coordinate of grid[0,0,0].
    dx_mm : float
        Voxel pitch.
    bodies : List[Body]
        Bodies with repaired meshes.
    """
    if not bodies:
        raise ValueError("Voxelize edilecek body yok.")

    _classify_casting_bodies(bodies, gravity_vector=gravity_vector)

    bbox_min, bbox_max = _global_bbox(bodies)
    bbox_size = bbox_max - bbox_min
    margin = 4

    grid, origin, dx, repaired_bodies = _voxelize_at_dim(
        bodies, target_dim, margin, progress_callback, fix_mesh
    )

    # Resolution sanity check: dx must be <= t_min / 3 to capture thin walls.
    is_metal = grid != 0
    if is_metal.any():
        sdf_grid = ndimage.distance_transform_edt(is_metal) * dx
        min_sdf = float(sdf_grid[is_metal].min())
        t_min = 2.0 * min_sdf
        if t_min > 0.0 and dx > t_min / 3.0:
            required_dim = int(np.ceil(np.max(bbox_size) / (t_min / 3.0)))
            if required_dim > target_dim and required_dim <= MAX_RES:
                warnings.warn(
                    f"Voxel pitch {dx:.3f} mm > t_min/3 ({t_min/3.0:.3f} mm). "
                    f"Re-voxelizing at dimension {required_dim} to satisfy Nyquist criterion."
                )
                target_dim = required_dim
                grid, origin, dx, repaired_bodies = _voxelize_at_dim(
                    bodies, target_dim, margin, progress_callback, fix_mesh
                )
            elif required_dim > MAX_RES:
                warnings.warn(
                    f"Tavsiye edilen çözünürlük ({required_dim}) MAX_RES ({MAX_RES}) aşıyor. "
                    f"İnce cidarlar (t_min ≈ {t_min:.2f} mm) voxel ağında kopabilir."
                )

    return grid, origin, dx, repaired_bodies


def build_part_grid(
    bodies: List[Body],
    target_voxels: int = 10_000_000,
    max_dim: int = 600,
    margin_vox: int = 4,
) -> Tuple[np.ndarray, np.ndarray, float, List[Body]]:
    """Build a high-resolution voxel grid containing the PART and connected
    casting-metal bodies (risers, runners, gates, sprues, pouring basin).

    The grid resolution is chosen so that the total number of voxels is close to
    ``target_voxels`` while respecting the part's aspect ratio.  Including the
    gating/riser geometry is essential for accurate hot-spot connectivity and
    feeding-distance calculations at high resolution.
    """
    part_bodies = [b for b in bodies if b.body_type == BodyType.PART]
    if not part_bodies:
        return build_voxel_grid(bodies, target_dim=BASE_RES)

    # Use the part bbox to anchor the resolution, then include nearby casting
    # metal bodies so that sprue/runner/riser thermal mass and connectivity
    # are visible to the high-resolution hotspot detector.
    part_bbox_min, part_bbox_max = _global_bbox(part_bodies)
    part_size = part_bbox_max - part_bbox_min
    part_max_size = float(part_size.max())
    if part_max_size <= 0.0:
        return build_voxel_grid(bodies, target_dim=BASE_RES)

    # Keep casting-metal bodies within one part-size of the part bbox.  This
    # preserves the fine part resolution while still capturing connected gating.
    padding = part_max_size
    padded_min = part_bbox_min - padding
    padded_max = part_bbox_max + padding
    casting_bodies = [b for b in bodies if b.body_type in BODY_METAL_TYPES]
    nearby_casting = [
        b
        for b in casting_bodies
        if _bboxes_overlap_or_close(
            padded_min, padded_max, b.mesh.bounds[0], b.mesh.bounds[1], tol=0.0
        )
    ]
    all_bodies = part_bodies + [b for b in nearby_casting if b not in part_bodies]

    bbox_min, bbox_max = _global_bbox(all_bodies)
    size = bbox_max - bbox_min
    max_size = float(size.max())
    if max_size <= 0.0:
        return build_voxel_grid(bodies, target_dim=BASE_RES)

    volume = float(np.prod(size))
    if volume > 0.0:
        # Choose part_dim so that total voxels ~= target_voxels.
        part_dim = int(round((target_voxels * max_size ** 3 / volume) ** (1.0 / 3.0)))
    else:
        part_dim = int(round(target_voxels ** (1.0 / 3.0)))

    # Clamp resolution: avoid impossibly fine voxels and enforce a ceiling.
    part_dim = max(60, min(part_dim, max_dim))
    dx = max_size / part_dim
    # Ensure at least 0.05 mm voxel pitch (finer is usually overkill and slow).
    if dx < 0.05:
        part_dim = int(round(max_size / 0.05))
        part_dim = max(60, min(part_dim, max_dim))

    return build_voxel_grid(all_bodies, target_dim=part_dim, progress_callback=None)
