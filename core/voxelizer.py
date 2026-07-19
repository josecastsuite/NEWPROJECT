"""Convert a set of Body meshes into a single labelled voxel grid - JoseCast v7."""

import warnings
from typing import List, Optional, Tuple

import numpy as np
import trimesh
from scipy import ndimage

from core.types import Body, BodyType

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

    # Identify the riser.  First try the gravity-opposite direction; if that
    # fails, look for a body that is strongly off the main gating axis (e.g. a
    # side/top riser in a horizontal gating system).
    riser_idx: Optional[int] = None
    max_up = 0.2 * part_min_dim
    for v in connected:
        vec = bodies[v].center - part_center
        proj = float(np.dot(vec, up))
        if proj > max_up:
            max_up = proj
            riser_idx = v

    if riser_idx is None:
        # Fallback: the riser is the outlier perpendicular to the dominant
        # direction of the remaining auxiliary bodies.
        gating_candidates = [v for v in connected]
        centered = np.vstack([bodies[v].center - part_center for v in gating_candidates])
        if centered.shape[0] >= 2:
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

    gating = [v for v in connected if v != riser_idx]
    if riser_idx is not None:
        bodies[riser_idx].body_type = BodyType.RISER

    if not gating:
        return

    # Main gating direction: from part centroid to the average gating centroid.
    gating_centers = np.vstack([bodies[v].center for v in gating])
    flow_vec = gating_centers.mean(axis=0) - part_center
    flow_norm = float(np.linalg.norm(flow_vec))
    if flow_norm > 1e-12:
        flow_dir = flow_vec / flow_norm
    else:
        flow_dir = np.array([1.0, 0.0, 0.0])

    gating_sorted = sorted(
        gating,
        key=lambda i: float(np.dot(bodies[i].center - part_center, flow_dir)),
    )

    if len(gating_sorted) == 1:
        bodies[gating_sorted[0]].body_type = BodyType.INGATE
    else:
        bodies[gating_sorted[0]].body_type = BodyType.INGATE
        bodies[gating_sorted[-1]].body_type = BodyType.SPRUE
        for idx in gating_sorted[1:-1]:
            bodies[idx].body_type = BodyType.RUNNER


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
    """Build a high-resolution voxel grid containing only the PART bodies.

    The grid resolution is chosen so that the total number of voxels is close to
    ``target_voxels`` while respecting the part's aspect ratio.  A small margin is
    added around the part to avoid boundary clipping.
    """
    part_bodies = [b for b in bodies if b.body_type == BodyType.PART]
    if not part_bodies:
        return build_voxel_grid(bodies, target_dim=BASE_RES)

    bbox_min, bbox_max = _global_bbox(part_bodies)
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

    # build_voxel_grid adds its own margin; pass only part bodies so the bbox is tight.
    return build_voxel_grid(part_bodies, target_dim=part_dim, progress_callback=None)
