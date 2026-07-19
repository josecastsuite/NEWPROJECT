"""Convert a set of Body meshes into a single labelled voxel grid - JoseCast v7."""

from typing import List, Optional, Tuple

import numpy as np
import trimesh

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
        return 1.0
    for b in bodies:
        b.mesh.apply_scale(scale)
        b.vertices = b.mesh.vertices.copy()
        b.center = b.mesh.center_mass if b.mesh.is_watertight else b.mesh.centroid
        b.volume_cm3 = b.mesh.volume / 1000.0
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


def build_voxel_grid(
    bodies: List[Body],
    target_dim: int = BASE_RES,
    progress_callback: Optional[callable] = None,
    fix_mesh: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, List[Body]]:
    """
    Build a global voxel grid.

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

    bbox_min, bbox_max = _global_bbox(bodies)
    bbox_size = bbox_max - bbox_min
    dx = float(np.max(bbox_size) / target_dim)
    if dx <= 0:
        raise ValueError("Geçersiz bounding box.")

    # Add a small empty border
    margin = 4
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
