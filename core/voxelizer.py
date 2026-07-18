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
