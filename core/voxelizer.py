"""Convert a set of Body meshes into a single labelled voxel grid."""

from typing import List, Optional, Tuple

import numpy as np
import trimesh
from scipy import ndimage

from core.types import Body, BodyType


def _global_bbox(bodies: List[Body], padding: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Return (min, max) of all body vertices in mm with a small integer-voxel padding."""
    mins = np.vstack([b.mesh.bounds[0] for b in bodies])
    maxs = np.vstack([b.mesh.bounds[1] for b in bodies])
    return mins.min(axis=0), maxs.max(axis=0)


def build_voxel_grid(
    bodies: List[Body],
    target_dim: int = 96,
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
        Bodies with possibly repaired meshes.
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
            # Fill small holes so the voxelization is watertight
            mesh.fill_holes()
            mesh.merge_vertices()

        if len(mesh.faces) == 0:
            continue

        # Trimesh voxelization returns a VoxelGrid object.
        voxelized = trimesh.voxel.creation.voxelize(mesh, pitch=dx)
        if voxelized is None:
            continue

        # By default Trimesh voxelizes the surface. .fill() makes it a solid grid.
        voxelized = voxelized.fill()
        matrix = voxelized.matrix
        if not np.any(matrix):
            continue

        # Trimesh VoxelGrid stores the origin in the 4x4 transform.
        local_origin = voxelized.transform[:3, 3].astype(np.float64)

        # Map local matrix indices into global grid.
        # global coord = local_origin + (i,j,k)*dx
        # global index = (coord - origin) / dx
        offset = (local_origin - origin) / dx
        offset_i = int(round(offset[0]))
        offset_j = int(round(offset[1]))
        offset_k = int(round(offset[2]))

        mi, mj, mk = matrix.shape
        # Clip to grid bounds
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

        # Overlapping bodies: later one wins; but in a proper STEP they should not overlap.
        grid[i0:i1, j0:j1, k0:k1][mask] = int(body.body_type)

        repaired_bodies.append(body)

    if progress_callback:
        progress_callback(50)

    return grid, origin, dx, repaired_bodies
