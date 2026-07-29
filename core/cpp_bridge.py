"""Bridge to the C++ Josecast core (OpenVDB + AMGCL + nanobind)."""

import os
import sys
import numpy as np
import trimesh
from typing import List, Tuple, Optional, Callable

from core.types import Body, BodyType


def _find_core_module():
    """Locate the compiled josecast_core shared module."""
    # When running from the repo, the build tree is expected under cpp/build/src.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(repo_root, "cpp", "build", "src"),
    ]
    for p in candidates:
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import josecast_core
        return josecast_core
    except ImportError:
        return None


JOSECAST_CORE = _find_core_module()


def has_cpp_core() -> bool:
    return JOSECAST_CORE is not None


def _repair_mesh(body: Body) -> Body:
    mesh = body.mesh.copy()
    mesh.process(validate=True, merge_tex=True, merge_norm=True)
    mesh.fill_holes()
    mesh.remove_unreferenced_vertices()
    body.mesh = mesh
    body.vertices = mesh.vertices.copy()
    body.faces = mesh.faces.copy()
    body.center = mesh.center_mass if mesh.is_watertight else mesh.centroid
    if mesh.is_watertight:
        body.volume_cm3 = float(mesh.volume) / 1000.0
    body.surface_area_cm2 = float(mesh.area) / 100.0
    return body


def build_voxel_grid_cpp(
    bodies: List[Body],
    target_dim: int = 160,
    progress_callback: Optional[Callable] = None,
    fix_mesh: bool = True,
    gravity_vector: Tuple[float, float, float] = (0.0, 0.0, -1.0),
    conservative: bool = True,
    margin: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, List[Body]]:
    """C++ OpenVDB voxelizer drop-in replacement for build_voxel_grid."""
    if JOSECAST_CORE is None:
        raise RuntimeError("C++ core not built. Run: cd cpp && cmake -B build && cmake --build build")

    # Bodies are expected to have been classified already by the caller.
    mins = np.vstack([b.mesh.bounds[0] for b in bodies])
    maxs = np.vstack([b.mesh.bounds[1] for b in bodies])
    bbox_min = mins.min(axis=0)
    bbox_max = maxs.max(axis=0)
    bbox_size = bbox_max - bbox_min

    max_size = float(bbox_size.max())
    dx = max_size / float(target_dim)

    grid_shape = np.ceil((bbox_size + 2 * margin * dx) / dx).astype(int)
    origin = bbox_min - margin * dx

    vox = JOSECAST_CORE.Voxelizer()
    repaired_bodies: List[Body] = []
    for idx, body in enumerate(bodies):
        if progress_callback:
            progress_callback(int((idx / len(bodies)) * 50))
        if fix_mesh:
            body = _repair_mesh(body)
        mesh = body.mesh
        if len(mesh.faces) == 0:
            continue
        verts = np.ascontiguousarray(mesh.vertices.astype(np.float32))
        faces = np.ascontiguousarray(mesh.faces.astype(np.int32))
        vox.add_body(int(idx), int(body.body_type), verts, faces)
        repaired_bodies.append(body)

    if progress_callback:
        progress_callback(50)

    grid, body_index, _sdf = vox.build(float(dx), origin.tolist(), grid_shape.tolist())
    grid = grid.astype(np.int16)
    body_index = body_index.astype(np.int32)

    # Conservative dilation matches the Python voxelizer's thin-wall handling.
    if conservative:
        from scipy import ndimage
        is_metal = grid != 0
        if is_metal.any():
            is_metal = ndimage.binary_dilation(is_metal, structure=np.ones((3, 3, 3), dtype=bool))
            grid = np.where(is_metal, grid, 0)
            body_index = np.where(is_metal, body_index, -1)

    return grid, body_index, origin, dx, repaired_bodies
