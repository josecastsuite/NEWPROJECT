"""Bridge to the C++ Josecast core (OpenVDB + AMGCL + nanobind)."""

import importlib.util
import os
import sys
import sysconfig
import numpy as np
import trimesh
from typing import List, Tuple, Optional, Callable

from core.types import Body, BodyType


def _find_core_module():
    """Locate the compiled josecast_core shared module (.so / .pyd)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    core_dir = os.path.dirname(os.path.abspath(__file__))

    # On Windows, make sure dependent DLLs in core/ and win_dlls/ are on the
    # search path before attempting to import the extension.
    if os.name == "nt":
        for dll_dir in (core_dir, os.path.join(repo_root, "win_dlls")):
            if os.path.isdir(dll_dir):
                try:
                    os.add_dll_directory(dll_dir)
                except (AttributeError, OSError):
                    pass

    # Insert candidate directories on sys.path for a normal import.
    candidates = [
        os.path.join(repo_root, "cpp", "build", "src"),
        core_dir,
    ]
    for p in candidates:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Try a normal import first (handles .so / matching .pyd already on path).
    last_error = ""
    try:
        import josecast_core
        return josecast_core, ""
    except Exception as exc:
        last_error = f"import josecast_core failed: {exc}"

    # Fallback: look for a .pyd/.so with a compatible ABI tag and load it
    # explicitly. This helps when the pre-built Windows artifact name does not
    # match what a simple `import` expects.
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or (
        ".pyd" if os.name == "nt" else ".so"
    )
    for base in candidates:
        if not os.path.isdir(base):
            continue
        for fname in sorted(os.listdir(base), reverse=True):
            if fname.startswith("josecast_core") and (
                fname.endswith(".pyd") or fname.endswith(".so")
            ):
                fpath = os.path.join(base, fname)
                try:
                    spec = importlib.util.spec_from_file_location(
                        "josecast_core", fpath
                    )
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        return mod, ""
                except Exception as exc:
                    last_error = f"{fpath}: {exc}"
                    print(f"[cpp_bridge] Could not load {fpath}: {exc}", file=sys.stderr)
    return None, last_error


_JOSECAST_CORE_LOAD_ERROR = ""
_JOSECAST_CORE_OBJ = _find_core_module()
if isinstance(_JOSECAST_CORE_OBJ, tuple):
    JOSECAST_CORE, _JOSECAST_CORE_LOAD_ERROR = _JOSECAST_CORE_OBJ
else:
    JOSECAST_CORE, _JOSECAST_CORE_LOAD_ERROR = None, str(_JOSECAST_CORE_OBJ)


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
        raise RuntimeError(
            f"C++ core not built. Python {sys.version}\nLoad error: {_JOSECAST_CORE_LOAD_ERROR}\n"
            "Run: cd cpp && cmake -B build && cmake --build build, "
            "or use a pre-built josecast_core.*.pyd matching your Python ABI."
        )

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
