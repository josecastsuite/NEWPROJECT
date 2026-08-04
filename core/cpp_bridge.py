"""Bridge to the C++ Josecast core (OpenVDB + AMGCL + nanobind)."""

import importlib.util
import os
import sys
import sysconfig
import numpy as np
import trimesh

from core.gate_mesh import _safe_fill_holes
from typing import Any, Dict, List, Tuple, Optional, Callable

from core.types import Body, BodyType, GatingNode


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

    # Try the canonical package import first so we never load two copies of
    # the same binary (nanobind type registration fails if the same .so is
    # imported both as ``josecast_core`` and as ``core.josecast_core``).
    last_error = ""
    try:
        import core.josecast_core
        return core.josecast_core, ""
    except Exception as exc:
        last_error = f"import core.josecast_core failed: {exc}"

    # Legacy / development fallback: build tree or bare .so/.pyd on sys.path.
    try:
        import josecast_core
        return josecast_core, ""
    except Exception as exc:
        last_error += f"; import josecast_core failed: {exc}"

    # Last resort: explicit file load.
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
    _safe_fill_holes(mesh)
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


_GATING_BODY_TYPES = {
    BodyType.SPRUE_THROAT,
    BodyType.SPRUE,
    BodyType.RUNNER,
    BodyType.DISTRIBUTOR,
    BodyType.CURUFLUK,
    BodyType.INGATE,
    BodyType.POURING_BASIN,
    BodyType.COOLING_SPRUE,
    BodyType.FILTER,
}


def _parse_node_name(name: str) -> Tuple[str, str]:
    if " → " in name:
        up, down = name.split(" → ", 1)
        return up.strip(), down.strip()
    return "", ""


def _build_gating_branches(nodes: List[Any]) -> List[List[int]]:
    """Return every source -> ingate path as a list of node indices."""
    if not nodes:
        return []
    down_to_node: Dict[str, int] = {}
    source_node: Optional[int] = None
    for i, n in enumerate(nodes):
        name = getattr(n, "name", "")
        if " → " not in name:
            continue
        up, down = _parse_node_name(name)
        if up == "Kaynak":
            source_node = i
        down_to_node[down] = i

    up_names = set()
    for n in nodes:
        name = getattr(n, "name", "")
        if " → " in name:
            up_names.add(_parse_node_name(name)[0])

    leaves: List[int] = []
    for i, n in enumerate(nodes):
        name = getattr(n, "name", "")
        if " → " not in name or i == source_node:
            continue
        down = _parse_node_name(name)[1]
        if down not in up_names:
            leaves.append(i)

    branches: List[List[int]] = []
    for leaf in leaves:
        path = [leaf]
        while True:
            name = getattr(nodes[path[0]], "name", "")
            if " → " not in name:
                break
            up = _parse_node_name(name)[0]
            if up == "Kaynak" or up not in down_to_node:
                break
            path.insert(0, down_to_node[up])
        branches.append(path)

    if not branches and source_node is not None:
        branches = [[source_node] + [i for i in range(len(nodes)) if i != source_node]]
    return branches


def compute_analytic_flow_velocity(
    result: Any,
    bodies: List[Body],
    body_index: np.ndarray,
    origin_mm: np.ndarray,
    dx_mm: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Build a single-source gate velocity field for UI visualisation.

    Uses the V8.1 engine: CGAL mean-curvature-flow centerlines, exact CAD
    cross-sectional areas, V8.1 guards, and area-weighted flow rates.  The
    returned ``velocity_magnitude`` is used both for surface colour and for
    node labels, so labels and colours always read the same value.
    """
    if JOSECAST_CORE is None or not hasattr(JOSECAST_CORE, "extract_skeleton"):
        return None

    flow = getattr(result, "flow_result", None)
    if flow is None:
        return None
    nodes = list(getattr(flow, "gating_nodes", None) or [])
    if not nodes:
        return None

    from core.gating_velocity_engine import GateVelocityEngine, EngineConfig

    q_total = float(getattr(flow, "Q_m3_s", 0.0))
    cfg = EngineConfig(sample_spacing_mm=max(dx_mm, 1.0))
    engine = GateVelocityEngine(cfg)

    casting_params = getattr(result, "casting_params", None)
    user_velocity_m_s = float(getattr(casting_params, "ingate_velocity_m_s", 0.0) or 0.0)
    velocity_section_key = getattr(casting_params, "velocity_section_key", None)

    try:
        velocity, _colors = engine.compute(
            bodies,
            body_index,
            origin_mm,
            dx_mm,
            nodes,
            q_total,
            user_velocity_m_s=user_velocity_m_s,
            velocity_section_key=velocity_section_key,
        )
    except Exception as exc:
        print(f"[cpp_bridge] V8.1 gate velocity engine failed: {exc}", file=sys.stderr)
        return None
    # pois kept for compatibility; it is not used by the viewer.
    return velocity, velocity
