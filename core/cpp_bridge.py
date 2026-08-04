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
    """Build a conformal spline-tube velocity field for UI visualisation.

    Returns ``(velocity_bulk, velocity_poiseuille)`` arrays with the same shape
    as ``result.grid``.  The bulk array is intended for surface colouring and
    matches the gating-node label values at the node centroids.
    """
    if JOSECAST_CORE is None or not hasattr(JOSECAST_CORE, "solve_spline_tube_field"):
        return None

    flow = getattr(result, "flow_result", None)
    if flow is None:
        return None
    nodes = list(getattr(flow, "gating_nodes", None) or [])
    if not nodes:
        return None
    if not hasattr(result, "sdf") or result.sdf.size == 0:
        return None
    sdf = result.sdf
    if sdf.shape != body_index.shape:
        return None

    # Per-body 3-D gate-flow summaries; used to assign sensible Q/A values to
    # gating bodies that are not on the main source->ingate path.
    gate_summaries: Dict[str, Dict[str, float]] = getattr(flow, "gate_flow_results", {}) or {}

    branches = _build_gating_branches(nodes)
    if not branches:
        return None

    # ------------------------------------------------------------------
    # Map body names referenced by gating nodes to their branch.
    # A body name may appear as the upstream or downstream end of a node;
    # the source body is shared by all branches and can use any of them.
    # ------------------------------------------------------------------
    body_name_to_idx: Dict[str, int] = {
        b.name: i for i, b in enumerate(bodies) if b.name
    }
    node_body_to_branch: Dict[str, int] = {}
    for bi, branch in enumerate(branches):
        for ni in branch:
            up, down = _parse_node_name(getattr(nodes[ni], "name", ""))
            if up and up not in node_body_to_branch and up in body_name_to_idx:
                node_body_to_branch[up] = bi
            if down and down not in node_body_to_branch and down in body_name_to_idx:
                node_body_to_branch[down] = bi

    # ------------------------------------------------------------------
    # Augment the gating-node list with one-node "branches" for any gate
    # body that is not referenced by the main gating graph.  This lets the
    # analytic solver paint those bodies with a physically consistent local
    # velocity from the per-body 3-D gate-flow summary (Q/A) instead of
    # snapping them to an unrelated branch.
    # ------------------------------------------------------------------
    synthetic_nodes: List[GatingNode] = []
    synthetic_branches: List[List[int]] = []
    synthetic_body_name: List[str] = []
    for idx, body in enumerate(bodies):
        if body.name in node_body_to_branch:
            continue
        if body.body_type not in _GATING_BODY_TYPES:
            continue
        summary = gate_summaries.get(body.name, {})
        v = float(summary.get("section_velocity_m_s", 0.0))
        q = float(summary.get("outlet_flux_m3_s", 0.0))
        if v <= 1e-18 or q <= 1e-18:
            # Fallback to geometric / user-specified area and a nearby node
            # velocity if the 3-D gate summary is unavailable.
            v = 0.0
            q = 0.0
            for n in nodes:
                nv = getattr(n, "max_velocity_m_s", 0.0)
                if nv > v:
                    v = nv
                    q = getattr(n, "flow_rate_m3_s", 0.0)
        a_cm2 = 0.0
        if v > 1e-18 and q > 1e-18:
            a_cm2 = float(q / v) * 1e4
        if a_cm2 <= 1e-12:
            a_cm2 = float(getattr(body, "section_area_cm2", 0.0))
        if a_cm2 <= 1e-12:
            continue
        centroid = getattr(body, "center", None)
        if centroid is None or not isinstance(centroid, np.ndarray):
            mask = body_index == idx
            if mask.any():
                coords = np.argwhere(mask)
                centroid = (coords.mean(axis=0) + 0.5) * dx_mm + origin_mm
            else:
                centroid = np.zeros(3)
        synth = GatingNode(
            name=body.name,
            body_type=body.body_type.name,
            velocity_m_s=v,
            section_area_cm2=float(a_cm2),
            centroid_mm=tuple(float(x) for x in centroid),
            flow_rate_m3_s=q,
            max_velocity_m_s=v,
        )
        synth_idx = len(nodes) + len(synthetic_nodes)
        synthetic_nodes.append(synth)
        synthetic_branches.append([synth_idx])
        synthetic_body_name.append(body.name)

    all_nodes = nodes + synthetic_nodes
    all_branches = branches + synthetic_branches

    n = len(all_nodes)
    centroids = np.zeros((n, 3), dtype=np.float64)
    velocity = np.zeros(n, dtype=np.float64)
    area = np.zeros(n, dtype=np.float64)
    for i, node in enumerate(all_nodes):
        centroids[i] = getattr(node, "centroid_mm", (0.0, 0.0, 0.0))
        v = getattr(node, "max_velocity_m_s", 0.0)
        if v <= 1e-12:
            v = getattr(node, "velocity_m_s", 0.0)
        velocity[i] = v
        area[i] = float(getattr(node, "section_area_cm2", 0.0)) * 1e-4

    branch_offsets = np.zeros(len(all_branches) + 1, dtype=np.int32)
    flat_nodes: List[int] = []
    for bi, b in enumerate(all_branches):
        flat_nodes.extend(int(i) for i in b)
        branch_offsets[bi + 1] = len(flat_nodes)
    branch_node_indices = np.asarray(flat_nodes, dtype=np.int32)

    # ------------------------------------------------------------------
    # Build the per-voxel branch map.  Node-referenced bodies use the node
    # branch so the colour at a node centroid equals the label.  Synthetic
    # (single-node) bodies get their own branch.  A body may be referenced
    # by multiple branches; the first one is used (shared source segments are
    # identical anyway).
    # ------------------------------------------------------------------
    voxel_branch = np.full(body_index.shape, -1, dtype=np.int32)
    for idx, body in enumerate(bodies):
        mask = body_index == idx
        if not mask.any():
            continue
        branch: Optional[int] = None
        if body.name in node_body_to_branch:
            branch = node_body_to_branch[body.name]
        elif body.name in synthetic_body_name:
            branch = len(branches) + synthetic_body_name.index(body.name)
        if branch is None:
            continue
        voxel_branch[mask] = branch

    try:
        bulk, pois = JOSECAST_CORE.solve_spline_tube_field(
            np.ascontiguousarray(sdf.astype(np.float64, copy=False)),
            np.ascontiguousarray(voxel_branch.astype(np.int32, copy=False)),
            float(dx_mm),
            np.asarray(origin_mm, dtype=np.float64).tolist(),
            np.ascontiguousarray(centroids),
            velocity,
            area,
            branch_node_indices,
            branch_offsets,
        )
    except Exception as exc:
        print(f"[cpp_bridge] solve_spline_tube_field failed: {exc}", file=sys.stderr)
        return None
    return bulk, pois
