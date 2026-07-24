"""Ingate / runner / sprue geometric gating calculations - JoseCast v8.0."""

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import math

import numpy as np
import trimesh
from scipy import ndimage
from scipy.sparse import coo_matrix, csgraph
from scipy.spatial import ConvexHull

from core.gating_calculator import (
    auto_fill_time as _gc_auto_fill_time,
    calc_campbell_parameters,
    compute_gating,
    compute_modulus_and_riser as _gc_compute_modulus_and_riser,
    effective_head,
)
from core.gating_engine import (
    GatingEngineInput,
    _VELOCITY_RANGES as _ENGINE_VELOCITY_RANGES,
    _classify_from_velocities,
    _section_velocity_limit,
    calculate_gating_design,
)
from core.materials import get_alloy, get_mold, chvorinov_c_from_properties
from core.sdf_analyzer import COST_26, NEIGH_26
from core.types import (
    BODY_FEEDER_TYPES,
    BODY_METAL_TYPES,
    AnalysisResult,
    Body,
    BodyType,
    FillingResult,
    GateResult,
    SectionFlow,
)


def _neighbor_offsets_6():
    return [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]


def _map_node_velocity_to_section(key: str) -> Optional[str]:
    """Map a node-velocity key or gating-node body_type to a canonical section."""
    up = key.upper()
    if "INGATE" in up:
        return "gate"
    if "DISTRIBUTOR" in up:
        return "distributor"
    if "CURUFLUK" in up:
        return "curufluk"
    if "RUNNER" in up:
        return "runner"
    if "SPRUE" in up:
        return "sprue"
    return None


def _upstream_section_key(body_type: str) -> str:
    """Upper-case upstream section key from a 'UP→DOWN' gating node body_type."""
    up = (body_type or "").split("→")[0].strip().upper()
    return up


def _actual_velocities_from_flow(flow: FillingResult) -> Dict[str, float]:
    """Return the gating section velocities reported by the Darcy flow solver."""
    node_v = getattr(flow, "node_velocities", {}) or {}
    buckets: Dict[str, List[float]] = {}
    for key, val in node_v.items():
        section = _map_node_velocity_to_section(key)
        if not section:
            continue
        buckets.setdefault(section, []).append(float(val))
    return {k: float(np.mean(v)) for k, v in buckets.items() if v}


def _section_area_from_flow(flow: FillingResult, section: str) -> float:
    """Total contact area (cm2) for a gating section from the flow result."""
    section = section.lower()
    total = 0.0
    for node in getattr(flow, "gating_nodes", []) or []:
        up = _map_node_velocity_to_section(node.body_type or "")
        if up == section:
            total += float(node.section_area_cm2 or 0.0)
    return total


def _section_flows_from_flow(
    flow: FillingResult,
    rho: float,
    mu: float,
    g: float,
    velocity_targets: Dict[str, Tuple[float, float]],
) -> Dict[str, SectionFlow]:
    """Build SectionFlow objects directly from Darcy flow contact nodes.

    Keys match the original gating section names (SPRUE_BASE, SPRUE_THROAT,
    RUNNER, INGATE, DISTRIBUTOR, CURUFLUK).  Reynolds and Froude numbers use
    the equivalent hydraulic diameter computed from the summed contact area.
    """
    section_data: Dict[str, List[Tuple[float, float, float]]] = {}
    for node in getattr(flow, "gating_nodes", []) or []:
        up = _upstream_section_key(node.body_type or "")
        if not up:
            continue
        section_data.setdefault(up, []).append(
            (node.velocity_m_s, node.section_area_cm2, node.flow_rate_m3_s)
        )
    out: Dict[str, SectionFlow] = {}
    for up_section, rows in section_data.items():
        areas = [a for _, a, _ in rows]
        qs = [q for _, _, q in rows]
        total_area_cm2 = float(np.sum(areas))
        if total_area_cm2 <= 0.0:
            continue
        # Weighted average velocity by flow rate.
        total_q = float(np.sum(qs))
        v_m_s = (
            float(np.average([v for v, _, _ in rows], weights=qs))
            if total_q > 0.0
            else float(np.mean([v for v, _, _ in rows]))
        )
        d_m = 2.0 * math.sqrt(max(total_area_cm2 * 1e-4, 0.0) / math.pi)
        re = float(rho * v_m_s * d_m / max(mu, 1e-9))
        fr = float(v_m_s / math.sqrt(max(g * d_m, 1e-9)))
        # Map uppercase upstream key to the lower-case target range key.
        lo_hi_key = _map_node_velocity_to_section(up_section) or up_section.lower()
        lo, hi = velocity_targets.get(lo_hi_key, (0.0, 1.0))
        a_min, a_max = 0.0, 1e9
        out[up_section] = SectionFlow(
            velocity_m_s=v_m_s,
            area_cm2=total_area_cm2,
            thickness_mm=d_m * 1000.0,
            reynolds=re,
            froude=fr,
            turbulent=(re > 2300.0 or fr > 0.8),
            max_velocity_m_s=hi,
            target_v_min_m_s=lo,
            target_v_max_m_s=hi,
            target_area_min_cm2=a_min,
            target_area_max_cm2=a_max,
        )
    return out


def _apply_edge_mask(arr, di, dj, dk):
    if di > 0:
        arr[-1, :, :] = False
    elif di < 0:
        arr[0, :, :] = False
    if dj > 0:
        arr[:, -1, :] = False
    elif dj < 0:
        arr[:, 0, :] = False
    if dk > 0:
        arr[:, :, -1] = False
    elif dk < 0:
        arr[:, :, 0] = False
    return arr


def _gate_source_mask(grid: np.ndarray) -> np.ndarray:
    """Return gating bodies that can feed metal into the part.

    Filter, pouring basin, distributor and curufluk may also act as entry
    points; cooling sprue is a chill and must not be treated as a feeder.
    """
    return np.isin(
        grid,
        [BodyType.INGATE, BodyType.RUNNER, BodyType.SPRUE, BodyType.SPRUE_THROAT,
         BodyType.DISTRIBUTOR, BodyType.CURUFLUK, BodyType.FILTER, BodyType.POURING_BASIN],
    )


# v8.5: helpers from Filling_time_tr.py / gating_calculator_tr.py
# Imported directly from core.gating_calculator so the analyzer uses the exact
# equations from the user's working field scripts.

def _area_to_diameter_mm(area_cm2: float) -> float:
    """Circular equivalent diameter [mm] from area [cm2]."""
    area_m2 = area_cm2 / 1e4
    if area_m2 <= 0.0:
        return 0.0
    return 1000.0 * np.sqrt(4.0 * area_m2 / np.pi)


def _repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Repair a copy of the body mesh for cross-section calculations."""
    m = mesh.copy()
    try:
        m.fill_holes()
        m.merge_vertices()
        m.remove_unreferenced_vertices()
    except Exception:
        pass
    return m


def _flow_axis(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return the principal (longest) axis of a body."""
    pts = mesh.vertices - mesh.vertices.mean(axis=0)
    if len(pts) < 3:
        return np.array([0.0, 0.0, 1.0])
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        return np.array([0.0, 0.0, 1.0])
    return axis / norm


def _section_2d_area_and_perim(
    section: trimesh.path.Path3D,
    axis: np.ndarray,
    origin: np.ndarray,
) -> Tuple[float, float]:
    """Return area (mm²) and perimeter (mm) of a 3D section path.

    The vertices are projected onto an orthonormal basis perpendicular to
    ``axis`` and the convex-hull area is used; the perimeter comes from the
    path length (trimesh does not require shapely for this).
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (float(np.linalg.norm(axis)) + 1e-12)
    # Choose a reference vector not parallel to axis.
    tmp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(axis, tmp)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(axis, tmp)
    u = u / (float(np.linalg.norm(u)) + 1e-12)
    v = np.cross(axis, u)
    v = v / (float(np.linalg.norm(v)) + 1e-12)

    verts = section.vertices - origin
    coords = np.column_stack((verts @ u, verts @ v))
    area = 0.0
    if len(coords) >= 3:
        try:
            hull = ConvexHull(coords)
            area = float(hull.volume)
        except Exception:
            area = 0.0
    perim = float(getattr(section, "length", 0.0))
    return area, perim


def _section_profile_detailed(
    mesh: trimesh.Trimesh,
    axis: np.ndarray,
    n: int = 50,
) -> List[Tuple[float, float, float, float]]:
    """Slice a body perpendicular to its flow axis and return (t, area, perimeter, circularity).

    t is the signed distance along ``axis`` from the body centroid.
    Area and perimeter are in mm² / mm.  Circularity = 4πA / P² (1.0 for a perfect circle).
    Partial end-cap intersections may return very small areas / perimeters.
    """
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        return []
    axis = axis / norm
    center = mesh.vertices.mean(axis=0)
    pts = mesh.vertices - center
    proj = pts @ axis
    lo, hi = float(proj.min()), float(proj.max())
    if hi <= lo:
        return []
    values = np.linspace(lo, hi, max(n, 5))
    rows: List[Tuple[float, float, float, float]] = []
    for t in values:
        origin = center + axis * t
        section = mesh.section(plane_origin=origin, plane_normal=axis)
        if section is None:
            continue
        area, perim = _section_2d_area_and_perim(section, axis, origin)
        if area > 0.0 and perim > 0.0:
            circ = 4.0 * math.pi * area / (perim * perim)
        else:
            circ = 0.0
        rows.append((t, area, perim, circ))
    return rows


def _section_area_profile(
    mesh: trimesh.Trimesh,
    axis: np.ndarray,
    n: int = 50,
) -> List[float]:
    """Return only cross-sectional areas [mm2] for callers that do not need perimeter."""
    return [a for _, a, _, _ in _section_profile_detailed(mesh, axis, n=n)]


def _body_flow_length(mesh: trimesh.Trimesh, axis: np.ndarray) -> float:
    """Return the body extent along the flow axis in mm."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        return 0.0
    axis = axis / norm
    proj = (mesh.vertices - mesh.vertices.mean(axis=0)) @ axis
    return float(proj.max() - proj.min())


def _characteristic_cross_section_area(
    mesh: trimesh.Trimesh,
    axis: np.ndarray,
    n: int = 50,
) -> float:
    """Return the most representative cross-sectional area [mm2] perpendicular to axis.

    The algorithm looks for a constant (plateau) cross-section first.  If found,
    it returns the mean of that plateau; for circular plateaus it uses the equivalent
    circle area from the perimeter to compensate for tessellation coarseness.
    If no plateau exists, circular bodies are classified as conical (monotonic)
    or non-monotonic; conical uses the minimum circular area (throat), otherwise the
    maximum circular area.  Non-circular / prismatic bodies use the median area.
    """
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        return 0.0
    axis = axis / norm

    rows = _section_profile_detailed(mesh, axis, n=n)
    if not rows:
        length_mm = _body_flow_length(mesh, axis)
        if length_mm > 0.0:
            return float(mesh.volume / (length_mm * 1e-3))
        return 0.0

    t = np.array([r[0] for r in rows])
    areas = np.array([r[1] for r in rows])
    perims = np.array([r[2] for r in rows])
    circs = np.array([r[3] for r in rows])

    max_area = float(areas.max())
    if max_area <= 0.0:
        return 0.0

    best_window: Optional[Tuple[int, int]] = None
    best_score = -1.0
    min_len = 3
    for i in range(len(areas) - min_len + 1):
        for j in range(i + min_len - 1, len(areas)):
            w_areas = areas[i : j + 1]
            if w_areas.min() < 0.15 * max_area:
                continue
            if w_areas.max() / w_areas.min() > 1.25:
                continue
            score = (j - i + 1) * w_areas.mean()
            if score > best_score:
                best_score = score
                best_window = (i, j)

    if best_window is not None:
        i, j = best_window
        mean_circ = float(circs[i : j + 1].mean())
        if mean_circ > 0.85:
            return float((perims[i : j + 1] ** 2 / (4.0 * math.pi)).mean())
        return float(areas[i : j + 1].mean())

    valid = areas > 0.05 * max_area
    if not valid.any():
        return float(np.median(areas))

    mean_circ = float(circs[valid].mean())
    if mean_circ > 0.85:
        circ_areas = perims ** 2 / (4.0 * math.pi)
        x = np.arange(len(areas))
        if valid.sum() > 2:
            a_valid = areas[valid]
            x_valid = x[valid]
            cov = np.cov(x_valid, a_valid)
            if cov[0, 0] > 0.0:
                r = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
            else:
                r = 0.0
            if abs(r) > 0.65:
                interior = np.ones_like(areas, dtype=bool)
                interior[0] = interior[-1] = False
                if not (interior & valid).any():
                    interior = valid
                return float(circ_areas[interior & valid].min())
        return float(circ_areas[valid].max())

    central = areas[1:-1] if len(areas) > 2 else areas
    return float(np.median(central))


def _sprue_circular_base_and_throat(
    mesh: trimesh.Trimesh,
    axis: np.ndarray,
    n: int = 50,
) -> Tuple[float, float]:
    """Return (base_area_mm2, throat_area_mm2) for a sprue.

    ``base_area`` is the characteristic/main circular cross-section.
    ``throat_area`` is the minimum reliable circular cross-section.
    """
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        return 0.0, 0.0
    axis = axis / norm

    rows = _section_profile_detailed(mesh, axis, n=n)
    if not rows:
        length_mm = _body_flow_length(mesh, axis)
        if length_mm > 0.0:
            avg = float(mesh.volume / (length_mm * 1e-3))
            return avg, avg
        return 0.0, 0.0

    t = np.array([r[0] for r in rows])
    areas = np.array([r[1] for r in rows])
    perims = np.array([r[2] for r in rows])
    circs = np.array([r[3] for r in rows])
    circ_areas = np.where(perims > 0.0, perims ** 2 / (4.0 * math.pi), 0.0)

    max_area = float(areas.max())
    if max_area <= 0.0:
        return 0.0, 0.0

    # Use the largest contiguous region where the cross-section is well inside
    # the body (area > 30 % of max) and reasonably circular.  End-cap partial
    # intersections are excluded because they can look circular while being tiny.
    significant = (areas > 0.30 * max_area) & (circs > 0.85) & (circ_areas > 0.0)
    runs = []
    i = 0
    while i < len(areas):
        if significant[i]:
            j = i
            while j < len(areas) and significant[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1

    if runs:
        # Prefer a run that does not touch the first/last slice (avoids partials).
        good_runs = [r for r in runs if r[0] > 0 and r[1] < len(areas)]
        if not good_runs:
            good_runs = runs
        run = max(good_runs, key=lambda r: r[1] - r[0])
        i, j = run
        base = float(circ_areas[i:j].max())
        throat = float(circ_areas[i:j].min())
        return base, throat

    # Prismatic / non-circular sprue: use the median cross-sectional area.
    base = float(np.median(areas[1:-1])) if len(areas) > 2 else float(np.median(areas))
    throat = base
    return base, throat


_GATING_BODY_TYPES = frozenset([
    BodyType.SPRUE,
    BodyType.SPRUE_THROAT,
    BodyType.RUNNER,
    BodyType.DISTRIBUTOR,
    BodyType.CURUFLUK,
    BodyType.INGATE,
    BodyType.POURING_BASIN,
])


def _bboxes_overlap(
    min_a: np.ndarray,
    max_a: np.ndarray,
    min_b: np.ndarray,
    max_b: np.ndarray,
    tol: float = 2.0,
) -> bool:
    """Return True if two bounding boxes overlap or are within ``tol`` of each other."""
    return bool(np.all((max_a + tol) >= min_b) and np.all((max_b + tol) >= min_a))


def _body_cross_section_mm2(
    body: Body,
    axis: Optional[np.ndarray] = None,
) -> Tuple[float, Optional[float]]:
    """Return (main area, optional throat area) [mm2] for a gating body.

    SPRUE returns both base and throat; SPRUE_THROAT returns (throat, throat);
    other body types return (characteristic area, None).
    """
    mesh = _repair_mesh(body.mesh)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return 0.0, None
    if axis is None:
        axis = _flow_axis(mesh)
    else:
        axis = np.asarray(axis, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm <= 0.0:
            axis = _flow_axis(mesh)
        else:
            axis = axis / norm
    if body.body_type == BodyType.SPRUE:
        base_mm2, throat_mm2 = _sprue_circular_base_and_throat(mesh, axis)
        return base_mm2, throat_mm2
    if body.body_type == BodyType.SPRUE_THROAT:
        area_mm2 = _characteristic_cross_section_area(mesh, axis)
        return area_mm2, area_mm2
    area_mm2 = _characteristic_cross_section_area(mesh, axis)
    return area_mm2, None


def _build_gating_topology(
    bodies: List[Body],
    gravity_vector: Tuple[float, float, float] = (0.0, 0.0, -1.0),
    tol_mm: float = 2.0,
) -> Dict[str, object]:
    """Build a directed flow graph for the gating system.

    Nodes are bodies of ``_GATING_BODY_TYPES``.  Edges point from the body with
    the larger projection onto the ``up`` direction (opposite to gravity) toward
    the lower one.  This gives an upstream -> downstream tree that preserves
    series/parallel topology.
    """
    gating = [b for b in bodies if b.body_type in _GATING_BODY_TYPES]
    empty = {
        "bodies": [],
        "parent": {},
        "children": {},
        "order": [],
        "sources": [],
        "up": np.array([0.0, 0.0, 1.0]),
        "flow_axis": {},
        "areas_mm2": {},
        "centers": {},
    }
    if not gating:
        return empty

    n = len(gating)
    mins = [b.mesh.bounds[0] for b in gating]
    maxs = [b.mesh.bounds[1] for b in gating]
    centers = np.vstack([b.center for b in gating])
    g = np.asarray(gravity_vector, dtype=np.float64)
    g_norm = float(np.linalg.norm(g)) + 1e-12
    up = -g / g_norm
    proj = centers @ up

    # Adjacency from bounding-box proximity.
    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _bboxes_overlap(mins[i], maxs[i], mins[j], maxs[j], tol_mm):
                adj[i].append(j)
                adj[j].append(i)

    # Direct edges from higher projection to lower projection.
    incoming: List[List[int]] = [[] for _ in range(n)]
    outgoing: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in adj[i]:
            if i == j:
                continue
            if proj[i] > proj[j] + 1e-9:
                outgoing[i].append(j)
                incoming[j].append(i)
            elif proj[j] > proj[i] + 1e-9:
                outgoing[j].append(i)
                incoming[i].append(j)
            else:
                # Equal projection: stable tie-break by index.
                if i < j:
                    outgoing[i].append(j)
                    incoming[j].append(i)
                else:
                    outgoing[j].append(i)
                    incoming[i].append(j)

    sources = [i for i in range(n) if not incoming[i]]
    if not sources:
        sources = [int(np.argmax(proj))]

    # Topological order: always expand the highest unprocessed node first.
    in_degree = [len(incoming[i]) for i in range(n)]
    ready = sorted(sources, key=lambda i: -proj[i])
    order: List[int] = []
    while ready:
        u = ready.pop(0)
        order.append(u)
        for v in outgoing[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                ready.append(v)
        ready.sort(key=lambda i: -proj[i])

    # Local flow direction per body: prefer direction to children; if leaf, from parent.
    flow_axis: Dict[str, np.ndarray] = {}
    for i, b in enumerate(gating):
        if outgoing[i]:
            dir_vec = np.mean([centers[v] - centers[i] for v in outgoing[i]], axis=0)
        elif incoming[i]:
            dir_vec = centers[i] - centers[incoming[i][0]]
        else:
            dir_vec = g
        norm = float(np.linalg.norm(dir_vec)) + 1e-12
        flow_axis[b.name] = dir_vec / norm

    # Cross-sectional area of every body, measured perpendicular to its flow axis.
    areas_mm2: Dict[str, float] = {}
    for b in gating:
        base, throat = _body_cross_section_mm2(b, axis=flow_axis[b.name])
        areas_mm2[b.name] = float(base)
        if throat is not None:
            areas_mm2[b.name + ":throat"] = float(throat)

    body_list = gating
    return {
        "bodies": body_list,
        "parent": {
            b.name: (gating[incoming[i][0]].name if incoming[i] else None)
            for i, b in enumerate(body_list)
        },
        "children": {
            b.name: [gating[v].name for v in outgoing[i]]
            for i, b in enumerate(body_list)
        },
        "order": [gating[i].name for i in order],
        "sources": [gating[i].name for i in sources],
        "up": up,
        "flow_axis": flow_axis,
        "areas_mm2": areas_mm2,
        "centers": {b.name: centers[i] for i, b in enumerate(body_list)},
    }


def _real_gating_areas_from_bodies(
    bodies: List[Body],
    gravity_vector: Tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> Dict[str, float]:
    """Compute real sprue/runner/distributor/curufluk/ingate cross-section areas from CAD meshes.

    Series elements are not summed; the representative area for RUNNER and
    DISTRIBUTOR is taken from the terminal body that feeds the next section
    (gate / distributor / curufluk).  INGATE bodies are summed because they
    usually represent parallel gates.
    """
    topology = _build_gating_topology(bodies, gravity_vector=gravity_vector)
    if not topology["bodies"]:
        return {
            "runner_total_mm2": 0.0,
            "runner_total_cm2": 0.0,
            "distributor_total_mm2": 0.0,
            "distributor_total_cm2": 0.0,
            "curufluk_total_mm2": 0.0,
            "curufluk_total_cm2": 0.0,
            "ingate_total_mm2": 0.0,
            "ingate_total_cm2": 0.0,
            "sprue_base_mm2": 0.0,
            "sprue_base_cm2": 0.0,
            "sprue_throat_mm2": 0.0,
            "sprue_throat_cm2": 0.0,
            "n_ingates": 0,
        }

    body_map = {b.name: b for b in topology["bodies"]}
    areas = topology["areas_mm2"]
    children = topology["children"]
    order = topology["order"]

    def _child_of_types(name: str, types: Tuple[BodyType, ...]) -> bool:
        return any(body_map[c].body_type in types for c in children.get(name, []))

    def _select_terminal(btype: BodyType, feed_types: Tuple[BodyType, ...]) -> List[str]:
        candidates = [n for n in order if body_map[n].body_type == btype]
        terminals = [n for n in candidates if _child_of_types(n, feed_types)]
        if terminals:
            return terminals
        # Fallback: if all are in series, take the one closest to the gates.
        if candidates:
            return [candidates[-1]]
        return []

    # INGATE: always sum parallel gates.
    ingate_names = [n for n in order if body_map[n].body_type == BodyType.INGATE]
    ingate_total_mm2 = sum(areas[n] for n in ingate_names)
    n_ingates = len(ingate_names)

    # RUNNER: terminal runner feeding distributor/curufluk/ingate.
    runner_names = _select_terminal(
        BodyType.RUNNER,
        (BodyType.DISTRIBUTOR, BodyType.CURUFLUK, BodyType.INGATE),
    )
    runner_total_mm2 = sum(areas[n] for n in runner_names)

    # DISTRIBUTOR: terminal distributor feeding curufluk/ingate.
    distributor_names = _select_terminal(
        BodyType.DISTRIBUTOR,
        (BodyType.CURUFLUK, BodyType.INGATE),
    )
    distributor_total_mm2 = sum(areas[n] for n in distributor_names)

    # CURUFLUK: sum all curufluk bodies that are part of the flow path.
    curufluk_names = [n for n in order if body_map[n].body_type == BodyType.CURUFLUK]
    curufluk_total_mm2 = sum(areas[n] for n in curufluk_names)

    # SPRUE/POURING_BASIN: collect base and throat along each source branch.
    sprue_base_total_mm2 = 0.0
    sprue_throat_min_mm2 = 0.0
    has_sprue = any(body_map[n].body_type == BodyType.SPRUE for n in order)
    visited: set = set()

    def _collect_sprue(name: str) -> None:
        nonlocal sprue_base_total_mm2, sprue_throat_min_mm2
        if name in visited:
            return
        visited.add(name)
        b = body_map[name]
        if b.body_type == BodyType.SPRUE:
            base = areas[name]
            throat = areas.get(name + ":throat", base)
            sprue_base_total_mm2 += base
            if sprue_throat_min_mm2 == 0.0:
                sprue_throat_min_mm2 = throat
            else:
                sprue_throat_min_mm2 = min(sprue_throat_min_mm2, throat)
        elif b.body_type == BodyType.SPRUE_THROAT:
            throat = areas[name]
            if sprue_throat_min_mm2 == 0.0:
                sprue_throat_min_mm2 = throat
            else:
                sprue_throat_min_mm2 = min(sprue_throat_min_mm2, throat)
        elif b.body_type == BodyType.POURING_BASIN and not has_sprue:
            # Only use pouring-basin area as a sprue fallback when no sprue exists.
            base = areas[name]
            if sprue_base_total_mm2 == 0.0:
                sprue_base_total_mm2 = base
                sprue_throat_min_mm2 = base
        for c in children.get(name, []):
            if body_map[c].body_type in _GATING_BODY_TYPES:
                _collect_sprue(c)

    for src in topology["sources"]:
        _collect_sprue(src)

    return {
        "runner_total_mm2": runner_total_mm2,
        "runner_total_cm2": runner_total_mm2 / 100.0,
        "distributor_total_mm2": distributor_total_mm2,
        "distributor_total_cm2": distributor_total_mm2 / 100.0,
        "curufluk_total_mm2": curufluk_total_mm2,
        "curufluk_total_cm2": curufluk_total_mm2 / 100.0,
        "ingate_total_mm2": ingate_total_mm2,
        "ingate_total_cm2": ingate_total_mm2 / 100.0,
        "sprue_base_mm2": sprue_base_total_mm2,
        "sprue_base_cm2": sprue_base_total_mm2 / 100.0,
        "sprue_throat_mm2": sprue_throat_min_mm2,
        "sprue_throat_cm2": sprue_throat_min_mm2 / 100.0,
        "n_ingates": n_ingates,
    }


def _volumes_from_bodies(bodies: List[Body]) -> Tuple[float, float]:
    """Return (part_volume_cm3, total_metal_volume_cm3) from CAD body volumes."""
    part_volume_cm3 = 0.0
    total_metal_volume_cm3 = 0.0
    for body in bodies:
        if body.body_type == BodyType.PART:
            part_volume_cm3 += max(body.volume_cm3, 0.0)
        if body.body_type in BODY_METAL_TYPES:
            total_metal_volume_cm3 += max(body.volume_cm3, 0.0)
    return part_volume_cm3, total_metal_volume_cm3


def _gating_area_design(
    W_total_kg: float,
    rho_kg_m3: float,
    H_eff_m: float,
    t_fill_s: float,
    Cd: float,
    gating_ratio: Tuple[float, float, float] = (1.0, 2.0, 1.0),
    n_ingates: int = 1,
) -> Dict[str, float]:
    """Wrap compute_gating from gating_calculator_tr.py; return cm² / mm."""
    if H_eff_m <= 0.0 or t_fill_s <= 0.0 or rho_kg_m3 <= 0.0:
        return {
            "As_cm2": 0.0,
            "Ar_total_cm2": 0.0,
            "Ag_total_cm2": 0.0,
            "Ag_each_cm2": 0.0,
            "Vc_ms": 0.0,
            "d_sprue_mm": 0.0,
            "d_ingate_each_mm": 0.0,
            "ratio": gating_ratio,
        }

    res = compute_gating(
        W_kg=W_total_kg,
        rho_kgm3=rho_kg_m3,
        H_m=H_eff_m,
        t_fill_s=t_fill_s,
        Cd=Cd,
        gating_ratio=gating_ratio,
        n_ingates=max(n_ingates, 1),
    )
    conv = 1e4  # m² -> cm²
    return {
        "As_cm2": res["As_m2"] * conv,
        "Ar_total_cm2": res["Ar_total_m2"] * conv,
        "Ag_total_cm2": res["Ag_total_m2"] * conv,
        "Ag_each_cm2": res["Ag_each_m2"] * conv,
        "Vc_ms": float(res["Vc_ms"]),
        "d_sprue_mm": res["d_sprue_m"] * 1000.0,
        "d_ingate_each_mm": res["d_ingate_m"] * 1000.0,
        "ratio": gating_ratio,
    }


def _default_gating_ratio(alloy_key: str) -> Tuple[float, float, float]:
    """Default As:Ar:Ag design ratio from gating_calculator_tr.py material defaults."""
    key = alloy_key.lower()
    if "gri" in key or "sfero" in key or "ggg" in key:
        return (1.0, 0.75, 0.5)
    if "al" in key or "alum" in key:
        return (1.0, 2.0, 1.5)
    return (1.0, 2.0, 1.0)


def _target_gate_velocity_m_s(alloy_key: str, wall_category: str = "orta cidarlı") -> float:
    """Target gate velocity for auto-tuning the As:Ar:Ag ratio."""
    key = alloy_key.lower()
    base = 1.3
    if "gri" in key or "sfero" in key or "ggg" in key or "pik" in key:
        base = 0.8
    elif "al" in key or "alum" in key:
        base = 0.4
    if wall_category == "ince cidarlı":
        base *= 1.15
    elif wall_category == "kalın cidarlı":
        base *= 0.9
    return float(np.clip(base, 0.2, 3.0))


def _auto_tune_gating_ratio(
    H_eff_m: float,
    base_ratio: Tuple[float, float, float],
    target_v_gate_m_s: float,
    part_mass_kg: float = 0.0,
) -> Tuple[float, float, float]:
    """Return an As:Ar:Ag ratio that keeps Ag_ratio large enough so v_gate <= target.

    The sprue velocity is v_c = sqrt(2*g*H_eff). With As_ratio = 1 the per-gate
    velocity is v_gate = v_c / Ag_ratio. To hit a target gate velocity we need
    Ag_ratio = v_c / target, but we never make the gate smaller than the sprue
    (Ag_ratio < 1) because that would choke at the gate, not the sprue.
    """
    As_ratio, Ar_ratio, Ag_ratio = base_ratio
    if H_eff_m <= 0 or target_v_gate_m_s <= 0:
        return base_ratio
    v_c = math.sqrt(2.0 * 9.81 * H_eff_m)
    if v_c <= 0:
        return base_ratio
    # Do not increase gate velocity above v_c (never make Ag < As).
    effective_target = min(target_v_gate_m_s, v_c * 0.95)
    new_Ag = v_c / max(effective_target, 0.05)
    # Keep Ag >= As and clamp to reasonable values.
    new_Ag = max(new_Ag, As_ratio)
    new_Ag = float(np.clip(new_Ag, 0.5, 5.0))
    # For very small castings keep the base ratio to avoid extremes.
    if part_mass_kg > 0.0 and part_mass_kg < 0.5:
        return base_ratio
    return (As_ratio, Ar_ratio, new_Ag)


def auto_fill_time(mass_kg: float, alloy_key: str = "", alloy_name: str = "") -> float:
    """Practical fill-time estimate from gating_calculator_tr.py.

    Wraps core.gating_calculator.auto_fill_time and clamps the result so very
    small / very large masses do not drive design into unrealistic regions.
    """
    if mass_kg <= 0.0:
        return 3.0
    name = (alloy_name or alloy_key or "Çelik")
    t = _gc_auto_fill_time(name, mass_kg)
    return float(np.clip(t, 0.2, 120.0))


def compute_modulus_and_riser(
    W_part_kg: float,
    rho_kg_m3: float,
    A_cast_m2: float,
    k_mod: float = 1.2,
) -> Dict[str, float]:
    """Wrap compute_modulus_and_riser from gating_calculator_tr.py."""
    if A_cast_m2 <= 0.0 or W_part_kg <= 0.0 or rho_kg_m3 <= 0.0:
        return {
            "V_cast_m3": 0.0,
            "M_cast_m": 0.0,
            "M_riser_req_m": 0.0,
            "riser_D_m": 0.0,
            "riser_H_m": 0.0,
            "riser_M_m": 0.0,
        }
    return _gc_compute_modulus_and_riser(
        W_part_kg=W_part_kg,
        rho_kgm3=rho_kg_m3,
        A_cast_m2=A_cast_m2,
        k_mod=k_mod,
    )


def ingate_contact_area_and_mask(grid: np.ndarray, dx: float) -> tuple:
    """Return (total ingate-part contact face area in mm2, source voxels touching part)."""
    source = _gate_source_mask(grid)
    part = grid == BodyType.PART
    contact_source = np.zeros_like(source)
    face_count = 0
    for di, dj, dk in _neighbor_offsets_6():
        rolled = np.roll(part, (di, dj, dk), axis=(0, 1, 2))
        _apply_edge_mask(rolled, di, dj, dk)
        faces = source & rolled
        contact_source |= faces
        face_count += int(faces.sum())
    return face_count * dx * dx, contact_source


def _part_touching_ingate_mask(grid: np.ndarray) -> np.ndarray:
    """Return part voxels that have at least one gate-source (ingate/runner/sprue) neighbor."""
    source = _gate_source_mask(grid)
    part = grid == BodyType.PART
    touch = np.zeros_like(part)
    for di, dj, dk in _neighbor_offsets_6():
        rolled = np.roll(source, (di, dj, dk), axis=(0, 1, 2))
        _apply_edge_mask(rolled, di, dj, dk)
        touch |= rolled & part
    return touch


def _minimum_cross_section_area(mask: np.ndarray, dx: float) -> float:
    """Approximate minimum cross-sectional area of a voxel set using PCA slicing."""
    pts = np.argwhere(mask)
    if len(pts) < 3:
        return 0.0
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    principal = principal / (np.linalg.norm(principal) + 1e-12)
    proj = centered @ principal
    slices = np.round(proj).astype(int)
    counts = []
    for s in np.unique(slices):
        counts.append((s, np.sum(slices == s)))
    if not counts:
        return 0.0
    max_count = max(c for _, c in counts)
    min_area = float("inf")
    for s, count in counts:
        # Ignore end slices that contain only a few voxels; they are not a real cross-section.
        # Use a small relative threshold (2% or at least 2 voxels) so thin throats are kept.
        if count < max(2, max_count * 0.02):
            continue
        area = count * dx * dx
        if area < min_area:
            min_area = area
    if min_area == float("inf"):
        # Fallback: use the largest slice if all were tiny.
        s, c = max(counts, key=lambda x: x[1])
        min_area = c * dx * dx
    return min_area


def _sprue_base_area(sprue_mask: np.ndarray, dx: float) -> float:
    """Area of the lowest-Z slice of the sprue."""
    pts = np.argwhere(sprue_mask)
    if len(pts) == 0:
        return 0.0
    min_z = pts[:, 2].min()
    base_slice = sprue_mask[:, :, int(min_z)]
    count = int(base_slice.sum())
    return count * dx * dx


def _mean_thickness(mask: np.ndarray, dx: float) -> float:
    """Mean wall thickness of a voxel set (2 * internal distance transform)."""
    if not mask.any():
        return 0.0
    edt = ndimage.distance_transform_edt(mask) * dx
    return float(edt[mask].mean()) * 2.0


def _distance_to_sprue_26(channel_mask: np.ndarray, sprue_mask: np.ndarray, dx: float) -> np.ndarray:
    """26-neighbor Dijkstra distance from every channel voxel to the sprue."""
    dist = np.full(channel_mask.shape, np.inf, dtype=np.float64)
    if not (channel_mask & sprue_mask).any():
        return dist

    idx = np.full(channel_mask.shape, -1, dtype=np.int64)
    vox = np.argwhere(channel_mask)
    n = int(vox.shape[0])
    idx[tuple(vox.T)] = np.arange(n)

    rows, cols, vals = [], [], []
    for (di, dj, dk), c in zip(NEIGH_26, COST_26):
        ni = vox[:, 0] + di
        nj = vox[:, 1] + dj
        nk = vox[:, 2] + dk
        mask = (
            (ni >= 0)
            & (ni < channel_mask.shape[0])
            & (nj >= 0)
            & (nj < channel_mask.shape[1])
            & (nk >= 0)
            & (nk < channel_mask.shape[2])
        )
        if not mask.any():
            continue
        neighbor_idx = idx[ni[mask], nj[mask], nk[mask]]
        source_idx = np.arange(n)[mask]
        valid = neighbor_idx >= 0
        if not valid.any():
            continue
        rows.append(source_idx[valid])
        cols.append(neighbor_idx[valid])
        vals.append(np.full(valid.sum(), c * dx, dtype=np.float32))

    sprue_flat = np.where(sprue_mask[tuple(vox.T)])[0]
    rows.append(np.full(len(sprue_flat), n, dtype=np.int64))
    cols.append(sprue_flat.astype(np.int64))
    vals.append(np.zeros(len(sprue_flat), dtype=np.float32))

    graph = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n + 1, n + 1),
    ).tocsr()
    flat_dist = csgraph.dijkstra(graph, directed=False, indices=n, return_predecessors=False)
    dist[tuple(vox.T)] = flat_dist[:n].astype(np.float64)
    return dist


def _count_elbows_along_path(
    dist: np.ndarray,
    channel_mask: np.ndarray,
    start: Tuple[int, int, int],
    angle_threshold_deg: float = 60.0,
) -> int:
    """Trace from start toward decreasing dist and count sharp direction changes."""
    shape = dist.shape
    current = start
    if not channel_mask[current]:
        return 0
    path = [current]
    visited = {current}
    for _ in range(1000):
        i, j, k = current
        if dist[i, j, k] <= 0:
            break
        best = None
        best_d = dist[i, j, k]
        for di, dj, dk in _neighbor_offsets_6():
            ni, nj, nk = i + di, j + dj, k + dk
            if not (0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]):
                continue
            if not channel_mask[ni, nj, nk]:
                continue
            d = dist[ni, nj, nk]
            if d < best_d:
                best_d = d
                best = (ni, nj, nk)
        if best is None or best in visited:
            break
        visited.add(best)
        path.append(best)
        current = best

    if len(path) < 3:
        return 0
    elbows = 0
    cos_thresh = np.cos(np.deg2rad(angle_threshold_deg))
    for a in range(1, len(path) - 1):
        v1 = np.array(path[a]) - np.array(path[a - 1])
        v2 = np.array(path[a + 1]) - np.array(path[a])
        n1 = v1 / (np.linalg.norm(v1) + 1e-12)
        n2 = v2 / (np.linalg.norm(v2) + 1e-12)
        if np.dot(n1, n2) < cos_thresh:
            elbows += 1
    return elbows


# Campbell-style velocity ranges for pressurized / unpressurized / semi-pressurized
# gating systems (m/s).  Ref: Campbell casting practice / foundry design handbooks.
_GATING_VELOCITY_TARGETS = {
    "basınçlı (pressurized)": {
        "sprue": (1.0, 1.2),
        "runner": (1.2, 1.5),
        "gate": (1.8, 2.5),
    },
    "basınçsız (unpressurized)": {
        "sprue": (1.5, 2.0),
        "runner": (0.8, 1.2),
        "gate": (0.4, 0.7),
    },
    "yarı basınçlı (semi-pressurized)": {
        "sprue": (1.2, 1.5),
        "runner": (0.6, 1.0),
        "gate": (0.9, 1.2),
    },
}


def _target_area_range_cm2(Q_m3_s: float, v_lo: float, v_hi: float) -> Tuple[float, float]:
    """Return (A_min, A_max) in cm² so that v = Q/A stays inside [v_lo, v_hi]."""
    if Q_m3_s <= 0 or v_lo <= 0 or v_hi <= 0:
        return 0.0, 0.0
    # A = Q / v ; larger v needs smaller A
    a_min_m2 = Q_m3_s / v_hi
    a_max_m2 = Q_m3_s / v_lo
    return a_min_m2 * 1e4, a_max_m2 * 1e4


def _normalized_distance_to_range(v: float, lo: float, hi: float) -> float:
    if lo <= v <= hi:
        return 0.0
    width = max(hi - lo, 0.1)
    if v < lo:
        return (lo - v) / width
    return (v - hi) / width


def _classify_gating_system(v_sprue: float, v_runner: float, v_gate: float) -> str:
    """Classify by velocity/area ordering first, then by absolute range proximity.

    Pressurized: As > Ar > Ag  => v_sprue <= v_runner <= v_gate
    Unpressurized: As < Ar < Ag => v_sprue >= v_runner >= v_gate
    Semi-pressurized: Ar is largest => v_runner is lowest.
    """
    avg = max((v_sprue + v_runner + v_gate) / 3.0, 0.01)

    # Normalized ordering penalties (primary signal)
    def press_penalty() -> float:
        return (max(0.0, v_sprue - v_runner) + max(0.0, v_runner - v_gate)) / avg

    def unpress_penalty() -> float:
        return (max(0.0, v_runner - v_sprue) + max(0.0, v_gate - v_runner)) / avg

    def semi_penalty() -> float:
        return (
            max(0.0, v_runner - v_sprue)
            + max(0.0, v_runner - v_gate)
            + 0.5 * abs(v_sprue - v_gate) / avg
        ) / avg

    # Small range-distance tie-breaker so unrealistic fill times do not override ordering.
    range_score = 0.0
    for v, lo, hi in [
        (v_sprue, *(_GATING_VELOCITY_TARGETS["basınçlı (pressurized)"]["sprue"])),
        (v_runner, *(_GATING_VELOCITY_TARGETS["basınçlı (pressurized)"]["runner"])),
        (v_gate, *(_GATING_VELOCITY_TARGETS["basınçlı (pressurized)"]["gate"])),
    ]:
        width = max(hi - lo, 0.1)
        if v < lo:
            range_score += (lo - v) / width
        elif v > hi:
            range_score += (v - hi) / width

    candidates = {
        "basınçlı (pressurized)": press_penalty() + 0.05 * range_score,
        "basınçsız (unpressurized)": unpress_penalty() + 0.05 * range_score,
        "yarı basınçlı (semi-pressurized)": semi_penalty() + 0.05 * range_score,
    }
    return min(candidates, key=candidates.get)


def _wall_thickness_category(wall_thickness_mm: float) -> str:
    if wall_thickness_mm < 6.0:
        return "ince cidarlı"
    if wall_thickness_mm <= 15.0:
        return "orta cidarlı"
    return "kalın cidarlı"


def _recommend_gating_system(category: str) -> Tuple[str, str]:
    """Return (recommended_system, reason) based on wall thickness."""
    if category == "ince cidarlı":
        return (
            "basınçlı (pressurized)",
            "İnce cidarlı parçada hızlı ve türbülanslı olmayan doldurma için yüksek gate hızı gerekir; "
            "basınçlı sistemde gate hızı 1.8–2.5 m/s hedeflenir.",
        )
    if category == "kalın cidarlı":
        return (
            "basınçsız (unpressurized)",
            "Kalın cidarlı parçada doldurma süresi daha uzun olabilir; türbülansı önlemek için "
            "gate hızı 0.4–0.7 m/s olan basınçsız sistem tercih edilir.",
        )
    return (
        "yarı basınçlı (semi-pressurized)",
        "Orta cidarlı parçalar için sprue/runner/gate hızları dengeli olan yarı basınçlı sistem uygundur.",
    )


def _compute_section_flow(
    section_key: str,
    area_cm2: float,
    thickness_mm: float,
    Q_m3_s: float,
    rho: float,
    mu: float,
    g: float,
    target_v_min_m_s: float,
    target_v_max_m_s: float,
    target_area_min_cm2: float = 0.0,
    target_area_max_cm2: float = 0.0,
) -> SectionFlow:
    """Velocity, Reynolds, Froude and turbulence flag for one gating section."""
    area_m2 = area_cm2 / 1e4
    if area_m2 > 0 and Q_m3_s > 0:
        velocity = Q_m3_s / area_m2
    else:
        velocity = 0.0
    D = max(thickness_mm / 1000.0, 1e-6)
    reynolds = 0.0
    froude = 0.0
    turbulent = False
    if velocity > 0:
        reynolds = rho * velocity * D / mu
        froude = velocity / np.sqrt(g * D)
        # Use the target maximum velocity as the turbulence trigger for all sections.
        v_limit = target_v_max_m_s if target_v_max_m_s > 0.0 else 999.0
        turbulent = (reynolds > 20000.0) or (velocity > v_limit)
    return SectionFlow(
        velocity_m_s=velocity,
        area_cm2=area_cm2,
        thickness_mm=thickness_mm,
        reynolds=reynolds,
        froude=froude,
        turbulent=turbulent,
        max_velocity_m_s=target_v_max_m_s,
        target_v_min_m_s=target_v_min_m_s,
        target_v_max_m_s=target_v_max_m_s,
        target_area_min_cm2=target_area_min_cm2,
        target_area_max_cm2=target_area_max_cm2,
    )


def analyze_gating(
    result: AnalysisResult,
    fill_time_s: Optional[float] = None,
    discharge_coeff: float = 0.8,
    casting_params=None,
    bodies: Optional[List[Body]] = None,
    user_section_areas_cm2: Optional[Dict[str, float]] = None,
) -> Optional[GateResult]:
    """Compute gate/sprue/runner design from gating_calculator_tr.py / Filling_time_tr.py.

    CAD mesh areas are used only as a secondary comparison; the primary
    velocities, areas and recommendations come from part mass, effective
    metal head and the As:Ar:Ag ratio.
    """
    from core.types import CastingParameters

    grid = result.grid
    sdf = result.sdf
    dx = result.dx_mm
    alloy = get_alloy(result.alloy_key)
    mold = get_mold(result.mold_key)

    use_bodies = bodies is not None and len(bodies) > 0
    gravity_vector = (0.0, 0.0, -1.0)
    if casting_params is not None and isinstance(casting_params, CastingParameters):
        gravity_vector = getattr(casting_params, "gravity_vector", gravity_vector) or gravity_vector
    real_areas = (
        _real_gating_areas_from_bodies(bodies, gravity_vector=gravity_vector)
        if use_bodies
        else {}
    )

    # User-supplied cross-section areas from the 3D viewer override automatic
    # mesh measurements so the engineer can correct ambiguous geometries.
    if user_section_areas_cm2:
        key_map = {
            "SPRUE_BASE": "sprue_base_cm2",
            "SPRUE_THROAT": "sprue_throat_cm2",
            "RUNNER": "runner_total_cm2",
            "DISTRIBUTOR": "distributor_total_cm2",
            "CURUFLUK": "curufluk_total_cm2",
            "INGATE": "ingate_total_cm2",
        }
        for ui_key, real_key in key_map.items():
            val = user_section_areas_cm2.get(ui_key)
            if val is not None and val > 0.0:
                real_areas[real_key] = float(val)
                # Store the raw mm² value as well so downstream code is consistent.
                real_areas[real_key.replace("_cm2", "_mm2")] = float(val) * 100.0

    if casting_params is not None and isinstance(casting_params, CastingParameters):
        fill_time_s = casting_params.t_fill_s
        alloy = replace(
            alloy,
            t_pour_c=casting_params.t_pour_c,
            t_liquidus_c=casting_params.t_liquidus_c,
            t_solidus_c=casting_params.t_solidus_c,
            rho_kg_m3=casting_params.rho_liquid_kg_m3,
            viscosity_pa_s=casting_params.viscosity_pa_s,
        )
    part_mask = grid == BodyType.PART
    is_metal = result.is_metal
    ingate = grid == BodyType.INGATE
    runner = grid == BodyType.RUNNER
    distributor = grid == BodyType.DISTRIBUTOR
    curufluk = grid == BodyType.CURUFLUK
    sprue = (grid == BodyType.SPRUE) | (grid == BodyType.SPRUE_THROAT)
    source = _gate_source_mask(grid)
    has_ingate = ingate.any()
    has_distributor = distributor.any()
    has_curufluk = curufluk.any()

    # CAD geometry areas (support / comparison only)
    if use_bodies:
        gate_area_cm2 = real_areas.get("ingate_total_cm2", 0.0)
        runner_min_area_cm2 = real_areas.get("runner_total_cm2", 0.0)
        distributor_area_cm2 = real_areas.get("distributor_total_cm2", 0.0)
        curufluk_area_cm2 = real_areas.get("curufluk_total_cm2", 0.0)
        sprue_base_cm2 = real_areas.get("sprue_base_cm2", 0.0)
        sprue_throat_cm2 = real_areas.get("sprue_throat_cm2", 0.0)

        contact_area_mm2, _ = ingate_contact_area_and_mask(grid, dx)
        gate_contact_area_cm2 = gate_area_cm2 if gate_area_cm2 > 0.0 else contact_area_mm2 / 100.0
        if gate_area_cm2 <= 0.0:
            gate_area_cm2 = gate_contact_area_cm2

        part_volume_cm3, total_metal_volume_cm3 = _volumes_from_bodies(bodies)
    else:
        gate_contact_area_mm2, _ = ingate_contact_area_and_mask(grid, dx)
        gate_contact_area_cm2 = gate_contact_area_mm2 / 100.0

        ingate_min_area_mm2 = _minimum_cross_section_area(ingate, dx) if has_ingate else 0.0
        ingate_min_area_cm2 = ingate_min_area_mm2 / 100.0
        if has_ingate:
            if ingate_min_area_cm2 <= 0:
                ingate_min_area_cm2 = gate_contact_area_cm2
            else:
                ingate_min_area_cm2 = min(ingate_min_area_cm2, gate_contact_area_cm2)
        gate_area_cm2 = ingate_min_area_cm2 if has_ingate else gate_contact_area_cm2

        runner_min_area_mm2 = _minimum_cross_section_area(runner, dx)
        runner_min_area_cm2 = runner_min_area_mm2 / 100.0

        distributor_area_cm2 = 0.0
        curufluk_area_cm2 = 0.0

        sprue_throat_mm2 = _minimum_cross_section_area(sprue, dx) if sprue.any() else 0.0
        sprue_throat_cm2 = sprue_throat_mm2 / 100.0
        sprue_base_bottom_mm2 = _sprue_base_area(sprue, dx) if sprue.any() else 0.0
        sprue_base_bottom_cm2 = sprue_base_bottom_mm2 / 100.0
        sprue_base_cm2 = sprue_base_bottom_cm2

        part_volume_mm3 = float(part_mask.sum()) * (dx ** 3)
        part_volume_cm3 = part_volume_mm3 / 1000.0
        total_metal_volume_mm3 = float(is_metal.sum()) * (dx ** 3)
        total_metal_volume_cm3 = total_metal_volume_mm3 / 1000.0

    runner_thickness_mm = _mean_thickness(runner, dx)
    distributor_thickness_mm = _mean_thickness(distributor, dx)
    curufluk_thickness_mm = _mean_thickness(curufluk, dx)
    sprue_thickness_mm = _mean_thickness(sprue, dx)
    ingate_thickness_mm = _mean_thickness(ingate if has_ingate else source, dx)

    part_weight_g = part_volume_cm3 * alloy.density_g_cm3
    part_mass_kg = part_weight_g / 1000.0
    total_weight_g = total_metal_volume_cm3 * alloy.density_g_cm3
    total_metal_volume_m3 = total_metal_volume_cm3 / 1e6
    total_mass_kg = total_weight_g / 1000.0
    pour_yield = part_volume_cm3 / total_metal_volume_cm3 if total_metal_volume_cm3 > 0 else 1.0

    # Fill time (Filling_time_tr.py + gating_calculator_tr.py auto_fill_time)
    superheat = max(alloy.t_pour_c - alloy.t_liquidus_c, 0.0)
    wall_thickness_mm = getattr(result, "wall_thickness_mm", 0.0) or 20.0
    wall_cat = _wall_thickness_category(wall_thickness_mm)

    campbell_res = calc_campbell_parameters(part_mass_kg, alloy.rho_kg_m3, wall_thickness_mm, superheat)
    campbell_fill_time_s = campbell_res["t_fill"]
    campbell_fill_time_basis = campbell_res["t_base_detail"]
    auto_fill_time_s = auto_fill_time(part_mass_kg, alloy.key, alloy.name)
    user_fill_time_s = fill_time_s if (fill_time_s and fill_time_s > 0) else None
    recommended_fill_time_s = auto_fill_time_s
    fill_time_basis = "auto_fill_time"
    design_fill_time_s = user_fill_time_s

    # Number of ingate bodies
    if use_bodies:
        n_ingates = max(int(real_areas.get("n_ingates", 1)), 1)
    elif has_ingate:
        _, n_ingates = ndimage.label(ingate)
    else:
        n_ingates = 1

    # Effective metal head from geometry + mass reduction + elbow losses.
    # Use average ferrostatic head (h_max - c/2) to account for backpressure as
    # the mold fills; c is the part height in the casting direction.
    metal_pts = np.argwhere(result.is_metal)
    if len(metal_pts) > 0:
        total_height_mm = float((metal_pts[:, 2].max() - metal_pts[:, 2].min()) * dx)
    else:
        total_height_mm = 0.0
    part_mask = result.grid == BodyType.PART
    part_pts = np.argwhere(part_mask)
    if len(part_pts) > 0:
        part_height_mm = float((part_pts[:, 2].max() - part_pts[:, 2].min()) * dx)
    else:
        part_height_mm = total_height_mm
    h_avg_mm = max(total_height_mm - 0.5 * part_height_mm, total_height_mm * 0.1)
    height_m = total_height_mm / 1000.0
    H_eff_m = effective_head(h_avg_mm / 1000.0, part_mass_kg)
    H_eff_m = float(np.clip(H_eff_m, 0.02, 0.60))

    channel_mask = np.isin(
        grid,
        [BodyType.INGATE, BodyType.RUNNER, BodyType.DISTRIBUTOR, BodyType.CURUFLUK, BodyType.SPRUE, BodyType.SPRUE_THROAT, BodyType.FILTER, BodyType.POURING_BASIN],
    )
    sprue_mask = sprue & channel_mask
    elbow_count = 0
    head_loss_m = 0.0
    if channel_mask.any() and sprue_mask.any():
        source_vox = np.argwhere(ingate) if has_ingate else np.argwhere(runner & channel_mask)
        if len(source_vox) > 0:
            dist_to_sprue = _distance_to_sprue_26(channel_mask, sprue_mask, dx)
            sample = source_vox[np.linspace(0, len(source_vox) - 1, min(20, len(source_vox))).astype(int)]
            counts = []
            for v in sample:
                counts.append(_count_elbows_along_path(dist_to_sprue, channel_mask, tuple(v)))
            elbow_count = int(round(np.median(counts))) if counts else 0
            v_loss_m_s = math.sqrt(2.0 * 9.81 * H_eff_m)
            h_loss_per_elbow_m = alloy.elbow_loss_k * (v_loss_m_s ** 2) / (2.0 * 9.81)
            head_loss_m = h_loss_per_elbow_m * elbow_count
    # Engine will subtract head_loss from its own effective-head calculation.
    # We keep the raw H_eff_m for the local loss estimate and update H_eff_m
    # from the engine result afterwards.
    head_reduction_percent = 100.0 * (1.0 - (max(H_eff_m - head_loss_m, 0.02) / max(height_m, 1e-9)))

    # Geometry-aware gating design engine.
    # It uses Q = A·v with material/system specific velocity targets, while
    # preserving cross-sectional areas the user explicitly picked in the 3D
    # viewer.  Auto-computed throat areas are not forced on the engine so it can
    # keep throat = base unless the user measured it.
    engine_measured_cm2: Dict[str, float] = {}
    ui_to_real = {
        "INGATE": "ingate_total_cm2",
        "RUNNER": "runner_total_cm2",
        "SPRUE_BASE": "sprue_base_cm2",
        "SPRUE_THROAT": "sprue_throat_cm2",
    }
    if user_section_areas_cm2:
        for ui_key, real_key in ui_to_real.items():
            val = user_section_areas_cm2.get(ui_key)
            if val is not None and val > 0.0:
                engine_measured_cm2[ui_key] = float(val)
    # Fall back to CAD/auto-measured values for comparison/warning only.
    for ui_key, real_key in ui_to_real.items():
        if ui_key in engine_measured_cm2:
            continue
        val = real_areas.get(real_key, 0.0)
        if val > 0.0:
            engine_measured_cm2[ui_key] = float(val)

    user_gate_velocity = 0.0
    user_velocity_section = "INGATE"
    if casting_params is not None and isinstance(casting_params, CastingParameters):
        user_gate_velocity = float(getattr(casting_params, "ingate_velocity_m_s", 0.0) or 0.0)
        user_velocity_section = str(getattr(casting_params, "velocity_section_key", "INGATE") or "INGATE")

    # v9.2: extended geometry features for the gating engine.
    if result.subvoxel_sdf.size and part_mask.any():
        part_sdf_vals = result.subvoxel_sdf[part_mask]
        t_min_mm = 2.0 * float(np.percentile(part_sdf_vals, 5))
        t_max_mm = 2.0 * float(np.percentile(part_sdf_vals, 95))
    else:
        t_min_mm = wall_thickness_mm * 0.5
        t_max_mm = wall_thickness_mm * 1.2
    surface_to_volume_ratio_1_mm = (
        result.part_surface_area_mm2 / result.part_volume_mm3
        if result.part_volume_mm3 > 0.0
        else 0.0
    )
    hotspot_count = len(result.hotspots)
    max_hotspot_m_mm = (
        max([hs.m_value_mm for hs in result.hotspots], default=0.0)
        if result.hotspots
        else 0.0
    )
    pore_risk_max = (
        float(result.risk[part_mask].max())
        if result.risk.size and part_mask.any()
        else 0.0
    )

    engine_input = GatingEngineInput(
        total_metal_volume_m3=total_metal_volume_m3,
        total_mass_kg=total_mass_kg,
        part_volume_m3=part_volume_cm3 / 1e6,
        part_mass_kg=part_mass_kg,
        part_height_mm=part_height_mm,
        total_height_mm=total_height_mm,
        max_flow_path_mm=float(result.bbox_size_mm.max()),
        wall_thickness_mm=wall_thickness_mm,
        wall_thickness_min_mm=t_min_mm,
        wall_thickness_max_mm=t_max_mm,
        surface_to_volume_ratio_1_mm=surface_to_volume_ratio_1_mm,
        hotspot_count=hotspot_count,
        max_hotspot_m_mm=max_hotspot_m_mm,
        pore_risk_max=pore_risk_max,
        alloy_key=alloy.key,
        alloy_name=alloy.name,
        rho_kg_m3=alloy.rho_kg_m3,
        viscosity_pa_s=alloy.viscosity_pa_s,
        latent_heat_j_kg=alloy.latent_heat_j_kg,
        cp_j_kgk=alloy.cp_j_kgk,
        t_pour_c=alloy.t_pour_c,
        t_liquidus_c=alloy.t_liquidus_c,
        t_fill_s=user_fill_time_s,
        user_gate_velocity_m_s=user_gate_velocity if user_gate_velocity > 0 else None,
        user_velocity_section_key=user_velocity_section,
        discharge_coeff=discharge_coeff,
        measured_areas_cm2=engine_measured_cm2,
        n_gates=n_ingates if n_ingates > 1 else None,
        head_loss_m=head_loss_m,
        max_gates=8,
    )
    design = calculate_gating_design(engine_input)
    H_eff_m = design.h_eff_mm / 1000.0  # engine's H_eff already includes losses

    # Pull results back into the names the rest of analyze_gating expects.
    As_m2 = design.sprue_base_area_cm2 / 1e4
    Ar_total_m2 = design.runner_total_area_cm2 / 1e4
    Ag_total_m2 = design.gate_total_area_cm2 / 1e4
    Ag_each_m2 = design.gate_each_area_cm2 / 1e4
    Vc_ms = design.v_choke_m_s
    Q_design_m3_s = design.q_m3_s
    fill_time_s = design.t_fill_s
    design_fill_time_s = fill_time_s
    ingate_Q_each = Q_design_m3_s / max(design.n_gates, 1)

    v_sprue_design = design.sprue_velocity_m_s
    v_runner_design = design.runner_velocity_m_s
    v_gate_design = design.gate_velocity_m_s

    d_sprue_mm = 1000.0 * math.sqrt(4.0 * max(As_m2, 0.0) / math.pi)
    d_ingate_each_mm = 1000.0 * math.sqrt(4.0 * max(Ag_each_m2, 0.0) / math.pi)

    # Keep a ratio for reporting; engine uses velocities, not a fixed ratio.
    if As_m2 > 0.0:
        final_ratio = (1.0, Ar_total_m2 / As_m2, Ag_total_m2 / As_m2)
    else:
        final_ratio = (1.0, 2.0, 1.0)

    recommended_system = design.recommended_gating_system
    detected_system = design.gating_system
    gating_system_reason = (
        f"Tasarım gating sistemi: {detected_system} (önerilen: {recommended_system}). Parça: {wall_cat}. "
        f"Hızlar (tasarım): sprue={v_sprue_design:.2f}, runner={v_runner_design:.2f}, gate={v_gate_design:.2f} m/s. "
        f"Oran As:Ar:Ag ≈ {final_ratio[0]:.2f}:{final_ratio[1]:.2f}:{final_ratio[2]:.2f}."
    )
    if design.warnings:
        gating_system_reason += " Uyarılar: " + "; ".join(design.warnings)

    # Target ranges from the engine for SectionFlow / UI limits.
    # Clamp the upper bound by material-specific safe velocity so steel/Al do not
    # inherit gray-iron target ranges.
    raw_targets = _ENGINE_VELOCITY_RANGES.get(
        detected_system,
        _ENGINE_VELOCITY_RANGES["yarı basınçlı (semi-pressurized)"],
    )
    velocity_targets = {}
    for section in ("sprue", "runner", "gate"):
        lo, hi = raw_targets[section]
        hi = min(hi, _section_velocity_limit(detected_system, alloy.key, section))
        # Keep a valid min/max interval; if the raw lower bound exceeds the
        # material-clamped upper bound, lower the lower bound proportionally.
        lo = min(lo, hi * 0.8)
        velocity_targets[section] = (lo, hi)
    sprue_v_range = velocity_targets["sprue"]
    runner_v_range = velocity_targets["runner"]
    gate_v_range = velocity_targets["gate"]
    sprue_A_min, sprue_A_max = _target_area_range_cm2(Q_design_m3_s, *sprue_v_range)
    runner_A_min, runner_A_max = _target_area_range_cm2(Q_design_m3_s, *runner_v_range)
    gate_A_min, gate_A_max = _target_area_range_cm2(ingate_Q_each, *gate_v_range)

    # Primary SectionFlow objects from the engine design.
    # _compute_section_flow already computes v = Q/A, so we do not override it;
    # this keeps SPRUE_THROAT velocity correct when its area differs from base.
    d_runner_mm = 1000.0 * math.sqrt(4.0 * max(Ar_total_m2, 0.0) / math.pi)
    section_flows: Dict[str, SectionFlow] = {}
    section_specs = [
        ("SPRUE_BASE", design.sprue_base_area_cm2, d_sprue_mm, sprue_v_range[0], sprue_v_range[1], sprue_A_min, sprue_A_max, Q_design_m3_s),
        ("SPRUE_THROAT", design.sprue_throat_area_cm2, d_sprue_mm, sprue_v_range[0], sprue_v_range[1], sprue_A_min, sprue_A_max, Q_design_m3_s),
        ("RUNNER", design.runner_total_area_cm2, d_runner_mm, runner_v_range[0], runner_v_range[1], runner_A_min, runner_A_max, Q_design_m3_s),
        ("INGATE", design.gate_each_area_cm2, d_ingate_each_mm, gate_v_range[0], gate_v_range[1], gate_A_min, gate_A_max, ingate_Q_each),
    ]
    mu = max(alloy.viscosity_pa_s, 1e-6)
    for key, area_cm2, thickness_mm, v_min, v_max, a_min, a_max, q_for_section in section_specs:
        sf = _compute_section_flow(
            key, area_cm2, thickness_mm, q_for_section,
            alloy.rho_kg_m3, mu, 9.81, v_min, v_max, a_min, a_max
        )
        section_flows[key] = sf

    ingate_flow = section_flows["INGATE"]
    runner_flow = section_flows["RUNNER"]
    sprue_flow = section_flows["SPRUE_BASE"]

    # The engine already re-classified the system from velocities; use it.
    n_ingates = design.n_gates

    # Ingat quality
    part_sdf = sdf[part_mask]
    max_part_sdf = float(part_sdf.max()) if len(part_sdf) > 0 else 0.0
    part_touch = _part_touching_ingate_mask(grid)
    contact_sdf = sdf[part_touch]
    if len(contact_sdf) > 0:
        ingate_avg_m = float(np.mean(contact_sdf))
        ingate_max_m = float(np.max(contact_sdf))
    else:
        ingate_avg_m = 0.0
        ingate_max_m = 0.0
    threshold = 0.8 * max_part_sdf
    ingate_on_thick = ingate_avg_m > threshold if max_part_sdf > 0 else False

    # Actual CAD velocities (comparison only)
    actual_area = {
        "sprue": sprue_base_cm2 if sprue_base_cm2 > 0 else As_m2 * 1e4,
        "runner": runner_min_area_cm2 if runner_min_area_cm2 > 0 else Ar_total_m2 * 1e4,
        "distributor": distributor_area_cm2,
        "curufluk": curufluk_area_cm2,
        "gate": gate_area_cm2 if gate_area_cm2 > 0 else Ag_total_m2 * 1e4,
    }
    actual_v = {}
    for k, a in actual_area.items():
        a_m2 = a / 1e4
        actual_v[k] = Q_design_m3_s / a_m2 if a_m2 > 0 else 0.0

    # P2: when a 3-D Darcy flow result exists, use it as the single source of
    # truth for measured section velocities.  Merge flow velocities on top of
    # the design values so missing sections still have a fallback and KeyError
    # is avoided later in report strings.
    flow_result = getattr(result, "flow_result", None)
    if flow_result is not None and getattr(flow_result, "Q_m3_s", 0.0) > 0.0:
        actual_v.update(_actual_velocities_from_flow(flow_result))

    # Classify the real system from the measured velocities.
    detected_system = _classify_from_velocities(
        actual_v.get("sprue", 0.0),
        actual_v.get("runner", 0.0),
        actual_v.get("gate", 0.0),
        v_distributor=actual_v.get("distributor", 0.0),
        v_curufluk=actual_v.get("curufluk", 0.0),
    )

    # Recompute target ranges based on the measured system so warnings match
    # the physical behaviour, not just the design assumption.
    raw_targets = _ENGINE_VELOCITY_RANGES.get(
        detected_system,
        _ENGINE_VELOCITY_RANGES["yarı basınçlı (semi-pressurized)"],
    )
    velocity_targets = {}
    for section in ("sprue", "runner", "gate"):
        lo, hi = raw_targets[section]
        hi = min(hi, _section_velocity_limit(detected_system, alloy.key, section))
        lo = min(lo, hi * 0.8)
        velocity_targets[section] = (lo, hi)
    sprue_v_range = velocity_targets["sprue"]
    runner_v_range = velocity_targets["runner"]
    gate_v_range = velocity_targets["gate"]

    # P2: overwrite the design SectionFlow objects with the Darcy flow result
    # once the target ranges are known.
    if flow_result is not None and getattr(flow_result, "Q_m3_s", 0.0) > 0.0:
        section_flows = _section_flows_from_flow(
            flow_result, alloy.rho_kg_m3, mu, 9.81, velocity_targets
        )
        ingate_flow = section_flows.get("INGATE", section_flows.get("gate", ingate_flow))
        runner_flow = section_flows.get("RUNNER", section_flows.get("runner", runner_flow))
        sprue_flow = section_flows.get("SPRUE_BASE", section_flows.get("sprue", sprue_flow))

    # Velocity penalty: a section must be failed if its real velocity exceeds the
    # recommended maximum, even if the area ratio looks acceptable on paper.
    _section_names_tr = {
        "sprue": "Döküm ağzı (sprue)",
        "runner": "Yolluk",
        "distributor": "Dağıtıcı",
        "curufluk": "Curufluk",
        "gate": "Meme",
    }
    _section_targets = {
        "sprue": sprue_v_range,
        "runner": runner_v_range,
        "distributor": runner_v_range,
        "curufluk": gate_v_range,
        "gate": gate_v_range,
    }
    for k, v in actual_v.items():
        lo, hi = _section_targets[k]
        if hi > 0 and v > hi:
            result.recommendations.append(
                f"UYARI: {_section_names_tr[k]} gerçek hızı {v:.2f} m/s, "
                f"hedef maksimum {hi:.2f} m/s'yi aşıyor; kesit alanını büyütün veya sayısını artırın."
            )

    # Update the gating system reason with the measured velocities and system.
    gating_system_reason = (
        f"Ölçülen gating sistemi: {detected_system} (önerilen: {recommended_system}). Parça: {wall_cat}. "
        f"Hızlar (tasarım / ölçülen): sprue={v_sprue_design:.2f}/{actual_v.get('sprue', 0.0):.2f}, "
        f"runner={v_runner_design:.2f}/{actual_v.get('runner', 0.0):.2f}, "
        f"dağıtıcı={actual_v.get('distributor', 0.0):.2f}, curufluk={actual_v.get('curufluk', 0.0):.2f}, "
        f"gate={v_gate_design:.2f}/{actual_v.get('gate', 0.0):.2f} m/s. "
        f"Oran As:Ar:Ag ≈ {final_ratio[0]:.2f}:{final_ratio[1]:.2f}:{final_ratio[2]:.2f}."
    )
    if design.warnings:
        gating_system_reason += " Uyarılar: " + "; ".join(design.warnings)

    # Add measured distributor / curufluk flows to the section report.
    if (has_distributor or distributor_area_cm2 > 0.0) and mu > 0.0:
        d_distributor_mm = 1000.0 * math.sqrt(4.0 * max(distributor_area_cm2, 0.0) / math.pi)
        section_flows["DISTRIBUTOR"] = _compute_section_flow(
            "DISTRIBUTOR",
            distributor_area_cm2,
            d_distributor_mm,
            Q_design_m3_s,
            alloy.rho_kg_m3,
            mu,
            9.81,
            runner_v_range[0],
            runner_v_range[1],
            runner_A_min,
            runner_A_max,
        )
    if (has_curufluk or curufluk_area_cm2 > 0.0) and mu > 0.0:
        d_curufluk_mm = 1000.0 * math.sqrt(4.0 * max(curufluk_area_cm2, 0.0) / math.pi)
        section_flows["CURUFLUK"] = _compute_section_flow(
            "CURUFLUK",
            curufluk_area_cm2,
            d_curufluk_mm,
            Q_design_m3_s,
            alloy.rho_kg_m3,
            mu,
            9.81,
            gate_v_range[0],
            gate_v_range[1],
            gate_A_min,
            gate_A_max,
        )

    # Fluidity length with the design gate velocity
    t_stream = max(ingate_thickness_mm, 2.0 * result.dominant_m_mm, 2.0)
    M_stream = t_stream / 2.0
    C = chvorinov_c_from_properties(alloy, mold)
    t_s_stream = C * M_stream ** 2
    superheat = max(alloy.t_pour_c - alloy.t_liquidus_c, 0.0)
    l_eff = alloy.latent_heat_j_kg + alloy.cp_j_kgk * superheat
    superheat_ratio = max(alloy.cp_j_kgk * superheat / l_eff, 0.1) if l_eff > 0 else 0.1
    t_superheat = t_s_stream * superheat_ratio
    # After the cavity is full the metal stops flowing, so cap by the fill time.
    t_superheat = min(t_superheat, design_fill_time_s)
    v_metal_m_s = v_gate_design
    if v_metal_m_s <= 0 and H_eff_m > 0:
        v_metal_m_s = math.sqrt(2.0 * 9.81 * H_eff_m)
    # Fluidity length cannot exceed the physical casting size; cap to avoid
    # unrealistic 5–10 m values while preserving the "can it fill?" check.
    max_flow_path_mm = float(result.bbox_size_mm.max())
    fluidity_length_mm = min(v_metal_m_s * t_superheat * 1000.0, max_flow_path_mm)

    max_dim_mm = float(result.bbox_size_mm.max())
    result.recommendations = [
        r for r in result.recommendations
        if not r.startswith("Sıvı akışkanlık") and not r.startswith("Akışkanlık uzunluğu")
    ]
    if max_dim_mm > fluidity_length_mm:
        result.recommendations.append(
            f"Sıvı akışkanlık uzunluğu Lf = {fluidity_length_mm:.1f} mm, parça boyutu {max_dim_mm:.1f} mm. "
            "Soğuk birleşme (cold shut) riski - döküm sıcaklığını artırın, giriş hızını artırın."
        )
    else:
        result.recommendations.append(
            f"Akışkanlık uzunluğu Lf = {fluidity_length_mm:.1f} mm, parça boyutu {max_dim_mm:.1f} mm -> yeterli."
        )

    velocity_summary = " | ".join(
        f"{k}: {sf.velocity_m_s:.2f}m/s (Re={sf.reynolds:.0f}, Fr={sf.froude:.2f})"
        for k, sf in section_flows.items()
        if sf.area_cm2 > 0
    )
    if velocity_summary:
        result.recommendations.append(f"Kesit hızları (tasarım) -> {velocity_summary}")

    result.recommendations.append(gating_system_reason)

    result.recommendations.append(
        f"Dolum süresi: kullanılan {fill_time_s:.2f} s; pratik öneri {auto_fill_time_s:.2f} s; "
        f"Campbell önerisi {campbell_fill_time_s:.2f} s ({campbell_fill_time_basis}). "
        f"Döküm verimi: %{pour_yield*100:.1f}."
    )

    result.recommendations.append(
        f"Tasarım kesit alanları (As:Ar:Ag={final_ratio[0]:.2f}:{final_ratio[1]:.2f}:{final_ratio[2]:.2f}): "
        f"sprue taban={As_m2*1e4:.2f} cm², runner toplam={Ar_total_m2*1e4:.2f} cm², "
        f"gate toplam={Ag_total_m2*1e4:.2f} cm² (her biri={Ag_each_m2*1e4:.2f} cm²); "
        f"çaplar: sprue Ø={d_sprue_mm:.1f} mm, gate Ø={d_ingate_each_mm:.1f} mm; "
        f"sprue hızı v_c={Vc_ms:.2f} m/s."
    )

    result.recommendations.append(
        f"CAD ölçümü (karşılaştırma): sprue taban={sprue_base_cm2:.2f} cm², runner={runner_min_area_cm2:.2f} cm², "
        f"gate={gate_area_cm2:.2f} cm². Bu alanlarla gerçek hızlar: "
        f"sprue={actual_v.get('sprue', 0.0):.2f}, runner={actual_v.get('runner', 0.0):.2f}, gate={actual_v.get('gate', 0.0):.2f} m/s."
    )

    # Feeder / part mass and volume ratios
    if result.riser_results:
        total_riser_mass_kg = sum(r.mass_kg for r in result.riser_results)
    else:
        riser_volume_cm3 = sum(b.volume_cm3 for b in bodies if b.body_type == BodyType.RISER)
        total_riser_mass_kg = riser_volume_cm3 * alloy.density_g_cm3 / 1000.0
    gating_mass_kg = max(0.0, total_mass_kg - part_mass_kg - total_riser_mass_kg)
    feed_to_part_mass_ratio = ((total_riser_mass_kg + gating_mass_kg) / part_mass_kg) if part_mass_kg > 0 else 0.0
    feed_to_part_volume_ratio = ((total_metal_volume_cm3 - part_volume_cm3) / part_volume_cm3) if part_volume_cm3 > 0 else 0.0
    result.recommendations.append(
        f"Besleyici/yolluk toplam kütlesi = {total_riser_mass_kg + gating_mass_kg:.3f} kg; "
        f"parça kütlesi = {part_mass_kg:.3f} kg; besleyici/parça kütlesi oranı = {feed_to_part_mass_ratio:.2f}; "
        f"hacim oranı = {feed_to_part_volume_ratio:.2f}."
    )

    # Actual (measured/CAD) areas for the report; design areas come from the engine.
    actual_sprue_base_cm2 = sprue_base_cm2 if sprue_base_cm2 > 0.0 else As_m2 * 1e4
    actual_runner_cm2 = runner_min_area_cm2 if runner_min_area_cm2 > 0.0 else Ar_total_m2 * 1e4
    actual_gate_total_cm2 = gate_area_cm2 if gate_area_cm2 > 0.0 else Ag_total_m2 * 1e4

    def _section_ok(
        actual_cm2: float,
        design_cm2: float,
        actual_v_m_s: float,
        target_v_max_m_s: float,
    ) -> bool:
        """A gating section passes only if its area ratio is sane AND its real
        velocity does not exceed the recommended maximum."""
        area_ok = True
        if actual_cm2 > 0.0 and design_cm2 > 0.0:
            ratio = actual_cm2 / design_cm2
            area_ok = 0.6 <= ratio <= 1.5
        velocity_ok = True
        if target_v_max_m_s > 0.0 and actual_v_m_s > target_v_max_m_s:
            velocity_ok = False
        return area_ok and velocity_ok

    # P2: final flow-source overrides for the GateResult fields that the UI uses.
    if flow_result is not None and getattr(flow_result, "Q_m3_s", 0.0) > 0.0:
        ingate_velocity_m_s = getattr(flow_result, "ingate_contact_velocity_m_s", v_gate_design) or v_gate_design
        ingate_flow_rate_m3_s = getattr(flow_result, "total_ingate_flow_m3_s", Q_design_m3_s) or Q_design_m3_s
        ingate_fill_time_s = getattr(flow_result, "fill_time_s", design_fill_time_s) or design_fill_time_s
        velocity_fill_time_match_ok = (
            abs(ingate_fill_time_s - design_fill_time_s)
            <= 0.2 * max(design_fill_time_s, 1e-9)
        )
    else:
        ingate_velocity_m_s = v_gate_design
        ingate_flow_rate_m3_s = Q_design_m3_s
        ingate_fill_time_s = design_fill_time_s
        velocity_fill_time_match_ok = True

    # v9.2: gating fills the mould; it does not fix shrinkage hot spots.
    # Remind the user when unfed hot spots remain so that riser/chill/exothermic
    # decisions are not silently delegated to the gating system.
    if getattr(result, "hotspots", None):
        unfed = [hs for hs in result.hotspots if not hs.feed_ok]
        if unfed:
            has_riser = any(b.body_type == BodyType.RISER for b in bodies)
            if not has_riser:
                result.recommendations.append(
                    f"UYARI: Gating sistemi doldurmayı sağlar; {len(unfed)} adet beslenmeyen "
                    f"hot spot için ayrı riser, çıkıcı (chill) veya ekzotermik mini besleyici gerekebilir."
                )
            else:
                result.recommendations.append(
                    f"UYARI: Gating sistemi doldurmayı sağlar; {len(unfed)} adet hot spot "
                    f"mevcut besleyicilerle beslenemiyor. Besleyici boyutunu/yerini veya ek bir chill değerlendirin."
                )

    return GateResult(
        total_ingate_contact_area_cm2=actual_gate_total_cm2,
        runner_min_area_cm2=actual_runner_cm2,
        sprue_base_area_cm2=actual_sprue_base_cm2,
        required_sprue_area_cm2=As_m2 * 1e4,
        campbell_ok=True,
        bernoulli_ok=_section_ok(actual_sprue_base_cm2, As_m2 * 1e4, actual_v['sprue'], sprue_v_range[1]) if As_m2 > 0 else True,
        ingate_on_thick_region=ingate_on_thick,
        ingate_avg_m_mm=ingate_avg_m,
        ingate_max_m_mm=ingate_max_m,
        ingate_thickness_mm=ingate_thickness_mm,
        runner_thickness_mm=runner_thickness_mm,
        required_runner_area_cm2=Ar_total_m2 * 1e4,
        required_ingate_area_cm2=Ag_total_m2 * 1e4,
        runner_ok=_section_ok(actual_runner_cm2, Ar_total_m2 * 1e4, actual_v['runner'], runner_v_range[1]),
        ingate_ok=_section_ok(actual_gate_total_cm2, Ag_total_m2 * 1e4, actual_v['gate'], gate_v_range[1]),
        elbow_count=elbow_count,
        head_loss_mm=head_loss_m * 1000.0,
        effective_head_mm=H_eff_m * 1000.0,
        required_sprue_area_with_losses_cm2=As_m2 * 1e4,
        ingate_velocity_m_s=ingate_velocity_m_s,
        ingate_max_velocity_m_s=gate_v_range[1],
        reynolds=ingate_flow.reynolds,
        froude=ingate_flow.froude,
        turbulent=(ingate_flow.turbulent or actual_v.get('gate', 0.0) > gate_v_range[1]),
        ingate_flow_rate_m3_s=ingate_flow_rate_m3_s,
        ingate_fill_time_s=ingate_fill_time_s,
        velocity_fill_time_match_ok=velocity_fill_time_match_ok,
        required_ingate_area_for_velocity_cm2=Ag_each_m2 * 1e4,
        velocity_area_ok=(
            _section_ok(actual_sprue_base_cm2, As_m2 * 1e4, actual_v['sprue'], sprue_v_range[1])
            and _section_ok(actual_runner_cm2, Ar_total_m2 * 1e4, actual_v['runner'], runner_v_range[1])
            and _section_ok(actual_gate_total_cm2, Ag_total_m2 * 1e4, actual_v['gate'], gate_v_range[1])
        ),
        fluidity_length_mm=fluidity_length_mm,
        sprue_throat_area_cm2=sprue_throat_cm2 if sprue_throat_cm2 > 0.0 else design.sprue_throat_area_cm2,
        sprue_base_bottom_area_cm2=actual_sprue_base_cm2,
        sprue_thickness_mm=sprue_thickness_mm,
        selected_section_key=user_velocity_section,
        selected_velocity_m_s=user_gate_velocity,
        section_flows=section_flows,
        effective_gate_section="INGATE" if has_ingate else "RUNNER (meme yok)",
        detected_gating_system=detected_system,
        recommended_gating_system=recommended_system,
        wall_thickness_category=wall_cat,
        gating_system_reason=gating_system_reason,
        recommended_fill_time_s=recommended_fill_time_s,
        fill_time_basis=fill_time_basis,
        auto_fill_time_s=auto_fill_time_s,
        campbell_fill_time_s=campbell_fill_time_s,
        campbell_fill_time_basis=campbell_fill_time_basis,
        head_reduction_percent=head_reduction_percent,
        total_poured_mass_kg=total_mass_kg,
        pouring_yield=pour_yield,
        design_sprue_base_area_cm2=As_m2 * 1e4,
        design_runner_area_cm2=Ar_total_m2 * 1e4,
        design_distributor_area_cm2=(Ar_total_m2 + Ag_total_m2) / 2.0 * 1e4,
        design_gate_total_area_cm2=Ag_total_m2 * 1e4,
        design_gate_each_area_cm2=Ag_each_m2 * 1e4,
        design_sprue_diameter_mm=d_sprue_mm,
        design_gate_diameter_mm=d_ingate_each_mm,
        design_choke_velocity_m_s=Vc_ms,
        design_gating_ratio=final_ratio,
        sprue_design_ok=_section_ok(actual_sprue_base_cm2, As_m2 * 1e4, actual_v['sprue'], sprue_v_range[1]),
        runner_design_ok=_section_ok(actual_runner_cm2, Ar_total_m2 * 1e4, actual_v['runner'], runner_v_range[1]),
        gate_design_ok=_section_ok(actual_gate_total_cm2, Ag_total_m2 * 1e4, actual_v['gate'], gate_v_range[1]),
        distributor_area_cm2=distributor_area_cm2,
        curufluk_area_cm2=curufluk_area_cm2,
        distributor_velocity_m_s=actual_v.get("distributor", 0.0),
        curufluk_velocity_m_s=actual_v.get("curufluk", 0.0),
        part_mass_kg=part_mass_kg,
        total_riser_mass_kg=total_riser_mass_kg,
        gating_mass_kg=gating_mass_kg,
        feed_to_part_mass_ratio=feed_to_part_mass_ratio,
        feed_to_part_volume_ratio=feed_to_part_volume_ratio,
    )

