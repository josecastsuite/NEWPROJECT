"""Ingate / runner / sprue geometric gating calculations - JoseCast v8.0."""

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import math

import numpy as np
import trimesh
from scipy import ndimage
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
    _score_systems,
    _section_velocity_limit,
    _wall_class,
    calculate_gating_design,
)
from scipy.spatial.distance import cdist
from core.materials import get_alloy, get_mold, chvorinov_c_from_properties
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
        # Prefer the 3-D mesh contact-surface velocity when available.
        v_report = (
            node.max_velocity_m_s
            if node.max_velocity_m_s > 1e-12
            else node.velocity_m_s
        )
        section_data.setdefault(up, []).append(
            (v_report, node.section_area_cm2, node.flow_rate_m3_s)
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
        # Target area range from Q = v A: A_min = Q / v_max, A_max = Q / v_min.
        q_total_m3_s = total_q
        if hi > 0.0 and q_total_m3_s > 0.0:
            a_min = float(q_total_m3_s / hi) * 1e4
        else:
            a_min = 0.0
        if lo > 0.0 and q_total_m3_s > 0.0:
            a_max = float(q_total_m3_s / lo) * 1e4
        else:
            a_max = 1e9
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
        # fill_holes() can be extremely slow / hang on dense meshes, so only
        # merge vertices and remove unreferenced vertices; the section-based
        # area calculation tolerates small non-manifold regions.
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


def _bbox_connection_axis(
    up_bbox: np.ndarray,
    dn_bbox: np.ndarray,
    up_center: np.ndarray,
    dn_center: np.ndarray,
    tol_mm: float = 0.5,
) -> Optional[np.ndarray]:
    """Return the axis (unit vector) along which two gating bodies face/feed each other.

    The connection is a pair of bounding boxes that touch/overlap in two axes and
    are separated by at most ``tol_mm`` in the third axis.  This is far more
    robust than centroid-to-centroid for offset runners, angled sprues etc.
    """
    axes = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    best_i = -1
    best_contact = -1.0
    best_sign = 1.0
    for i in range(3):
        a_min, a_max = float(up_bbox[0][i]), float(up_bbox[1][i])
        b_min, b_max = float(dn_bbox[0][i]), float(dn_bbox[1][i])
        overlap = min(a_max, b_max) - max(a_min, b_min)
        other = [
            min(up_bbox[1][j], dn_bbox[1][j]) - max(up_bbox[0][j], dn_bbox[0][j])
            for j in range(3) if j != i
        ]
        # Bodies must overlap or just touch in the other two axes; the candidate
        # axis must not be deeply embedded (overlap <= tol) and not too far apart.
        if overlap < -tol_mm or overlap > tol_mm or other[0] <= 0.0 or other[1] <= 0.0:
            continue
        contact = float(other[0]) * float(other[1])
        if contact > best_contact:
            best_contact = contact
            best_i = i
            if b_min >= a_max - tol_mm:
                best_sign = 1.0
            elif b_max <= a_min + tol_mm:
                best_sign = -1.0
            else:
                best_sign = 1.0 if dn_center[i] > up_center[i] else -1.0
    if best_i < 0:
        return None
    return float(best_sign) * axes[best_i]


def _downstream_body_for(body: Body, bodies: List[Body]) -> Optional[Body]:
    """Pick the downstream body that this gating element feeds."""
    candidate_types: Tuple[BodyType, ...]
    if body.body_type == BodyType.SPRUE:
        candidate_types = (BodyType.RUNNER, BodyType.INGATE, BodyType.PART)
    elif body.body_type == BodyType.RUNNER:
        candidate_types = (BodyType.INGATE, BodyType.PART)
    elif body.body_type == BodyType.INGATE:
        candidate_types = (BodyType.PART,)
    else:
        return None

    best: Optional[Body] = None
    best_sep = float("inf")
    b = _repair_mesh(body.mesh)
    for cand in bodies:
        if cand is body or cand.body_type not in candidate_types:
            continue
        cb = _repair_mesh(cand.mesh)
        axis = _bbox_connection_axis(b.bounds, cb.bounds, b.vertices.mean(axis=0), cb.vertices.mean(axis=0))
        if axis is None:
            continue
        # compute scalar separation along that axis
        i = int(np.argmax(np.abs(axis)))
        sep = max(
            0.0,
            max(b.bounds[0][i] - cb.bounds[1][i], cb.bounds[0][i] - b.bounds[1][i]),
        )
        if sep < best_sep:
            best_sep = sep
            best = cand
    return best


def _fallback_flow_axis(mesh: trimesh.Trimesh, downstream_center: Optional[np.ndarray] = None) -> np.ndarray:
    """When no downstream body is adjacent, use the longest bbox dimension as the flow axis."""
    sizes = mesh.bounds[1] - mesh.bounds[0]
    i = int(np.argmax(sizes))
    axis = np.zeros(3, dtype=float)
    axis[i] = 1.0
    if downstream_center is not None:
        v = downstream_center - mesh.vertices.mean(axis=0)
        if v[i] < 0:
            axis[i] = -1.0
    return axis


def _characteristic_cross_section_area(
    mesh: trimesh.Trimesh,
    axis: np.ndarray,
    n: int = 20,
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
    n: int = 20,
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


def _body_exit_or_throat_area(
    mesh: trimesh.Trimesh,
    axis: np.ndarray,
    is_sprue: bool = False,
    fallback_min: bool = False,
    n: int = 80,
) -> float:
    """Return the representative cross-sectional area [mm²] of a gating body.

    For runner/ingate the area is taken near the downstream end (``axis`` points
    downstream).  For a sprue the throat is the smaller of the two end areas.
    If ``fallback_min`` is true the minimum area along the axis is used (e.g. an
    ingate with no part contact).
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

    areas = np.array([r[1] for r in rows], dtype=np.float64)
    max_area = float(areas.max())
    if max_area <= 0.0:
        return 0.0

    # Ignore degenerate partial end caps.
    valid_idx = [i for i, a in enumerate(areas) if a > 0.05 * max_area]
    if not valid_idx:
        return float(np.median(areas))

    n_end = max(1, min(5, len(valid_idx) // 4))
    if fallback_min:
        return float(min(areas[i] for i in valid_idx))
    if is_sprue:
        front = [areas[i] for i in valid_idx[:n_end]]
        back = [areas[i] for i in valid_idx[-n_end:]]
        return float(min(front + back))
    # runner / ingate: downstream end is the last valid rows
    return float(np.mean([areas[i] for i in valid_idx[-n_end:]]))


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

    Areas are measured perpendicular to the local flow axis; for a sprue both the
    base and the smaller throat are returned.  This matches how metal actually
    flows and avoids the old centroid/PCA heuristics.
    """
    runner_total_mm2 = 0.0
    distributor_total_mm2 = 0.0
    curufluk_total_mm2 = 0.0
    ingate_total_mm2 = 0.0
    n_ingates = 0
    sprue_bases: List[float] = []
    sprue_throats: List[float] = []

    gating_types = (
        BodyType.SPRUE,
        BodyType.SPRUE_THROAT,
        BodyType.RUNNER,
        BodyType.DISTRIBUTOR,
        BodyType.CURUFLUK,
        BodyType.INGATE,
    )

    for body in bodies:
        if body.body_type not in gating_types:
            continue
        mesh = _repair_mesh(body.mesh)
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue

        downstream = _downstream_body_for(body, bodies)
        if downstream is not None:
            dn_mesh = _repair_mesh(downstream.mesh)
            axis = _bbox_connection_axis(
                mesh.bounds, dn_mesh.bounds,
                mesh.vertices.mean(axis=0), dn_mesh.vertices.mean(axis=0),
            )
        else:
            axis = None

        if axis is None:
            dn_center = downstream.mesh.vertices.mean(axis=0) if downstream is not None else None
            axis = _fallback_flow_axis(mesh, dn_center)

        main_mm2, throat_mm2 = _body_cross_section_mm2(body, axis=axis)
        main_mm2 = float(main_mm2) if main_mm2 > 0.0 else 0.0
        throat_mm2 = float(throat_mm2) if throat_mm2 is not None and throat_mm2 > 0.0 else main_mm2

        if body.body_type in (BodyType.SPRUE, BodyType.SPRUE_THROAT):
            sprue_bases.append(main_mm2)
            sprue_throats.append(throat_mm2)
        elif body.body_type == BodyType.RUNNER:
            runner_total_mm2 += main_mm2
        elif body.body_type == BodyType.DISTRIBUTOR:
            distributor_total_mm2 += main_mm2
        elif body.body_type == BodyType.CURUFLUK:
            curufluk_total_mm2 += main_mm2
        elif body.body_type == BodyType.INGATE:
            ingate_total_mm2 += main_mm2
            n_ingates += 1

    sprue_base_mm2 = float(np.sum(sprue_bases)) if sprue_bases else 0.0
    positive_throats = [t for t in sprue_throats if t > 0.0]
    sprue_throat_mm2 = (
        float(np.min(positive_throats))
        if positive_throats
        else (float(np.sum(sprue_throats)) if sprue_throats else 0.0)
    )

    return {
        "runner_total_mm2": runner_total_mm2,
        "runner_total_cm2": runner_total_mm2 / 100.0,
        "distributor_total_mm2": distributor_total_mm2,
        "distributor_total_cm2": distributor_total_mm2 / 100.0,
        "curufluk_total_mm2": curufluk_total_mm2,
        "curufluk_total_cm2": curufluk_total_mm2 / 100.0,
        "ingate_total_mm2": ingate_total_mm2,
        "ingate_total_cm2": ingate_total_mm2 / 100.0,
        "sprue_base_mm2": sprue_base_mm2,
        "sprue_base_cm2": sprue_base_mm2 / 100.0,
        "sprue_throat_mm2": sprue_throat_mm2,
        "sprue_throat_cm2": sprue_throat_mm2 / 100.0,
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


def _count_elbows_from_gating_nodes(
    gating_nodes,
    angle_threshold_deg: float = 60.0,
) -> int:
    """Count sharp direction changes along the gating tree using node centroids.

    Replaces the heavy 26-neighbour Dijkstra + gradient-descent path trace that
    was freezing on large gating systems.  Each ``GatingNode`` already carries
    the 3-D centroid of its connecting section, so the elbow count is derived
    from the angles between consecutive node-to-node vectors.
    """
    if not gating_nodes:
        return 0

    # Build an adjacency map keyed by the upstream body name.
    children: Dict[str, List] = {}
    roots = []
    for node in gating_nodes:
        name = node.name
        if "->" not in name:
            continue
        up, down = [s.strip() for s in name.split("->", 1)]
        if up == "Kaynak":
            roots.append(node)
        children.setdefault(up, []).append(node)

    if not roots:
        return 0

    cos_thresh = math.cos(math.radians(angle_threshold_deg))

    def _walk(node, prev_centroid):
        # Gather centroid path from this node downstream to every leaf.
        paths = []
        centroid = np.array(node.centroid_mm, dtype=np.float64)
        path = [prev_centroid, centroid] if prev_centroid is not None else [centroid]
        _, down = [s.strip() for s in node.name.split("->", 1)]
        kids = children.get(down, [])
        if not kids:
            return [path]
        for kid in kids:
            for sub in _walk(kid, centroid):
                paths.append(path + sub[1:])
        return paths

    counts = []
    for root in roots:
        # The source node's centroid is the upstream (pouring) point.
        for path in _walk(root, None):
            if len(path) < 3:
                continue
            elbows = 0
            for i in range(1, len(path) - 1):
                v1 = np.array(path[i]) - np.array(path[i - 1])
                v2 = np.array(path[i + 1]) - np.array(path[i])
                n1 = float(np.linalg.norm(v1))
                n2 = float(np.linalg.norm(v2))
                if n1 < 1e-12 or n2 < 1e-12:
                    continue
                cosang = float(np.dot(v1, v2)) / (n1 * n2)
                if cosang < cos_thresh:
                    elbows += 1
            counts.append(elbows)

    if not counts:
        return 0
    return int(round(float(np.median(counts))))


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


def _critical_velocity_m_s(alloy) -> float:
    """Alloy-specific critical entrainment / meniscus velocity (Campbell ceiling)."""
    return float(getattr(alloy, "critical_entrainment_velocity_m_s", 0.5) or 0.5)


def _oxidation_risk(alloy) -> float:
    """Relative oxide-film sensitivity used when choosing gate/system severity."""
    family = (alloy.material_family or "").lower()
    if family in ("al", "aluminum", "aluminium", "mg", "magnesium"):
        return 1.5
    if family in ("ductile_iron", "nodular", "sfero", "ggg", "sg"):
        return 1.2
    if family in ("stainless", "steel"):
        return 0.9
    if family in ("gray_iron", "grey_iron", "gri pik", "pik"):
        return 0.7
    if family in ("cu", "copper", "bronze", "brass"):
        return 1.0
    return 1.0


def _mold_runner_erosion_limit_m_s(mold) -> float:
    """Upper runner velocity limit to avoid sand erosion / wash-out."""
    name = (getattr(mold, "name", "") or "").lower()
    if "yeşil" in name or "green" in name or "kum" in name:
        return 1.2
    if "silica" in name or "silis" in name:
        return 1.5
    if "zircon" in name or "chromite" in name:
        return 2.0
    return 1.2


def _compute_flow_path_mm(result, source_mask=None) -> float:
    """Approximate longest flow distance from the gate/sprue to the far part point."""
    part_mask = result.grid == BodyType.PART
    if not part_mask.any():
        return float(result.bbox_size_mm.max())
    if source_mask is None or not source_mask.any():
        ingate = result.grid == BodyType.INGATE
        source_mask = ingate if ingate.any() else (
            (result.grid == BodyType.SPRUE) | (result.grid == BodyType.SPRUE_THROAT)
        )
    if not source_mask.any():
        return float(result.bbox_size_mm.max())
    marker = np.ones(part_mask.shape, dtype=np.uint8)
    marker[source_mask] = 0
    try:
        dist = ndimage.distance_transform_edt(
            marker,
            sampling=(float(result.dx_mm), float(result.dx_mm), float(result.dx_mm)),
        )
    except Exception:
        dist = ndimage.distance_transform_edt(marker)
    valid = part_mask & np.isfinite(dist)
    if not valid.any():
        return float(result.bbox_size_mm.max())
    return float(dist[valid].max())


def _isolated_hotspot_count(hotspots, threshold_mm: float) -> int:
    """Count hotspots whose nearest neighbour is farther than threshold."""
    n = len(hotspots)
    if n <= 1:
        return 0
    positions = np.array([np.asarray(getattr(h, "position_mm", h)).flatten()[:3] for h in hotspots])
    if positions.shape[0] > 100:
        positions = positions[:100]
    dists = cdist(positions, positions)
    np.fill_diagonal(dists, np.inf)
    nearest = dists.min(axis=1)
    return int(np.sum(nearest > threshold_mm))


def _fluidity_length_mm(v_metal_m_s, alloy, mold, t_stream_mm: float, fill_time_s: Optional[float] = None) -> float:
    """Length a fluid metal stream of thickness t_stream can travel before freezing."""
    M_stream = max(t_stream_mm, 2.0) / 2.0
    C = chvorinov_c_from_properties(alloy, mold)
    # C is in dk/cm^2, M_stream in mm -> t_s in seconds
    t_s_stream = C * (M_stream / 10.0) ** 2 * 60.0
    superheat = max(alloy.t_pour_c - alloy.t_liquidus_c, 0.0)
    l_eff = alloy.latent_heat_j_kg + alloy.cp_j_kgk * superheat
    superheat_ratio = max(alloy.cp_j_kgk * superheat / l_eff, 0.1) if l_eff > 0 else 0.1
    t_superheat = t_s_stream * superheat_ratio
    if fill_time_s:
        t_superheat = min(t_superheat, fill_time_s)
    return v_metal_m_s * t_superheat * 1000.0


def _part_fingerprint(result, bodies=None, n_ingates: int = 1) -> Dict[str, float]:
    """Geometry-aware part signature from the SDF / voxel grid and hotspots."""
    part_mask = result.grid == BodyType.PART
    if result.subvoxel_sdf.size and part_mask.any():
        thickness = (2.0 * result.subvoxel_sdf[part_mask]).astype(float)
    else:
        thickness = np.array([getattr(result, "wall_thickness_mm", 10.0) or 10.0], dtype=float)

    t_mean = float(np.mean(thickness))
    t_std = float(np.std(thickness))
    t_min = float(np.min(thickness))
    t_max = float(np.max(thickness))
    thin_vol_ratio = float(np.mean(thickness < 3.0))
    thick_vol_ratio = float(np.mean(thickness > 15.0))
    complexity = t_std / max(t_mean, 1e-6)

    ingate = result.grid == BodyType.INGATE
    source_mask = ingate if ingate.any() else (
        (result.grid == BodyType.SPRUE) | (result.grid == BodyType.SPRUE_THROAT)
    )
    flow_path_mm = _compute_flow_path_mm(result, source_mask)
    flow_path_mm = flow_path_mm * (1.0 + 0.3 * min(complexity, 2.0))
    flow_ratio = flow_path_mm / max(t_mean, 1.0)

    hotspots = getattr(result, "hotspots", None) or []
    threshold = max(t_mean * 2.0, 20.0)
    isolated_count = _isolated_hotspot_count(hotspots, threshold)

    return {
        "t_mean_mm": t_mean,
        "t_std_mm": t_std,
        "t_min_mm": t_min,
        "t_max_mm": t_max,
        "thin_vol_ratio": thin_vol_ratio,
        "thick_vol_ratio": thick_vol_ratio,
        "complexity": complexity,
        "flow_path_mm": flow_path_mm,
        "flow_ratio": flow_ratio,
        "isolated_hotspot_count": isolated_count,
    }


def _classify_from_ingate_velocity(
    v_ingate_m_s: float,
    alloy,
    wall_thickness_mm: float,
) -> Tuple[str, str]:
    """Classify the real gating system from the actual ingate velocity.

    Uses the alloy-specific critical entrainment velocity as the reference ceiling.
    """
    v_crit = _critical_velocity_m_s(alloy)
    if v_ingate_m_s >= v_crit * 0.85:
        return (
            "basınçlı (pressurized)",
            f"Meme hızı {v_ingate_m_s:.2f} m/s; kritik menisküs hızı {v_crit:.2f} m/s'nin %85'i üzerinde, "
            f"sistem basınçlı çalışıyor.",
        )
    if v_ingate_m_s <= v_crit * 0.45:
        return (
            "basınçsız (unpressurized)",
            f"Meme hızı {v_ingate_m_s:.2f} m/s; kritik menisküs hızı {v_crit:.2f} m/s'nin %45'i altında, "
            f"sistem basınçsız çalışıyor.",
        )
    return (
        "yarı basınçlı (semi-pressurized)",
        f"Meme hızı {v_ingate_m_s:.2f} m/s; kritik menisküs hızı {v_crit:.2f} m/s civarında, "
        f"sistem yarı basınçlı çalışıyor.",
    )


def _smart_gating_design(
    result,
    alloy,
    mold,
    casting_params,
    bodies,
    total_mass_kg: float,
    t_fill_s: float,
    H_eff_m: float,
    n_ingates: int,
    ingate_thickness_mm: float,
    wall_thickness_mm: float,
    gravity_vec,
    design,
    user_gate_velocity: float = 0.0,
    fingerprint: Optional[Dict[str, float]] = None,
) -> Dict[str, any]:
    """Geometry- and alloy-aware gating area / system recommendation.

    Derives As, Ar, Ag from Q = v*A with a choke velocity that respects the
    alloy-specific critical entrainment velocity and the available effective
    head.  Searches pressurized / semi / unpressurized candidates and returns
    the design with the lowest combined oxide / erosion / cold-shut / area score.
    """
    Vcrit = _critical_velocity_m_s(alloy)
    if fingerprint is None:
        fingerprint = _part_fingerprint(result, bodies, n_ingates)
    rho = alloy.rho_kg_m3
    Q = total_mass_kg / (rho * t_fill_s) if t_fill_s > 0 and rho > 0 else 0.0
    Cd = float(getattr(casting_params, "discharge_coeff", 0.8) or 0.8)
    v_s_bernoulli = Cd * math.sqrt(2.0 * 9.81 * max(H_eff_m, 0.02))
    V_choke = min(v_s_bernoulli, Vcrit)

    flow_path_mm = fingerprint["flow_path_mm"]
    t_mean_mm = fingerprint["t_mean_mm"]
    flow_ratio = fingerprint["flow_ratio"]

    V_required = flow_path_mm / (t_fill_s * 1000.0) if t_fill_s > 0 else 0.0
    warnings: List[str] = []

    if user_gate_velocity > 0:
        V_gate = min(user_gate_velocity, Vcrit, V_choke)
        if user_gate_velocity > V_choke:
            warnings.append(
                f"Kullanıcı meme hızı {user_gate_velocity:.2f} m/s, etkin başlıkla mümkün olan "
                f"{V_choke:.2f} m/s ile sınırlandı."
            )
    else:
        V_gate = min(max(V_required * 1.1, V_choke * 0.25), V_choke)

    if V_required > V_choke:
        warnings.append(
            f"Akış yolu ({flow_path_mm:.0f} mm) için gereken ortalama hız "
            f"{V_required:.2f} m/s, fiziksel limit {V_choke:.2f} m/s'yi aşıyor; "
            f"dolum süresi veya H_eff artırılmalı."
        )

    oxidation = _oxidation_risk(alloy)
    t_stream_mm = max(ingate_thickness_mm, 2.0 * t_mean_mm, 2.0)
    L_fluid_mm = _fluidity_length_mm(V_gate, alloy, mold, t_stream_mm, t_fill_s)
    cold_shut_risk = max(0.0, flow_path_mm - L_fluid_mm) / max(L_fluid_mm, 1.0)

    def _engine_system_scores():
        inp = GatingEngineInput(
            total_metal_volume_m3=0.0,
            total_mass_kg=0.0,
            alloy_key=alloy.key,
        )
        part_mask = result.grid == BodyType.PART
        t_min = fingerprint["t_min_mm"]
        t_max = fingerprint["t_max_mm"]
        sv = (
            result.part_surface_area_mm2 / result.part_volume_mm3
            if result.part_volume_mm3 > 0.0
            else 0.0
        )
        D_bulk = 2.0 * t_mean_mm
        slenderness = flow_path_mm / max(D_bulk, 1.0)
        head_ratio = (H_eff_m * 1000.0) / max(flow_path_mm, 1.0)
        thickness_var = t_max / max(t_min, 1.0)
        pore_risk_max = 0.0
        if result.risk.size and part_mask.any():
            pore_risk_max = float(result.risk[part_mask].max())
        features = {
            "t_avg": t_mean_mm,
            "t_min": t_min,
            "t_max": t_max,
            "surface_to_volume_ratio": sv,
            "slenderness": slenderness,
            "flow_ratio": flow_ratio,
            "head_ratio": head_ratio,
            "hotspot_count": float(len(getattr(result, "hotspots", []) or [])),
            "max_hotspot_m_mm": max([h.m_value_mm for h in result.hotspots] or [0.0]),
            "pore_risk_max": pore_risk_max,
            "thickness_var": thickness_var,
        }
        _, scores, _ = _score_systems(inp, features)
        return scores

    engine_scores = _engine_system_scores()

    def _re(area_m2: float, v_m_s: float) -> float:
        D = max(math.sqrt(4.0 * area_m2 / math.pi), 1e-6)
        return rho * v_m_s * D / max(alloy.viscosity_pa_s, 1e-6)

    runner_erosion_limit = _mold_runner_erosion_limit_m_s(mold)

    def base_score(As: float, Ar: float, Ag: float, Vs: float, Vr: float, Vg: float, system: str) -> float:
        s = 0.0
        if Vg > Vcrit:
            s += 1e6
        if Vs > Vcrit * 1.5:
            s += 1e6
        if Vr > Vcrit * 1.2:
            s += 1e6
        s += oxidation * 30.0 * max(0.0, Vs - Vcrit * 0.9) / max(Vcrit, 0.1)
        s += oxidation * 30.0 * max(0.0, Vr - Vcrit * 0.9) / max(Vcrit, 0.1)
        s += 50.0 * max(0.0, Vr - runner_erosion_limit) / max(runner_erosion_limit, 0.1)
        s += 20.0 * max(0.0, _re(As, Vs) - 20000.0) / 20000.0
        s += 20.0 * max(0.0, _re(Ar, Vr) - 20000.0) / 20000.0
        s += 20.0 * max(0.0, _re(Ag, Vg) - 20000.0) / 20000.0
        s += 40.0 * cold_shut_risk
        ref = Q / max(V_gate, 1e-6)
        s += 12.0 * max(0.0, (As + Ar + Ag) / max(ref, 1e-9) - 1.0)
        s += (100.0 - engine_scores.get(system, 50.0)) * 0.5
        return s

    candidates = []
    if V_gate > 0 and Q > 0:
        # Pressurized: gate is the choke, Vg highest.
        for pf in np.linspace(0.30, 0.95, 20):
            Vg = V_gate
            Vs = Vg * pf
            r_run = pf + (1.0 - pf) * 0.55
            Vr = Vg * pf / r_run
            Vs = min(Vs, Vcrit * 1.5, V_choke)
            Vr = min(Vr, Vcrit * 1.2, V_choke)
            if not (Vs > 0 and Vr > Vs and Vg > Vr):
                continue
            As = Q / Vs
            Ar = Q / Vr
            Ag = Q / Vg
            Pf = Ag / As
            system = "basınçlı (pressurized)"
            score = base_score(As, Ar, Ag, Vs, Vr, Vg, system)
            candidates.append(
                {"As": As, "Ar": Ar, "Ag": Ag, "Vs": Vs, "Vr": Vr, "Vg": Vg, "system": system, "Pf": Pf, "score": score}
            )

        # Unpressurized: sprue is the choke, Vs highest.
        for pf in np.linspace(1.05, 2.5, 20):
            Vs = min(V_gate * pf, V_choke)
            Vg = Vs / pf
            r_run = 1.0 + (pf - 1.0) * 0.5
            Vr = Vs / r_run
            Vs = min(Vs, Vcrit * 1.5)
            Vr = min(Vr, Vcrit * 1.2)
            if not (Vg > 0 and Vr > Vg and Vs > Vr):
                continue
            As = Q / Vs
            Ar = Q / Vr
            Ag = Q / Vg
            Pf = Ag / As
            system = "basınçsız (unpressurized)"
            score = base_score(As, Ar, Ag, Vs, Vr, Vg, system)
            candidates.append(
                {"As": As, "Ar": Ar, "Ag": Ag, "Vs": Vs, "Vr": Vr, "Vg": Vg, "system": system, "Pf": Pf, "score": score}
            )

        # Semi-pressurized: runner is the choke, Vr highest.
        for rs in np.linspace(1.1, 2.0, 6):
            for rg in np.linspace(1.1, 2.0, 6):
                Vr = min(V_gate * 1.05, V_choke)
                Vs = Vr / rs
                Vg = Vr / rg
                Vs = min(Vs, Vcrit * 1.5, V_choke)
                Vg = min(Vg, Vcrit, V_choke)
                if Vr <= 0 or not (max(Vs, Vg) < Vr):
                    continue
                As = Q / Vs
                Ar = Q / Vr
                Ag = Q / Vg
                Pf = Ag / As
                system = "yarı basınçlı (semi-pressurized)"
                score = base_score(As, Ar, Ag, Vs, Vr, Vg, system)
                candidates.append(
                    {"As": As, "Ar": Ar, "Ag": Ag, "Vs": Vs, "Vr": Vr, "Vg": Vg, "system": system, "Pf": Pf, "score": score}
                )

    if not candidates:
        # Fallback equal-area semi.
        Vg = max(V_gate, 0.1)
        Vs = Vg * 0.6
        Vr = Vg * 0.8
        As = Q / max(Vs, 0.01)
        Ar = Q / max(Vr, 0.01)
        Ag = Q / max(Vg, 0.01)
        best = {"As": As, "Ar": Ar, "Ag": Ag, "Vs": Vs, "Vr": Vr, "Vg": Vg, "system": "yarı basınçlı (semi-pressurized)", "Pf": 1.0, "score": 0.0}
    else:
        best = min(candidates, key=lambda c: c["score"])

    # Auto-correction
    if best["Vr"] > runner_erosion_limit:
        best["Ar"] = Q / runner_erosion_limit
        best["Vr"] = runner_erosion_limit

    for area_key, vel_key in (("As", "Vs"), ("Ar", "Vr"), ("Ag", "Vg")):
        area = best[area_key]
        v = best[vel_key]
        Re = _re(area, v)
        if Re > 20000.0:
            factor = (Re / 20000.0) ** 2
            best[area_key] = area * factor
            best[vel_key] = v / factor

    if best["Vg"] > Vcrit:
        factor = best["Vg"] / (Vcrit * 0.95)
        best["Ag"] *= factor
        best["Vg"] = Vcrit * 0.95

    As = best["As"]
    Ar = best["Ar"]
    Ag = best["Ag"]
    Vs = best["Vs"]
    Vr = best["Vr"]
    Vg = best["Vg"]
    system = best["system"]
    Pf = best["Pf"]

    Ag_each = Ag / max(n_ingates, 1)
    d_sprue_mm = 1000.0 * math.sqrt(4.0 * max(As, 0.0) / math.pi)
    d_ingate_each_mm = 1000.0 * math.sqrt(4.0 * max(Ag_each, 0.0) / math.pi)
    final_ratio = (1.0, float(Ar / max(As, 1e-12)), float(Ag / max(As, 1e-12)))

    L_fluid_final = _fluidity_length_mm(Vg, alloy, mold, t_stream_mm, t_fill_s)
    cold_shut_final = max(0.0, flow_path_mm - L_fluid_final) / max(L_fluid_final, 1.0)
    if cold_shut_final > 0.0:
        warnings.append(
            f"Akışkanlık uzunluğu ({L_fluid_final:.0f} mm) akış yolunu ({flow_path_mm:.0f} mm) "
            f"karşılamıyor; soğuk birleşme riski var."
        )

    reason = (
        f"Parça {alloy.name} için akıllı gate analizi: "
        f"ortalama kalınlık {t_mean_mm:.1f} mm, akış yolu {flow_path_mm:.0f} mm, "
        f"L/t = {flow_ratio:.1f}, ince cidarlı hacim %{fingerprint['thin_vol_ratio']*100:.0f}, "
        f"kalın cidarlı hacim %{fingerprint['thick_vol_ratio']*100:.0f}. "
        f"Kritik menisküs hızı {Vcrit:.2f} m/s; etkin başlık {H_eff_m:.3f} m ile "
        f"Bernoulli hızı {v_s_bernoulli:.2f} m/s, dolayısıyla hedef gate hızı {V_gate:.2f} m/s. "
        f"Önerilen sistem {system} (Pf = {Pf:.2f}); "
        f"sprue hızı {Vs:.2f} m/s, runner hızı {Vr:.2f} m/s, gate hızı {Vg:.2f} m/s. "
        f"Buna göre As:Ar:Ag ≈ {final_ratio[0]:.2f}:{final_ratio[1]:.2f}:{final_ratio[2]:.2f}; "
        f"sprue tabanı {As*1e4:.2f} cm², runner toplam {Ar*1e4:.2f} cm², "
        f"gate toplam {Ag*1e4:.2f} cm² (her biri {Ag_each*1e4:.2f} cm²); "
        f"çaplar sprue Ø{d_sprue_mm:.1f} mm, gate Ø{d_ingate_each_mm:.1f} mm. "
        f"Akışkanlık uzunluğu {L_fluid_final:.0f} mm. Benim önerim budur."
    )
    if warnings:
        reason += " Uyarılar: " + "; ".join(warnings)

    return {
        "As_m2": As,
        "Ar_total_m2": Ar,
        "Ag_total_m2": Ag,
        "Ag_each_m2": Ag_each,
        "v_sprue_design": Vs,
        "v_runner_design": Vr,
        "v_gate_design": Vg,
        "v_sprue_bernoulli": v_s_bernoulli,
        "v_choke_m_s": V_choke,
        "Q_design_m3_s": Q,
        "d_sprue_mm": d_sprue_mm,
        "d_ingate_each_mm": d_ingate_each_mm,
        "final_ratio": final_ratio,
        "recommended_system": system,
        "fingerprint": fingerprint,
        "Vcrit": Vcrit,
        "V_gate": V_gate,
        "flow_path_mm": flow_path_mm,
        "flow_ratio": flow_ratio,
        "L_fluid_mm": L_fluid_final,
        "cold_shut_risk": cold_shut_final,
        "reason": reason,
        "warnings": warnings,
    }


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
        gravity_vector = (
            getattr(casting_params, "gravity_direction", None)
            or getattr(casting_params, "gravity_vector", None)
            or gravity_vector
        )
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
    gravity_vec = np.array([0.0, 0.0, -1.0])
    if casting_params is not None and isinstance(casting_params, CastingParameters):
        gv = getattr(casting_params, "gravity_direction", None)
        if gv is not None:
            gv = np.asarray(gv, dtype=float)
            n = float(np.linalg.norm(gv))
            if n > 1e-6:
                gravity_vec = gv / n
    if fill_time_s is None:
        fill_time_s = 10.0

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
        # If the system has no ingate, the runner exit feeds the part directly.
        # Use the smaller of the runner exit cross-section and the part-runner
        # contact area as the effective gate area.
        if runner_min_area_cm2 > 0.0 and not has_ingate:
            if gate_area_cm2 <= 0.0:
                gate_area_cm2 = runner_min_area_cm2
            else:
                gate_area_cm2 = min(gate_area_cm2, runner_min_area_cm2)

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

    # Geometry-aware fingerprint for the smart gating engine.
    part_fingerprint = _part_fingerprint(result, bodies, n_ingates)

    # Effective metal head from geometry + mass reduction + elbow losses.
    # Height is measured along the user-selected gravity direction, not hard-coded Z.
    metal_pts = np.argwhere(result.is_metal)
    if len(metal_pts) > 0:
        projections = metal_pts @ gravity_vec
        total_height_mm = float((projections.max() - projections.min()) * dx)
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

    # Elbow/head-loss estimate from the discrete gating tree.  This avoids the
    # 26-neighbor Dijkstra over the channel voxel mask that was hanging at 28%.
    flow_result = getattr(result, "flow_result", None)
    gating_nodes = getattr(flow_result, "gating_nodes", []) if flow_result else []
    elbow_count = _count_elbows_from_gating_nodes(gating_nodes, angle_threshold_deg=60.0)
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
        max_flow_path_mm=part_fingerprint["flow_path_mm"],
        wall_thickness_mm=part_fingerprint["t_mean_mm"],
        wall_thickness_min_mm=part_fingerprint["t_min_mm"],
        wall_thickness_max_mm=part_fingerprint["t_max_mm"],
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

    # Use the engine for fill time, then override areas/velocities with the smart
    # geometry- and alloy-aware design.
    fill_time_s = design.t_fill_s
    design_fill_time_s = fill_time_s
    n_ingates = max(n_ingates, design.n_gates)

    smart = _smart_gating_design(
        result=result,
        alloy=alloy,
        mold=mold,
        casting_params=casting_params,
        bodies=bodies,
        total_mass_kg=total_mass_kg,
        t_fill_s=fill_time_s,
        H_eff_m=H_eff_m,
        n_ingates=n_ingates,
        ingate_thickness_mm=ingate_thickness_mm,
        wall_thickness_mm=wall_thickness_mm,
        gravity_vec=gravity_vec,
        design=design,
        user_gate_velocity=user_gate_velocity,
        fingerprint=part_fingerprint,
    )
    As_m2 = smart["As_m2"]
    Ar_total_m2 = smart["Ar_total_m2"]
    Ag_total_m2 = smart["Ag_total_m2"]
    Ag_each_m2 = smart["Ag_each_m2"]
    Vc_ms = smart["v_choke_m_s"]
    Q_design_m3_s = smart["Q_design_m3_s"]
    v_sprue_design = smart["v_sprue_design"]
    v_runner_design = smart["v_runner_design"]
    v_gate_design = smart["v_gate_design"]
    d_sprue_mm = smart["d_sprue_mm"]
    d_ingate_each_mm = smart["d_ingate_each_mm"]
    final_ratio = smart["final_ratio"]
    recommended_system = smart["recommended_system"]
    ingate_Q_each = Q_design_m3_s / max(n_ingates, 1)

    # Target ranges derived from the alloy critical entrainment velocity.
    Vcrit = smart["Vcrit"]
    def _range_for(section: str, lo_factor: float, hi_factor: float):
        hi = _section_velocity_limit(recommended_system, alloy.key, section) * hi_factor
        return (hi * lo_factor, hi)

    if "basınçlı" in recommended_system and "yarı" not in recommended_system:
        sprue_v_range = _range_for("sprue", 0.15, 0.7)
        runner_v_range = _range_for("runner", 0.2, 0.9)
        gate_v_range = _range_for("gate", 0.3, 1.0)
    elif "basınçsız" in recommended_system:
        sprue_v_range = _range_for("sprue", 0.4, 1.0)
        runner_v_range = _range_for("runner", 0.2, 0.7)
        gate_v_range = _range_for("gate", 0.05, 0.6)
    else:
        sprue_v_range = _range_for("sprue", 0.2, 0.9)
        runner_v_range = _range_for("runner", 0.2, 0.9)
        gate_v_range = _range_for("gate", 0.2, 0.8)
    velocity_targets = {
        "sprue": sprue_v_range,
        "runner": runner_v_range,
        "gate": gate_v_range,
    }
    sprue_A_min, sprue_A_max = _target_area_range_cm2(Q_design_m3_s, *sprue_v_range)
    runner_A_min, runner_A_max = _target_area_range_cm2(Q_design_m3_s, *runner_v_range)
    gate_A_min, gate_A_max = _target_area_range_cm2(ingate_Q_each, *gate_v_range)

    # Primary SectionFlow objects from the engine design.
    # _compute_section_flow already computes v = Q/A, so we do not override it;
    # this keeps SPRUE_THROAT velocity correct when its area differs from base.
    d_runner_mm = 1000.0 * math.sqrt(4.0 * max(Ar_total_m2, 0.0) / math.pi)
    section_flows: Dict[str, SectionFlow] = {}
    section_specs = [
        ("SPRUE_BASE", As_m2 * 1e4, d_sprue_mm, sprue_v_range[0], sprue_v_range[1], sprue_A_min, sprue_A_max, Q_design_m3_s),
        ("SPRUE_THROAT", As_m2 * 1e4, d_sprue_mm, sprue_v_range[0], sprue_v_range[1], sprue_A_min, sprue_A_max, Q_design_m3_s),
        ("RUNNER", Ar_total_m2 * 1e4, d_runner_mm, runner_v_range[0], runner_v_range[1], runner_A_min, runner_A_max, Q_design_m3_s),
        ("INGATE", Ag_each_m2 * 1e4, d_ingate_each_mm, gate_v_range[0], gate_v_range[1], gate_A_min, gate_A_max, ingate_Q_each),
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

    # n_ingates stays as the actual/design count used by the smart gating design.

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

    # Classify the real system from the actual INGATE (meme) velocity only.
    v_meme_actual = actual_v.get("gate", 0.0)
    actual_ingate_area_cm2 = actual_area["gate"]
    if use_bodies:
        actual_n_ingates = max(int(real_areas.get("n_ingates", 1)), 1)
    elif has_ingate:
        _, actual_n_ingates = ndimage.label(ingate)
    else:
        actual_n_ingates = 1

    detected_system, detected_reason = _classify_from_ingate_velocity(
        v_meme_actual, alloy, wall_thickness_mm
    )
    recommended_system = smart["recommended_system"]
    recommended_reason = smart["reason"]

    # Recompute target ranges based on the detected system so warnings match
    # the physical behaviour, using the alloy critical velocity ceiling.
    system_for_ranges = detected_system or recommended_system
    def _range_for_detected(section: str, lo_factor: float, hi_factor: float):
        hi = _section_velocity_limit(system_for_ranges, alloy.key, section) * hi_factor
        return (hi * lo_factor, hi)

    if "basınçlı" in system_for_ranges and "yarı" not in system_for_ranges:
        sprue_v_range = _range_for_detected("sprue", 0.15, 0.7)
        runner_v_range = _range_for_detected("runner", 0.2, 0.9)
        gate_v_range = _range_for_detected("gate", 0.3, 1.0)
    elif "basınçsız" in system_for_ranges:
        sprue_v_range = _range_for_detected("sprue", 0.4, 1.0)
        runner_v_range = _range_for_detected("runner", 0.2, 0.7)
        gate_v_range = _range_for_detected("gate", 0.05, 0.6)
    else:
        sprue_v_range = _range_for_detected("sprue", 0.2, 0.9)
        runner_v_range = _range_for_detected("runner", 0.2, 0.9)
        gate_v_range = _range_for_detected("gate", 0.2, 0.8)
    velocity_targets = {
        "sprue": sprue_v_range,
        "runner": runner_v_range,
        "gate": gate_v_range,
    }

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

    # Human-readable gating recommendation: current state + smart proposal.
    gating_system_reason = (
        f"Şu anki durum: parçada {actual_n_ingates} meme var; toplam meme alanı {actual_ingate_area_cm2:.2f} cm², "
        f"gerçek meme hızı {v_meme_actual:.2f} m/s. Bu verilere göre mevcut sistem {detected_system}. "
        f"{detected_reason} "
        f"Akıllı öneri: {recommended_reason}"
    )

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
    # C is in dk/cm^2, M_stream in mm -> t_s in seconds
    t_s_stream = C * (M_stream / 10.0) ** 2 * 60.0
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
    elif bodies is not None:
        riser_volume_cm3 = sum(b.volume_cm3 for b in bodies if b.body_type == BodyType.RISER)
        total_riser_mass_kg = riser_volume_cm3 * alloy.density_g_cm3 / 1000.0
    else:
        total_riser_mass_kg = 0.0
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
        ingate_velocity_m_s = v_meme_actual if v_meme_actual > 0.0 else v_gate_design
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

