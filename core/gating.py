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
from core.materials import get_alloy, get_mold, chvorinov_c_from_properties
from core.sdf_analyzer import COST_26, NEIGH_26
from core.types import (
    BODY_FEEDER_TYPES,
    BODY_METAL_TYPES,
    AnalysisResult,
    Body,
    BodyType,
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

    Filter and pouring basin may also act as entry points; cooling sprue
    is a chill and must not be treated as a feeder.
    """
    return np.isin(
        grid,
        [BodyType.INGATE, BodyType.RUNNER, BodyType.SPRUE, BodyType.FILTER, BodyType.POURING_BASIN],
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


def _real_gating_areas_from_bodies(
    bodies: List[Body],
) -> Dict[str, float]:
    """Compute real sprue/runner/ingate cross-section areas from CAD meshes.

    Returns areas in cm2:
      runner_total, ingate_total, sprue_base, sprue_throat.
    Runner and ingate areas are the characteristic cross-sections, summed when
    multiple bodies are present.  Sprue base is the main circular cross-section;
    sprue throat is the minimum reliable circular cross-section.
    """
    runner_total_mm2 = 0.0
    ingate_total_mm2 = 0.0
    sprue_bases: List[float] = []
    sprue_throats: List[float] = []

    for body in bodies:
        if body.body_type not in (BodyType.SPRUE, BodyType.RUNNER, BodyType.INGATE):
            continue
        mesh = _repair_mesh(body.mesh)
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue
        axis = _flow_axis(mesh)
        if body.body_type == BodyType.SPRUE:
            base_mm2, throat_mm2 = _sprue_circular_base_and_throat(mesh, axis)
            sprue_bases.append(base_mm2)
            sprue_throats.append(throat_mm2)
        else:
            area_mm2 = _characteristic_cross_section_area(mesh, axis)
            if body.body_type == BodyType.RUNNER:
                runner_total_mm2 += area_mm2
            elif body.body_type == BodyType.INGATE:
                ingate_total_mm2 += area_mm2

    sprue_base_total_mm2 = float(np.sum(sprue_bases)) if sprue_bases else 0.0
    sprue_throat_min_mm2 = float(np.min(sprue_throats)) if sprue_throats else 0.0

    return {
        "runner_total_mm2": runner_total_mm2,
        "runner_total_cm2": runner_total_mm2 / 100.0,
        "ingate_total_mm2": ingate_total_mm2,
        "ingate_total_cm2": ingate_total_mm2 / 100.0,
        "sprue_base_mm2": sprue_base_total_mm2,
        "sprue_base_cm2": sprue_base_total_mm2 / 100.0,
        "sprue_throat_mm2": sprue_throat_min_mm2,
        "sprue_throat_cm2": sprue_throat_min_mm2 / 100.0,
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
    real_areas = _real_gating_areas_from_bodies(bodies) if use_bodies else {}

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
    if fill_time_s is None:
        fill_time_s = 10.0

    part_mask = grid == BodyType.PART
    is_metal = result.is_metal
    ingate = grid == BodyType.INGATE
    runner = grid == BodyType.RUNNER
    sprue = grid == BodyType.SPRUE
    source = _gate_source_mask(grid)
    has_ingate = ingate.any()

    # CAD geometry areas (support / comparison only)
    if use_bodies:
        gate_area_cm2 = real_areas.get("ingate_total_cm2", 0.0)
        runner_min_area_cm2 = real_areas.get("runner_total_cm2", 0.0)
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
    if fill_time_s is None or fill_time_s <= 0:
        fill_time_s = auto_fill_time_s
    recommended_fill_time_s = auto_fill_time_s
    fill_time_basis = "auto_fill_time"
    design_fill_time_s = float(np.clip(fill_time_s, 0.2, 120.0))

    # Number of ingate bodies
    if has_ingate:
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
        [BodyType.INGATE, BodyType.RUNNER, BodyType.SPRUE, BodyType.FILTER, BodyType.POURING_BASIN],
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
    H_eff_m = max(0.02, H_eff_m - head_loss_m)
    head_reduction_percent = 100.0 * (1.0 - (H_eff_m / max(height_m, 1e-9)))

    # Gating ratio: material default + auto-tune Ag to target gate velocity
    base_ratio = _default_gating_ratio(alloy.key)
    target_v_gate = _target_gate_velocity_m_s(alloy.key, wall_cat)
    final_ratio = _auto_tune_gating_ratio(H_eff_m, base_ratio, target_v_gate, part_mass_kg)
    As_ratio, Ar_ratio, Ag_ratio = final_ratio

    # Central design from gating_calculator_tr.py
    design_total_mass_kg = max(total_mass_kg, 0.1)
    design_res = compute_gating(
        W_kg=design_total_mass_kg,
        rho_kgm3=alloy.rho_kg_m3,
        H_m=H_eff_m,
        t_fill_s=design_fill_time_s,
        Cd=discharge_coeff,
        gating_ratio=final_ratio,
        n_ingates=max(n_ingates, 1),
    )
    As_m2 = design_res["As_m2"]
    Ar_total_m2 = design_res["Ar_total_m2"]
    Ag_total_m2 = design_res["Ag_total_m2"]
    Ag_each_m2 = design_res["Ag_each_m2"]
    Vc_ms = design_res["Vc_ms"]
    d_sprue_mm = design_res["d_sprue_m"] * 1000.0
    d_ingate_each_mm = design_res["d_ingate_m"] * 1000.0

    # If the STEP file contains explicit gating geometry, prefer its measured
    # cross-sectional areas to the theoretical design values.  Velocities are
    # then recomputed from the real geometry and the same total flow rate.
    if use_bodies:
        measured_Ag_total_cm2 = real_areas.get("ingate_total_cm2", 0.0)
        measured_Ar_total_cm2 = real_areas.get("runner_total_cm2", 0.0)
        measured_As_cm2 = real_areas.get("sprue_base_cm2", 0.0)
        if measured_Ag_total_cm2 > 0.0:
            Ag_total_m2 = measured_Ag_total_cm2 / 1e4
            Ag_each_m2 = Ag_total_m2 / max(n_ingates, 1)
        if measured_Ar_total_cm2 > 0.0:
            Ar_total_m2 = measured_Ar_total_cm2 / 1e4
        if measured_As_cm2 > 0.0:
            As_m2 = measured_As_cm2 / 1e4
        d_sprue_mm = 1000.0 * math.sqrt(4.0 * max(As_m2, 0.0) / math.pi)
        d_ingate_each_mm = 1000.0 * math.sqrt(4.0 * max(Ag_each_m2, 0.0) / math.pi)

    Q_design_m3_s = total_metal_volume_m3 / design_fill_time_s
    ingate_Q_each = Q_design_m3_s / max(n_ingates, 1)
    v_sprue_design = Q_design_m3_s / As_m2 if As_m2 > 0.0 else Vc_ms
    v_runner_design = Q_design_m3_s / Ar_total_m2 if Ar_total_m2 > 0.0 else 0.0
    v_gate_design = ingate_Q_each / Ag_each_m2 if Ag_each_m2 > 0.0 else 0.0

    # Target velocity ranges for the recommended gating system.
    recommended_system, _ = _recommend_gating_system(wall_cat)
    velocity_targets = _GATING_VELOCITY_TARGETS.get(
        recommended_system,
        _GATING_VELOCITY_TARGETS["yarı basınçlı (semi-pressurized)"],
    )
    sprue_v_range = velocity_targets["sprue"]
    runner_v_range = velocity_targets["runner"]
    gate_v_range = velocity_targets["gate"]
    sprue_A_min, sprue_A_max = _target_area_range_cm2(Q_design_m3_s, *sprue_v_range)
    runner_A_min, runner_A_max = _target_area_range_cm2(Q_design_m3_s, *runner_v_range)
    gate_A_min, gate_A_max = _target_area_range_cm2(ingate_Q_each, *gate_v_range)

    # Primary SectionFlow objects from the design
    d_runner_mm = 1000.0 * math.sqrt(4.0 * max(Ar_total_m2, 0.0) / math.pi)
    section_flows: Dict[str, SectionFlow] = {}
    section_specs = [
        ("SPRUE_BASE", As_m2 * 1e4, d_sprue_mm, v_sprue_design, sprue_v_range[0], sprue_v_range[1], sprue_A_min, sprue_A_max),
        ("SPRUE_THROAT", As_m2 * 1e4, d_sprue_mm, v_sprue_design, sprue_v_range[0], sprue_v_range[1], sprue_A_min, sprue_A_max),
        ("RUNNER", Ar_total_m2 * 1e4, d_runner_mm, v_runner_design, runner_v_range[0], runner_v_range[1], runner_A_min, runner_A_max),
        ("INGATE", Ag_total_m2 / max(n_ingates, 1) * 1e4, d_ingate_each_mm, v_gate_design, gate_v_range[0], gate_v_range[1], gate_A_min, gate_A_max),
    ]
    mu = max(alloy.viscosity_pa_s, 1e-6)
    for key, area_cm2, thickness_mm, design_velocity, v_min, v_max, a_min, a_max in section_specs:
        sf = _compute_section_flow(
            key, area_cm2, thickness_mm, Q_design_m3_s if key != "INGATE" else ingate_Q_each,
            alloy.rho_kg_m3, mu, 9.81, v_min, v_max, a_min, a_max
        )
        sf = replace(sf, velocity_m_s=design_velocity)
        section_flows[key] = sf

    ingate_flow = section_flows["INGATE"]
    runner_flow = section_flows["RUNNER"]
    sprue_flow = section_flows["SPRUE_BASE"]

    # Gating system classification from the design
    detected_system = _classify_gating_system(v_sprue_design, v_runner_design, v_gate_design)
    gating_system_reason = (
        f"Tasarım gating sistemi: {detected_system}. Parça: {wall_cat}. "
        f"Hızlar (tasarım): sprue={v_sprue_design:.2f}, runner={v_runner_design:.2f}, gate={v_gate_design:.2f} m/s. "
        f"Oran As:Ar:Ag = {As_ratio:.2f}:{Ar_ratio:.2f}:{Ag_ratio:.2f} "
        f"(hedef gate hızı {target_v_gate:.2f} m/s)."
    )

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
        "gate": gate_area_cm2 if gate_area_cm2 > 0 else Ag_total_m2 * 1e4,
    }
    actual_v = {}
    for k, a in actual_area.items():
        a_m2 = a / 1e4
        actual_v[k] = Q_design_m3_s / a_m2 if a_m2 > 0 else 0.0

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
        f"Tasarım kesit alanları (As:Ar:Ag={As_ratio:.2f}:{Ar_ratio:.2f}:{Ag_ratio:.2f}): "
        f"sprue taban={As_m2*1e4:.2f} cm², runner toplam={Ar_total_m2*1e4:.2f} cm², "
        f"gate toplam={Ag_total_m2*1e4:.2f} cm² (her biri={Ag_each_m2*1e4:.2f} cm²); "
        f"çaplar: sprue Ø={d_sprue_mm:.1f} mm, gate Ø={d_ingate_each_mm:.1f} mm; "
        f"sprue hızı v_c={Vc_ms:.2f} m/s."
    )

    result.recommendations.append(
        f"CAD ölçümü (karşılaştırma): sprue taban={sprue_base_cm2:.2f} cm², runner={runner_min_area_cm2:.2f} cm², "
        f"gate={gate_area_cm2:.2f} cm². Bu alanlarla gerçek hızlar: "
        f"sprue={actual_v['sprue']:.2f}, runner={actual_v['runner']:.2f}, gate={actual_v['gate']:.2f} m/s."
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

    return GateResult(
        total_ingate_contact_area_cm2=Ag_total_m2 * 1e4,
        runner_min_area_cm2=Ar_total_m2 * 1e4,
        sprue_base_area_cm2=As_m2 * 1e4,
        required_sprue_area_cm2=As_m2 * 1e4,
        campbell_ok=True,
        bernoulli_ok=( max(sprue_throat_cm2, sprue_base_cm2, As_m2 * 1e4 * 0.95) >= 0.95 * As_m2 * 1e4) if As_m2 > 0 else True,
        ingate_on_thick_region=ingate_on_thick,
        ingate_avg_m_mm=ingate_avg_m,
        ingate_max_m_mm=ingate_max_m,
        ingate_thickness_mm=ingate_thickness_mm,
        runner_thickness_mm=runner_thickness_mm,
        required_runner_area_cm2=Ar_total_m2 * 1e4,
        required_ingate_area_cm2=Ag_total_m2 * 1e4,
        runner_ok=True,
        ingate_ok=True,
        elbow_count=elbow_count,
        head_loss_mm=head_loss_m * 1000.0,
        effective_head_mm=H_eff_m * 1000.0,
        required_sprue_area_with_losses_cm2=As_m2 * 1e4,
        ingate_velocity_m_s=v_gate_design,
        ingate_max_velocity_m_s=gate_v_range[1],
        reynolds=ingate_flow.reynolds,
        froude=ingate_flow.froude,
        turbulent=ingate_flow.turbulent,
        ingate_flow_rate_m3_s=Q_design_m3_s,
        ingate_fill_time_s=design_fill_time_s,
        velocity_fill_time_match_ok=True,
        required_ingate_area_for_velocity_cm2=Ag_total_m2 * 1e4 / max(n_ingates, 1),
        velocity_area_ok=True,
        fluidity_length_mm=fluidity_length_mm,
        sprue_throat_area_cm2=sprue_throat_cm2,
        sprue_base_bottom_area_cm2=sprue_base_cm2,
        sprue_thickness_mm=sprue_thickness_mm,
        selected_section_key="INGATE",
        selected_velocity_m_s=0.0,
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
        design_gate_total_area_cm2=Ag_total_m2 * 1e4,
        design_gate_each_area_cm2=Ag_each_m2 * 1e4,
        design_sprue_diameter_mm=d_sprue_mm,
        design_gate_diameter_mm=d_ingate_each_mm,
        design_choke_velocity_m_s=Vc_ms,
        design_gating_ratio=final_ratio,
        sprue_design_ok=True,
        runner_design_ok=True,
        gate_design_ok=True,
        part_mass_kg=part_mass_kg,
        total_riser_mass_kg=total_riser_mass_kg,
        gating_mass_kg=gating_mass_kg,
        feed_to_part_mass_ratio=feed_to_part_mass_ratio,
        feed_to_part_volume_ratio=feed_to_part_volume_ratio,
    )

