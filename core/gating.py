"""Ingate / runner / sprue geometric gating calculations - JoseCast v8.0."""

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import math

import numpy as np
import trimesh
from scipy import ndimage
from scipy.sparse import coo_matrix, csgraph
from scipy.spatial import ConvexHull

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
RHO_REF_KG_M3 = 7000.0


def _campbell_base_fill_time(m_kg: float) -> Tuple[float, str]:
    """Campbell Table 4.1 piecewise log interpolation for base fill time [s]."""
    m = max(m_kg, 1e-6)
    points = [
        (20.0, 2.0),
        (100.0, 6.0),
        (250.0, 14.0),
        (500.0, 30.0),
        (2000.0, 60.0),
        (3000.0, 130.0),
    ]
    if m <= points[0][0]:
        return points[0][1], f"m <= {points[0][0]} kg"
    for i in range(len(points) - 1):
        m1, t1 = points[i]
        m2, t2 = points[i + 1]
        if m1 < m <= m2:
            t = t1 + (t2 - t1) * (np.log10(m / m1) / np.log10(m2 / m1))
            return float(t), f"{m1}-{m2} kg (log interp)"
    return points[-1][1], f"m > {points[-1][0]} kg"


def _recommended_fill_time(
    m_kg: float, rho_kg_m3: float, thickness_mm: float, superheat_c: float
) -> Tuple[float, str]:
    """Campbell-corrected fill time: t = t_base * f_rho * f_thick * f_temp."""
    t_base, detail = _campbell_base_fill_time(m_kg)
    f_rho = (RHO_REF_KG_M3 / max(rho_kg_m3, 1e-6)) ** 0.35
    f_thick = (max(thickness_mm, 1.0) / 20.0) ** 0.2
    f_temp = (max(superheat_c, 10.0) / 100.0) ** 0.4
    return float(t_base * f_rho * f_thick * f_temp), detail


def _head_reduction_fraction(W_part_kg: float) -> float:
    """Mass-dependent head reduction for effective metal head [0..0.75]."""
    m = float(np.clip(W_part_kg, 0.0, 3000.0))
    points = [
        (0.0, 0.00),
        (100.0, 0.40),
        (250.0, 0.50),
        (500.0, 0.60),
        (1000.0, 0.70),
        (3000.0, 0.75),
    ]
    for i in range(len(points) - 1):
        m0, r0 = points[i]
        m1, r1 = points[i + 1]
        if m0 <= m <= m1:
            if m1 == m0:
                return float(r1)
            t = (m - m0) / (m1 - m0)
            return float(r0 + t * (r1 - r0))
    return float(points[-1][1])


def _effective_head_m(H_m: float, W_part_kg: float) -> float:
    """Apply mass-dependent reduction to the available metal head."""
    if H_m <= 0.0:
        return H_m
    frac = np.clip(_head_reduction_fraction(W_part_kg), 0.0, 0.9)
    return H_m * (1.0 - frac)


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
        area = 0.0
        perim = 0.0
        try:
            p2d, _ = section.to_2D()
            area = float(getattr(p2d, "area", 0.0))
            perim = float(getattr(p2d, "length", 0.0))
        except Exception:
            try:
                v = section.vertices[:, :2]
                if v.shape[0] >= 3:
                    hull = ConvexHull(v)
                    area = float(hull.volume)
                    perim = 0.0
            except Exception:
                area = 0.0
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

    valid = (areas > 0.05 * max_area) & (circs > 0.85) & (circ_areas > 0.0)
    if not valid.any():
        base = float(np.median(areas[1:-1])) if len(areas) > 2 else float(np.median(areas))
        throat = base
        return base, throat

    base = float(circ_areas[valid].max())
    interior = np.ones_like(areas, dtype=bool)
    interior[0] = interior[-1] = False
    if not (interior & valid).any():
        interior = valid
    throat = float(circ_areas[interior & valid].min())
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
    """
    Theoretical gating areas from total poured mass / head / fill time.
    Returns sprue base (choke), runner, gate total/each areas in cm2 and diameters in mm.
    """
    g = 9.81
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

    Vc = np.sqrt(2.0 * g * H_eff_m)  # choke velocity at sprue base
    As_m2 = W_total_kg / (rho_kg_m3 * Cd * t_fill_s * Vc)
    As_ratio, Ar_ratio, Ag_ratio = gating_ratio
    Ar_total_m2 = As_m2 * (Ar_ratio / max(As_ratio, 1e-9))
    Ag_total_m2 = As_m2 * (Ag_ratio / max(As_ratio, 1e-9))
    Ag_each_m2 = Ag_total_m2 / max(n_ingates, 1)

    return {
        "As_cm2": As_m2 * 1e4,
        "Ar_total_cm2": Ar_total_m2 * 1e4,
        "Ag_total_cm2": Ag_total_m2 * 1e4,
        "Ag_each_cm2": Ag_each_m2 * 1e4,
        "Vc_ms": float(Vc),
        "d_sprue_mm": _area_to_diameter_mm(As_m2 * 1e4),
        "d_ingate_each_mm": _area_to_diameter_mm(Ag_each_m2 * 1e4),
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


def auto_fill_time(mass_kg: float, alloy_key: str = "", alloy_name: str = "") -> float:
    """Practical fill-time estimate from gating_calculator_tr.py (not Campbell).

    Defaults: small <=5 kg, medium <=20 kg, large >20 kg.
    """
    if mass_kg <= 0:
        return 5.0
    if mass_kg <= 5.0:
        size = "small"
    elif mass_kg <= 20.0:
        size = "medium"
    else:
        size = "large"

    key = (alloy_key or "").lower()
    name = (alloy_name or "").lower()
    combined = key + " " + name

    if "çelik" in combined or "steel" in combined or "42" in combined:
        return {"small": 3.0, "medium": 5.0, "large": 8.0}.get(size, 5.0)
    if "gri" in combined or "sfero" in combined or "ggg" in combined or "nodular" in combined:
        return {"small": 2.5, "medium": 4.0, "large": 6.0}.get(size, 4.0)
    if "al" in combined or "alum" in combined or "alsi" in combined:
        return {"small": 3.0, "medium": 5.0, "large": 8.0}.get(size, 5.0)
    if "bronz" in combined or "bronze" in combined:
        return {"small": 2.0, "medium": 3.5, "large": 5.0}.get(size, 3.5)

    return {"small": 3.5, "medium": 5.5, "large": 8.5}.get(size, 5.5)


def compute_modulus_and_riser(
    W_part_kg: float,
    rho_kg_m3: float,
    A_cast_m2: float,
    k_mod: float = 1.2,
) -> Dict[str, float]:
    """Exact riser modulus / cylinder sizing from gating_calculator_tr.py.

    Assumes a cylindrical riser with H = D and top+bottom cooling.
    Searches diameter from 0.05 m to 1.0 m in 0.005 m steps.
    """
    if A_cast_m2 <= 0.0 or W_part_kg <= 0.0 or rho_kg_m3 <= 0.0:
        return {
            "V_cast_m3": 0.0,
            "M_cast_m": 0.0,
            "M_riser_req_m": 0.0,
            "riser_D_m": 0.0,
            "riser_H_m": 0.0,
            "riser_M_m": 0.0,
        }

    V_cast_m3 = W_part_kg / rho_kg_m3
    M_cast_m = V_cast_m3 / A_cast_m2
    M_riser_req_m = k_mod * M_cast_m

    D = 0.05
    D_max = 1.0
    step = 0.005
    best_D = None
    best_H = None
    best_M = None

    while D <= D_max:
        H = D
        V_r = math.pi * (D / 2.0) ** 2 * H
        A_r = math.pi * D * H + 2.0 * math.pi * (D / 2.0) ** 2
        M_r = V_r / A_r
        if M_r >= M_riser_req_m:
            best_D = D
            best_H = H
            best_M = M_r
            break
        D += step

    if best_D is None:
        best_D = D_max
        best_H = D_max
        V_r = math.pi * (best_D / 2.0) ** 2 * best_H
        A_r = math.pi * best_D * best_H + 2.0 * math.pi * (best_D / 2.0) ** 2
        best_M = V_r / A_r

    return {
        "V_cast_m3": V_cast_m3,
        "M_cast_m": M_cast_m,
        "M_riser_req_m": M_riser_req_m,
        "riser_D_m": best_D,
        "riser_H_m": best_H,
        "riser_M_m": best_M,
    }


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
    max_velocity_m_s: float,
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
        # Ingate also checked against Campbell max velocity; other sections rely on Re.
        if section_key == "INGATE":
            turbulent = (reynolds > 20000.0) or (velocity > max_velocity_m_s)
        else:
            turbulent = reynolds > 20000.0
    return SectionFlow(
        velocity_m_s=velocity,
        area_cm2=area_cm2,
        thickness_mm=thickness_mm,
        reynolds=reynolds,
        froude=froude,
        turbulent=turbulent,
        max_velocity_m_s=max_velocity_m_s,
    )


def analyze_gating(
    result: AnalysisResult,
    fill_time_s: Optional[float] = None,
    discharge_coeff: float = 0.8,
    casting_params=None,
    bodies: Optional[List[Body]] = None,
) -> Optional[GateResult]:
    """Compute gate/sprue/runner checks including Bernoulli elbow losses and all section velocities/Re/Fr.

    If ``bodies`` is supplied, real CAD cross-sectional areas and volumes are used
    instead of coarse voxel approximations.
    """
    from core.types import CastingParameters

    grid = result.grid
    sdf = result.sdf
    dx = result.dx_mm
    alloy = get_alloy(result.alloy_key)
    mold = get_mold(result.mold_key)

    use_bodies = bodies is not None and len(bodies) > 0
    real_areas = _real_gating_areas_from_bodies(bodies) if use_bodies else {}
    velocity_section_key = "INGATE"
    if casting_params is not None and isinstance(casting_params, CastingParameters):
        fill_time_s = casting_params.t_fill_s
        velocity_section_key = getattr(casting_params, "velocity_section_key", "INGATE") or "INGATE"
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

    # v8.8: use real CAD cross-sections and volumes when the original bodies are supplied.
    if use_bodies:
        # v9.0: characteristic (contact-equivalent) cross-sections from body meshes.
        gate_area_cm2 = real_areas.get("ingate_total_cm2", 0.0)
        runner_min_area_cm2 = real_areas.get("runner_total_cm2", 0.0)
        sprue_base_cm2 = real_areas.get("sprue_base_cm2", 0.0)
        sprue_base_bottom_cm2 = sprue_base_cm2
        sprue_throat_cm2 = real_areas.get("sprue_throat_cm2", 0.0)

        # If no separate ingate body exists, the runner/sprue-part contact area acts
        # as the gate (common in simple systems like Knuckle).
        contact_area_mm2, _ = ingate_contact_area_and_mask(grid, dx)
        gate_contact_area_cm2 = (
            gate_area_cm2 if gate_area_cm2 > 0.0 else contact_area_mm2 / 100.0
        )
        if gate_area_cm2 <= 0.0:
            gate_area_cm2 = gate_contact_area_cm2

        part_volume_cm3, total_metal_volume_cm3 = _volumes_from_bodies(bodies)
    else:
        # Fallback: coarse voxel-based areas.
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

    # Wall thickness / characteristic diameters for Re/Fr (keep voxel estimates where useful)
    runner_thickness_mm = _mean_thickness(runner, dx)
    sprue_thickness_mm = _mean_thickness(sprue, dx)
    ingate_thickness_mm = _mean_thickness(ingate if has_ingate else source, dx)

    part_weight_g = part_volume_cm3 * alloy.density_g_cm3
    part_mass_kg = part_weight_g / 1000.0

    total_weight_g = total_metal_volume_cm3 * alloy.density_g_cm3
    total_metal_volume_m3 = total_metal_volume_cm3 / 1e6
    total_mass_kg = total_weight_g / 1000.0
    pour_yield = part_volume_cm3 / total_metal_volume_cm3 if total_metal_volume_cm3 > 0 else 1.0

    # v8.6: practical auto fill time from gating_calculator_tr.py + Campbell info
    superheat = max(alloy.t_pour_c - alloy.t_liquidus_c, 0.0)
    wall_thickness_mm = getattr(result, "wall_thickness_mm", 0.0) or 20.0
    campbell_fill_time_s, campbell_fill_time_basis = _recommended_fill_time(
        part_mass_kg, alloy.rho_kg_m3, wall_thickness_mm, superheat
    )
    auto_fill_time_s = auto_fill_time(part_mass_kg, alloy.key, alloy.name)
    if fill_time_s is None or fill_time_s <= 0:
        fill_time_s = auto_fill_time_s
    # Practical recommendation used as the primary recommendation.
    recommended_fill_time_s = auto_fill_time_s
    fill_time_basis = "auto_fill_time"

    # Number of ingate bodies (for design area per gate)
    if has_ingate:
        _, n_ingates = ndimage.label(ingate)
    else:
        n_ingates = 1

    # Volumetric flow rate = total poured volume / fill time
    Q_m3_s = 0.0
    if fill_time_s > 0 and total_metal_volume_m3 > 0:
        Q_m3_s = total_metal_volume_m3 / fill_time_s

    # Section area map for velocity calculations
    section_area_cm2 = {
        "INGATE": gate_area_cm2,
        "RUNNER": runner_min_area_cm2,
        "SPRUE_THROAT": sprue_throat_cm2,
        "SPRUE_BASE": sprue_base_bottom_cm2,
    }
    # If the requested section has no area (e.g. no ingate body), fall back to RUNNER.
    if velocity_section_key == "INGATE" and section_area_cm2["INGATE"] <= 0.0:
        velocity_section_key = "RUNNER"
    selected_area_cm2 = section_area_cm2.get(velocity_section_key, gate_area_cm2)
    selected_area_m2 = selected_area_cm2 / 1e4

    # v8.5: user-specified inlet velocity for the selected section
    user_v = 0.0
    if casting_params is not None and isinstance(casting_params, CastingParameters):
        user_v = float(getattr(casting_params, "ingate_velocity_m_s", 0.0))

    t_fill_computed_s = fill_time_s
    velocity_fill_time_match_ok = True
    required_selected_area_for_velocity_m2 = 0.0
    velocity_area_ok = True

    if user_v > 0 and selected_area_m2 > 0 and total_metal_volume_m3 > 0:
        Q_m3_s = user_v * selected_area_m2
        t_fill_computed_s = total_metal_volume_m3 / Q_m3_s
        if fill_time_s > 0:
            tolerance = 0.15 * fill_time_s
            velocity_fill_time_match_ok = abs(t_fill_computed_s - fill_time_s) <= tolerance
            required_Q_m3_s = total_metal_volume_m3 / fill_time_s
            required_selected_area_for_velocity_m2 = required_Q_m3_s / user_v
            velocity_area_ok = selected_area_m2 >= required_selected_area_for_velocity_m2 * 0.95
    elif fill_time_s is not None and fill_time_s > 0 and total_metal_volume_m3 > 0:
        # Auto: total poured volume / fill time (correct Q through every section)
        Q_m3_s = total_metal_volume_m3 / fill_time_s
        t_fill_computed_s = fill_time_s

    # Compute per-section velocities and Re/Fr
    g_m_s2 = 9.81
    rho = alloy.rho_kg_m3
    mu = max(alloy.viscosity_pa_s, 1e-6)
    max_ingate_velocity = 0.5 if alloy.key in ("AlSi7",) else 1.0

    section_flows: Dict[str, SectionFlow] = {}
    for key, area_cm2 in section_area_cm2.items():
        thickness = {
            "INGATE": ingate_thickness_mm,
            "RUNNER": runner_thickness_mm,
            "SPRUE_THROAT": sprue_thickness_mm,
            "SPRUE_BASE": sprue_thickness_mm,
        }.get(key, ingate_thickness_mm)
        max_v = max_ingate_velocity if key == "INGATE" else 2.0
        section_flows[key] = _compute_section_flow(
            key, area_cm2, thickness, Q_m3_s, rho, mu, g_m_s2, max_v
        )

    ingate_flow = section_flows.get("INGATE", SectionFlow())
    runner_flow = section_flows.get("RUNNER", SectionFlow())
    sprue_flow = section_flows.get("SPRUE_BASE", SectionFlow())
    ingate_velocity_m_s = ingate_flow.velocity_m_s
    ingate_flow_rate_m3_s = Q_m3_s
    ingate_fill_time_s = t_fill_computed_s
    reynolds = ingate_flow.reynolds
    froude = ingate_flow.froude
    turbulent = ingate_flow.turbulent

    # v8.8: classify by the sprue exit/base area, not the physical throat, because the
    # hydraulic ordering As > Ar > Ag is determined by the sprue exit cross-section.
    effective_gate_section = "INGATE" if has_ingate else "RUNNER (meme yok)"
    v_sprue = sprue_flow.velocity_m_s
    v_runner = runner_flow.velocity_m_s
    v_gate = ingate_flow.velocity_m_s
    detected_system = _classify_gating_system(v_sprue, v_runner, v_gate)
    wall_thickness = getattr(result, "wall_thickness_mm", 0.0)
    wall_cat = _wall_thickness_category(wall_thickness) if wall_thickness > 0 else "orta cidarlı"

    # Attach target ranges to the detected (or fallback) system for reference only.
    system_for_targets = detected_system if detected_system in _GATING_VELOCITY_TARGETS else "basınçsız (unpressurized)"

    def _set_targets(flow: SectionFlow, key: str):
        section_key_map = {
            "INGATE": "gate",
            "RUNNER": "runner",
            "SPRUE_THROAT": "sprue",
            "SPRUE_BASE": "sprue",
        }
        vel_key = section_key_map.get(key)
        targets = _GATING_VELOCITY_TARGETS.get(system_for_targets, {})
        if vel_key and vel_key in targets and Q_m3_s > 0:
            v_lo, v_hi = targets[vel_key]
            flow.target_v_min_m_s = v_lo
            flow.target_v_max_m_s = v_hi
            flow.target_area_min_cm2, flow.target_area_max_cm2 = _target_area_range_cm2(
                Q_m3_s, v_lo, v_hi
            )

    for key, sf in section_flows.items():
        _set_targets(sf, key)

    gating_system_reason = (
        f"Tespit edilen gating sistemi: {detected_system}. Parça: {wall_cat} "
        f"(baskın duvar t≈{wall_thickness:.1f} mm). "
        f"Hedef hız aralıkları ({system_for_targets}): "
        f"sprue={_GATING_VELOCITY_TARGETS[system_for_targets]['sprue'][0]:.1f}-{_GATING_VELOCITY_TARGETS[system_for_targets]['sprue'][1]:.1f}, "
        f"runner={_GATING_VELOCITY_TARGETS[system_for_targets]['runner'][0]:.1f}-{_GATING_VELOCITY_TARGETS[system_for_targets]['runner'][1]:.1f}, "
        f"gate={_GATING_VELOCITY_TARGETS[system_for_targets]['gate'][0]:.1f}-{_GATING_VELOCITY_TARGETS[system_for_targets]['gate'][1]:.1f} m/s."
    )

    # Sprue height H in m; fallback to total metal height
    metal_pts = np.argwhere(result.is_metal)
    if len(metal_pts) > 0:
        height_mm = float((metal_pts[:, 2].max() - metal_pts[:, 2].min()) * dx)
    else:
        height_mm = 0.0
    height_m = height_mm / 1000.0

    # v8.5: effective metal head with mass-dependent reduction (gating_calculator_tr.py)
    H_eff_m = _effective_head_m(height_m, part_mass_kg)
    head_reduction_percent = 100.0 * (1.0 - (H_eff_m / max(height_m, 1e-9)))
    H_eff_cm = H_eff_m * 100.0
    g_cgs = 981.0  # cm/s^2

    if H_eff_cm > 0 and fill_time_s > 0:
        velocity = np.sqrt(2.0 * g_cgs * H_eff_cm)
        # Use total poured weight (gating+riser+part) for choke area
        as_req_cm2 = total_weight_g / (
            alloy.density_g_cm3 * discharge_coeff * fill_time_s * velocity
        )
    else:
        as_req_cm2 = 0.0
        velocity = 0.0

    # Bernoulli with elbow losses along the gating channel
    channel_mask = np.isin(
        grid,
        [BodyType.INGATE, BodyType.RUNNER, BodyType.SPRUE, BodyType.FILTER, BodyType.POURING_BASIN],
    )
    sprue_mask = sprue & channel_mask
    effective_head_cm = H_eff_cm
    elbow_count = 0
    head_loss_cm = 0.0
    required_sprue_area_with_losses_cm2 = as_req_cm2
    if channel_mask.any() and sprue_mask.any():
        # Count elbows from the actual ingate(s) or runner exit if there is no ingate.
        source_vox = np.argwhere(ingate) if has_ingate else np.argwhere(runner & channel_mask)
        if len(source_vox) > 0:
            dist_to_sprue = _distance_to_sprue_26(channel_mask, sprue_mask, dx)
            sample = source_vox[
                np.linspace(0, len(source_vox) - 1, min(20, len(source_vox))).astype(int)
            ]
            counts = []
            for v in sample:
                counts.append(_count_elbows_along_path(dist_to_sprue, channel_mask, tuple(v)))
            elbow_count = int(round(np.median(counts))) if counts else 0
            if velocity > 0:
                v_loss_m_s = velocity / 100.0
                h_loss_per_elbow_m = alloy.elbow_loss_k * (v_loss_m_s ** 2) / (2.0 * 9.81)
                head_loss_cm = h_loss_per_elbow_m * 100.0 * elbow_count
                effective_head_cm = max(0.0, H_eff_cm - head_loss_cm)
                if effective_head_cm > 0 and fill_time_s > 0:
                    v_eff = np.sqrt(2.0 * g_cgs * effective_head_cm)
                    required_sprue_area_with_losses_cm2 = total_weight_g / (
                        alloy.density_g_cm3 * discharge_coeff * fill_time_s * v_eff
                    )
                else:
                    required_sprue_area_with_losses_cm2 = float("inf")

    # Bernoulli: controlling section is the sprue throat (minimum area)
    final_sprue_required_cm2 = max(as_req_cm2, required_sprue_area_with_losses_cm2)
    bernoulli_ok = sprue_throat_cm2 >= final_sprue_required_cm2

    # v8.5: theoretical gating areas from mass/head/fill time for cross-check
    design_ratio = _default_gating_ratio(alloy.key)
    design_areas = _gating_area_design(
        total_mass_kg,
        alloy.rho_kg_m3,
        H_eff_m,
        fill_time_s,
        discharge_coeff,
        design_ratio,
        n_ingates,
    )

    # v8.5: compare actual areas to theoretical mass/head/fill-time design
    def _design_ok(actual: float, design: float, tol: float = 0.30) -> bool:
        if design <= 0.0 or actual <= 0.0:
            return True
        return abs(actual - design) / design <= tol

    sprue_design_ok = _design_ok(sprue_base_bottom_cm2, float(design_areas["As_cm2"]))
    runner_design_ok = _design_ok(runner_min_area_cm2, float(design_areas["Ar_total_cm2"]))
    gate_design_ok = _design_ok(gate_contact_area_cm2, float(design_areas["Ag_total_cm2"]))

    # v8.4: area checks against target ranges from the recommended gating system.
    # The target area range for a section is A = Q / v, using the system's v range.
    gate_flow = section_flows.get("INGATE", SectionFlow())
    runner_flow = section_flows.get("RUNNER", SectionFlow())
    sprue_flow = section_flows.get("SPRUE_THROAT", SectionFlow())

    def _area_mid(sf: SectionFlow) -> float:
        if sf.target_area_min_cm2 > 0 and sf.target_area_max_cm2 > 0:
            return (sf.target_area_min_cm2 + sf.target_area_max_cm2) / 2.0
        return 0.0

    def _in_target_range(sf: SectionFlow, area_cm2: float) -> bool:
        if sf.target_area_min_cm2 <= 0 or sf.target_area_max_cm2 <= 0:
            return True
        lo = sf.target_area_min_cm2 * 0.95
        hi = sf.target_area_max_cm2 * 1.05
        return lo <= area_cm2 <= hi

    required_ingate_area_cm2 = _area_mid(gate_flow)
    required_runner_area_cm2 = _area_mid(runner_flow)
    required_sprue_area_cm2 = max(final_sprue_required_cm2, _area_mid(sprue_flow))

    # Campbell-style ratio check: keep a relaxed gate/runner area ratio, no forced warning.
    campbell_ok = True
    if runner_min_area_cm2 > 0 and gate_area_cm2 > 0:
        actual_ratio = gate_area_cm2 / runner_min_area_cm2
        campbell_ok = 0.3 <= actual_ratio <= 5.0
    else:
        campbell_ok = False

    runner_ok = True

    # Ingate location check
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

    ingate_ok = True

    # Fluidity length uses the ingate velocity
    t_stream = max(ingate_thickness_mm, runner_thickness_mm, 2.0 * result.dominant_m_mm, 2.0)
    M_stream = t_stream / 2.0
    C = chvorinov_c_from_properties(alloy, mold)
    t_s_stream = C * M_stream ** 2
    superheat = max(alloy.t_pour_c - alloy.t_liquidus_c, 0.0)
    l_eff = alloy.latent_heat_j_kg + alloy.cp_j_kgk * superheat
    superheat_ratio = max(alloy.cp_j_kgk * superheat / l_eff, 0.1) if l_eff > 0 else 0.1
    t_superheat = t_s_stream * superheat_ratio
    v_metal_m_s = ingate_velocity_m_s
    if v_metal_m_s <= 0 and height_mm > 0:
        v_metal_m_s = np.sqrt(2.0 * 9.81 * (height_mm / 1000.0))
    fluidity_length_mm = v_metal_m_s * t_superheat * 1000.0

    max_dim_mm = float(result.bbox_size_mm.max())
    result.recommendations = [
        r for r in result.recommendations if not r.startswith("Sıvı akışkanlık")
    ]
    if max_dim_mm > fluidity_length_mm:
        result.recommendations.append(
            f"Sıvı akışkanlık uzunluğu Lf = {fluidity_length_mm:.1f} mm, parça boyutu {max_dim_mm:.1f} mm. "
            f"Soğuk birleşme (cold shut) riski - döküm sıcaklığını artırın, kalıp sıcaklığını yükseltmeyin veya giriş hızını artırın."
        )
    else:
        result.recommendations.append(
            f"Akışkanlık uzunluğu Lf = {fluidity_length_mm:.1f} mm, parça boyutu {max_dim_mm:.1f} mm -> yeterli."
        )

    # Build a compact summary of all section velocities for recommendations
    velocity_summary = " | ".join(
        f"{k}: {sf.velocity_m_s:.2f}m/s (Re={sf.reynolds:.0f}, Fr={sf.froude:.2f})"
        for k, sf in section_flows.items()
        if sf.area_cm2 > 0
    )
    if velocity_summary:
        result.recommendations.append(f"Kesit hızları -> {velocity_summary}")

    result.recommendations.append(gating_system_reason)

    # v8.6: fill-time info (practical auto + Campbell); no forced mismatch warning
    result.recommendations.append(
        f"Dolum süresi: kullanılan {fill_time_s:.2f} s; pratik öneri {auto_fill_time_s:.2f} s; "
        f"Campbell önerisi {campbell_fill_time_s:.2f} s ({campbell_fill_time_basis}). "
        f"Döküm verimi: %{pour_yield*100:.1f}."
    )

    if design_areas["As_cm2"] > 0:
        ratio_str = ":".join(f"{r:.2f}" for r in design_ratio)
        result.recommendations.append(
            f"Teorik kesit alanları (As:Ar:Ag={ratio_str}): "
            f"sprue taban={design_areas['As_cm2']:.2f} cm², runner={design_areas['Ar_total_cm2']:.2f} cm², "
            f"gate toplam={design_areas['Ag_total_cm2']:.2f} cm² (her biri {design_areas['Ag_each_cm2']:.2f} cm²); "
            f"choke hızı={design_areas['Vc_ms']:.2f} m/s."
        )

    # v9.0: feeder / part mass and volume ratios.
    if result.riser_results:
        total_riser_mass_kg = sum(r.mass_kg for r in result.riser_results)
    else:
        # Fallback: estimate riser volume from any bodies not part/gating/core.
        riser_volume_cm3 = sum(
            b.volume_cm3 for b in bodies if b.body_type == BodyType.RISER
        )
        total_riser_mass_kg = riser_volume_cm3 * alloy.density_g_cm3 / 1000.0
    gating_mass_kg = max(0.0, total_mass_kg - part_mass_kg - total_riser_mass_kg)
    feed_to_part_mass_ratio = (
        (total_riser_mass_kg + gating_mass_kg) / part_mass_kg
        if part_mass_kg > 0.0 else 0.0
    )
    feed_to_part_volume_ratio = (
        (total_metal_volume_cm3 - part_volume_cm3) / part_volume_cm3
        if part_volume_cm3 > 0.0 else 0.0
    )
    result.recommendations.append(
        f"Besleyici/yolluk toplam kütlesi = {total_riser_mass_kg + gating_mass_kg:.3f} kg; "
        f"parça kütlesi = {part_mass_kg:.3f} kg; besleyici/parça kütlesi oranı = {feed_to_part_mass_ratio:.2f}; "
        f"hacim oranı = {feed_to_part_volume_ratio:.2f}."
    )

    return GateResult(
        total_ingate_contact_area_cm2=gate_contact_area_cm2,
        runner_min_area_cm2=runner_min_area_cm2,
        sprue_base_area_cm2=sprue_base_cm2,
        required_sprue_area_cm2=final_sprue_required_cm2,
        campbell_ok=campbell_ok,
        bernoulli_ok=bernoulli_ok,
        ingate_on_thick_region=ingate_on_thick,
        ingate_avg_m_mm=ingate_avg_m,
        ingate_max_m_mm=ingate_max_m,
        ingate_thickness_mm=ingate_thickness_mm,
        runner_thickness_mm=runner_thickness_mm,
        required_runner_area_cm2=required_runner_area_cm2,
        required_ingate_area_cm2=required_ingate_area_cm2,
        runner_ok=runner_ok,
        ingate_ok=ingate_ok,
        elbow_count=elbow_count,
        head_loss_mm=head_loss_cm * 10.0,
        effective_head_mm=effective_head_cm * 10.0,
        required_sprue_area_with_losses_cm2=required_sprue_area_with_losses_cm2,
        ingate_velocity_m_s=ingate_velocity_m_s,
        ingate_max_velocity_m_s=max_ingate_velocity,
        reynolds=reynolds,
        froude=froude,
        turbulent=turbulent,
        ingate_flow_rate_m3_s=ingate_flow_rate_m3_s,
        ingate_fill_time_s=ingate_fill_time_s,
        velocity_fill_time_match_ok=velocity_fill_time_match_ok,
        required_ingate_area_for_velocity_cm2=required_selected_area_for_velocity_m2 * 1e4,
        velocity_area_ok=velocity_area_ok,
        fluidity_length_mm=fluidity_length_mm,
        sprue_throat_area_cm2=sprue_throat_cm2,
        sprue_base_bottom_area_cm2=sprue_base_bottom_cm2,
        sprue_thickness_mm=sprue_thickness_mm,
        selected_section_key=velocity_section_key,
        selected_velocity_m_s=user_v,
        section_flows=section_flows,
        effective_gate_section=effective_gate_section,
        detected_gating_system=detected_system,
        recommended_gating_system=detected_system,
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
        design_sprue_base_area_cm2=float(design_areas["As_cm2"]),
        design_runner_area_cm2=float(design_areas["Ar_total_cm2"]),
        design_gate_total_area_cm2=float(design_areas["Ag_total_cm2"]),
        design_gate_each_area_cm2=float(design_areas["Ag_each_cm2"]),
        design_sprue_diameter_mm=float(design_areas["d_sprue_mm"]),
        design_gate_diameter_mm=float(design_areas["d_ingate_each_mm"]),
        design_choke_velocity_m_s=float(design_areas["Vc_ms"]),
        design_gating_ratio=design_ratio,
        sprue_design_ok=sprue_design_ok,
        runner_design_ok=runner_design_ok,
        gate_design_ok=gate_design_ok,
        part_mass_kg=part_mass_kg,
        total_riser_mass_kg=total_riser_mass_kg,
        gating_mass_kg=gating_mass_kg,
        feed_to_part_mass_ratio=feed_to_part_mass_ratio,
        feed_to_part_volume_ratio=feed_to_part_volume_ratio,
    )
