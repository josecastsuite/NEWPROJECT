"""Automatic riser design / recommendation engine for JoseCast Analyzer."""

from typing import List, Tuple

import numpy as np
from scipy import ndimage

from core.materials import Alloy
from core.types import AnalysisResult, BodyType, HotSpot, RiserProposal


def _riser_volume_sphere(diameter_mm: float) -> float:
    return (np.pi / 6.0) * (diameter_mm ** 3) / 1000.0  # cm³


def _diameter_for_sphere_m(m_mm: float) -> float:
    return 6.0 * m_mm


def _cylinder_diameter_for_m(m_mm: float, height_to_diameter: float = 1.0) -> float:
    # Cylinder with H = h2d * D, top cooling only (bottom attached to casting)
    # M = V/A = (pi D^2 H / 4) / (pi D H + 2 * pi D^2 / 4)
    # Solve for D given M and h2d.
    h2d = float(height_to_diameter)
    ratio = (h2d / 4.0) / (h2d + 0.5)
    return m_mm / ratio


def _surface_normal(
    is_metal: np.ndarray, surface_vox: np.ndarray
) -> np.ndarray:
    """Estimate outward surface normal from 6-neighbour non-metal directions."""
    directions = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=int,
    )
    normal = np.zeros(3, dtype=float)
    for d in directions:
        nb = surface_vox + d
        if (
            0 <= nb[0] < is_metal.shape[0]
            and 0 <= nb[1] < is_metal.shape[1]
            and 0 <= nb[2] < is_metal.shape[2]
            and not is_metal[nb[0], nb[1], nb[2]]
        ):
            normal += d.astype(float)
    norm = np.linalg.norm(normal)
    if norm > 0:
        normal /= norm
    else:
        normal = np.array([0.0, 0.0, 1.0])
    return normal


def _best_surface_attachment(
    result: AnalysisResult,
    hs: HotSpot,
    search_mm: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Find the best part surface point to attach a feeder/pad near a hot spot.

    Returns (surface_voxel, surface_position_mm, local_section_thickness_mm).
    """
    part_mask = result.grid == BodyType.PART
    is_metal = result.is_metal
    sdf = result.sdf
    dx = result.dx_mm
    origin = result.origin_mm
    shape = is_metal.shape
    hs_vox = np.round((hs.position_mm - origin) / dx).astype(int)

    # Surface voxels: part voxels with at least one non-metal 6-neighbor.
    eroded = ndimage.binary_erosion(is_metal, iterations=1)
    surface_mask = part_mask & (~eroded)
    if not surface_mask.any():
        return hs_vox, hs.position_mm.copy(), hs.t_section_mm

    search_vox = int(np.ceil(search_mm / dx))
    slices = tuple(
        slice(max(0, hs_vox[i] - search_vox), min(shape[i], hs_vox[i] + search_vox + 1))
        for i in range(3)
    )
    local_surface = surface_mask[slices]
    if not local_surface.any():
        # fall back to nearest surface anywhere
        local_surface = surface_mask
        base_vox = np.array([0, 0, 0])
    else:
        base_vox = np.array([s.start for s in slices], dtype=int)

    surf_voxels = np.argwhere(local_surface) + base_vox
    if len(surf_voxels) == 0:
        return hs_vox, hs.position_mm.copy(), hs.t_section_mm

    # Pre-compute distances (voxel coords)
    diff = surf_voxels - hs_vox
    distances_vox = np.linalg.norm(diff, axis=1)
    max_dist = max(distances_vox.max(), 1.0)

    scores = []
    thicknesses = []
    for vox in surf_voxels:
        n = _surface_normal(is_metal, vox)
        # prefer upward-facing, close to the hot spot, and on a reasonably thick section
        # estimate local thickness from metal neighbours just inside the surface
        inner_sdf_vals = []
        for d in np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        ):
            nb = vox + d
            if (
                0 <= nb[0] < shape[0]
                and 0 <= nb[1] < shape[1]
                and 0 <= nb[2] < shape[2]
                and is_metal[nb[0], nb[1], nb[2]]
            ):
                inner_sdf_vals.append(float(sdf[nb[0], nb[1], nb[2]]))
        t_local = 2.0 * (max(inner_sdf_vals) if inner_sdf_vals else 0.0)
        thicknesses.append(t_local)
        dist = float(np.linalg.norm(vox - hs_vox)) / max_dist
        score = 0.5 * n[2] - 0.3 * dist + 0.2 * min(t_local / max(hs.m_value_mm, 1.0), 2.0)
        # strongly penalise downward-facing surfaces
        if n[2] < -0.3:
            score -= 1.0
        scores.append(score)

    scores = np.array(scores)
    best_idx = int(np.argmax(scores))
    best_vox = surf_voxels[best_idx]
    best_pos = origin + best_vox * dx
    return best_vox, best_pos, thicknesses[best_idx]


def propose_risers(
    result: AnalysisResult, alloy: Alloy, existing_riser_count: int = 0
) -> List[RiserProposal]:
    """Generate concrete, geometry-aware riser / pad proposals for failing hot spots.

    Strategy:
      * If a riser already exists and all hot spots are fed, nothing to do.
      * For each failing hot spot, find the best part surface to attach a feeder
        (upward-facing, close, reasonably thick).
      * Use a cylindrical feeder/pad with H = 1.5 D; this gives a smaller volume
        than a sphere for the same modulus.
      * The required modulus is based on the local hot-spot modulus.
    """
    proposals: List[RiserProposal] = []
    if not result.hotspots:
        return proposals

    if existing_riser_count > 0 and all(hs.feed_ok for hs in result.hotspots):
        return proposals

    k_mod = getattr(alloy, "riser_m_factor", 1.2)
    h2d = 1.5  # height / diameter ratio for cylindrical feeder/pad

    for idx, hs in enumerate(result.hotspots):
        if hs.feed_ok and hs.darcy_ok and hs.heuvers_ok and existing_riser_count > 0:
            continue

        # Smart surface attachment: find the best place on the part surface.
        surface_vox, surface_pos, t_attach = _best_surface_attachment(result, hs)
        normal = _surface_normal(result.is_metal, surface_vox)

        # If the chosen surface points downward, prefer the vertical direction for
        # casting practicality while keeping the feeder outside the part.
        if normal[2] < 0:
            normal = np.array([0.0, 0.0, 1.0])

        # Decide between a feeder (riser) and a chill/çıkıcı.
        # Small, isolated hot spots on a relatively thin wall are good chill candidates
        # when at least one riser already exists; otherwise a feeder is still needed.
        prefer_chill = (
            existing_riser_count > 0
            and hs.m_value_mm <= 10.0
            and hs.t_section_mm <= 20.0
            and t_attach <= 20.0
        )

        if prefer_chill:
            shape = "chill"
            m_required = hs.m_value_mm
            # Chill insert: cylindrical metal (cast iron/steel) placed on the surface.
            diameter = max(1.5 * hs.t_section_mm, 2.0 * hs.m_value_mm, 12.0)
            height = diameter
            volume = (np.pi * (diameter ** 2) * height / 4.0) / 1000.0
            neck_diameter = 0.0
            neck_height = 0.0
        else:
            m_required = max(k_mod * hs.m_value_mm, 3.0)
            # Cylindrical feeder / pad is smaller than a sphere for the same modulus.
            shape = "cylinder"
            diameter = _cylinder_diameter_for_m(m_required, height_to_diameter=h2d)
            height = h2d * diameter
            volume = (np.pi * (diameter ** 2) * height / 4.0) / 1000.0
            # Neck/contact dimensions: the local section feeds into the feeder/pad.
            neck_diameter = max(diameter * 0.5, t_attach * 0.8, hs.t_section_mm * 0.8, 5.0)
            neck_height = max(t_attach * 0.5, hs.t_section_mm * 0.5, 3.0)

        # The centre is placed along the outward normal, base at surface.
        placement = surface_pos + normal * (height / 2.0)

        reason_parts = []
        if not hs.feed_ok:
            reason_parts.append("besleme mesafesi/yol yetersiz")
        if not hs.darcy_ok:
            reason_parts.append("Darcy basınç kaybı kesme")
        if not hs.heuvers_ok:
            reason_parts.append("Heuver çemberleri bozuk")
        if not hs.directional_ok:
            reason_parts.append("yönlü katılaşma bozuk")
        if existing_riser_count == 0:
            reason_parts.append("sistemde riser yok")
        reason = "; ".join(reason_parts) if reason_parts else "önlem amaçlı büyütme"
        reason += f" | bağlantı: ({surface_pos[0]:.1f}, {surface_pos[1]:.1f}, {surface_pos[2]:.1f}) mm, normal=({normal[0]:.2f},{normal[1]:.2f},{normal[2]:.2f})"

        proposals.append(
            RiserProposal(
                target_hotspot_index=idx,
                target_hotspot_position_mm=np.array(hs.position_mm, dtype=float),
                placement_mm=placement,
                reason=reason,
                m_required_mm=m_required,
                shape=shape,
                diameter_mm=diameter,
                height_mm=height,
                volume_cm3=float(volume),
                neck_diameter_mm=float(neck_diameter),
                neck_height_mm=float(neck_height),
            )
        )

    return proposals
