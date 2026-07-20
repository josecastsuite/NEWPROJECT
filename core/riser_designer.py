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
    """Generate concrete, geometry-aware riser / pad / chill / exothermic mini-riser
    proposals for failing hot spots.

    Strategy:
      * If a riser already exists and all hot spots are fed, nothing to do.
      * For each failing hot spot, find the best part surface to attach a feeder
        (upward-facing, close, reasonably thick).
      * Small, isolated hot spots on thin walls are good chill candidates.
      * If a conventional riser would be larger than the part, try an exothermic
        mini-riser first; if that still does not fit, flag it infeasible and warn
        the user explicitly instead of generating an unbuildable body.
    """
    proposals: List[RiserProposal] = []
    if not result.hotspots:
        return proposals

    if existing_riser_count > 0 and all(hs.feed_ok for hs in result.hotspots):
        return proposals

    part_mask = result.grid == BodyType.PART
    if not part_mask.any():
        return proposals

    # Global part geometry for sanity checks.
    part_volume_cm3 = max(
        result.part_volume_mm3 / 1000.0,
        float(part_mask.sum()) * (result.dx_mm ** 3) / 1000.0,
    )
    part_surface_mm2 = max(result.part_surface_area_mm2, 1.0)
    part_global_m_mm = (
        result.part_volume_mm3 / part_surface_mm2 if part_surface_mm2 > 0 else 0.0
    )

    pi, pj, pk = np.where(part_mask)
    part_min = (
        np.array([pi.min(), pj.min(), pk.min()], dtype=float) * result.dx_mm
        + result.origin_mm
    )
    part_max = (
        np.array([pi.max(), pj.max(), pk.max()], dtype=float) * result.dx_mm
        + result.origin_mm
    )
    part_smallest_dim_mm = float((part_max - part_min).min())

    # Distance to any existing cooling sprue/chill body.
    chill_mask = result.grid == BodyType.COOLING_SPRUE
    chill_dist = None
    if chill_mask.any():
        chill_dist = ndimage.distance_transform_edt(
            ~chill_mask, sampling=result.dx_mm
        )

    k_mod = getattr(alloy, "riser_m_factor", 1.2)
    h2d = 1.5  # height / diameter ratio for cylindrical feeder
    exo_yield = getattr(alloy, "exothermic_volume_yield", 0.45)
    exo_scale = float(exo_yield) ** (1.0 / 3.0)

    for idx, hs in enumerate(result.hotspots):
        # Hot spot already fed by an existing riser: nothing to propose.
        if hs.feed_ok and existing_riser_count > 0:
            continue

        # Smart surface attachment: find the best place on the part surface.
        surface_vox, surface_pos, t_attach = _best_surface_attachment(result, hs)
        normal = _surface_normal(result.is_metal, surface_vox)
        if normal[2] < 0:
            normal = np.array([0.0, 0.0, 1.0])

        hs_vox = np.round((hs.position_mm - result.origin_mm) / result.dx_mm).astype(
            int
        )
        hs_vox = np.clip(hs_vox, 0, np.array(result.grid.shape) - 1)

        has_nearby_chill = False
        if chill_dist is not None:
            has_nearby_chill = float(
                chill_dist[hs_vox[0], hs_vox[1], hs_vox[2]]
            ) < max(20.0, 3.0 * hs.m_value_mm)

        is_small_thin = (
            hs.m_value_mm <= 10.0
            and hs.t_section_mm <= 20.0
            and t_attach <= 20.0
        )

        # Feasibility envelope.  A feeder/chill that is larger than the part itself
        # is unbuildable; exothermic mini-risers are tried before giving up.
        max_diameter_mm = max(t_attach * 2.0, part_smallest_dim_mm * 0.5, 20.0)
        max_volume_cm3 = 0.5 * part_volume_cm3

        m_required = max(k_mod * hs.m_value_mm, 3.0)

        shape = "cylinder"
        exothermic = False
        infeasible = False
        warning = ""

        if is_small_thin and not has_nearby_chill:
            # First choice for small, thin-wall hot spots is a chill/çıkıcı.
            shape = "chill"
            diameter = max(1.5 * hs.t_section_mm, 2.0 * hs.m_value_mm, 12.0)
            height = diameter
            volume = (np.pi * (diameter ** 2) * height / 4.0) / 1000.0

            if diameter > max_diameter_mm or volume > max_volume_cm3:
                # Chill does not fit; fall back to an exothermic mini-riser.
                shape = "exothermic"
                exothermic = True
                diameter = _cylinder_diameter_for_m(m_required, height_to_diameter=h2d)
                diameter *= exo_scale
                height = h2d * diameter
                volume = (np.pi * (diameter ** 2) * height / 4.0) / 1000.0

                if diameter > max_diameter_mm or volume > max_volume_cm3:
                    infeasible = True
                    warning = (
                        f"Önerilen çıkıcı (chill) çap {diameter:.1f} mm, hacim {volume:.1f} cm³ "
                        f"parça geometrisine sığmıyor (max çap ≈{max_diameter_mm:.1f} mm, "
                        f"max hacim ≈{max_volume_cm3:.1f} cm³). "
                        f"Çözüm kullanıcı kararıdır: bölgeyi kalınlaştırın, farklı bir soğutucu "
                        f"yerleştirin veya geometriyi değiştirin."
                    )
        else:
            # Larger hot spots: try a conventional cylindrical feeder first.
            diameter = _cylinder_diameter_for_m(m_required, height_to_diameter=h2d)
            height = h2d * diameter
            volume = (np.pi * (diameter ** 2) * height / 4.0) / 1000.0

            if diameter > max_diameter_mm or volume > max_volume_cm3:
                # Try an exothermic mini-riser to reduce volume.
                exo_diameter = diameter * exo_scale
                exo_height = height * exo_scale
                exo_volume = volume * exo_yield

                if exo_diameter <= max_diameter_mm and exo_volume <= max_volume_cm3:
                    shape = "exothermic"
                    exothermic = True
                    diameter, height, volume = exo_diameter, exo_height, exo_volume
                else:
                    infeasible = True
                    warning = (
                        f"Gerekli besleyici modülü M={m_required:.1f} mm, çap {diameter:.1f} mm, "
                        f"hacim {volume:.1f} cm³ parça geometrisine sığmıyor "
                        f"(max çap ≈{max_diameter_mm:.1f} mm, max hacim ≈{max_volume_cm3:.1f} cm³). "
                        f"Çözüm kullanıcı kararıdır: bölgeyi kalınlaştırın, soğutucu (chill) ekleyin, "
                        f"ekzotermik mini besleyici kullanın veya geometriyi değiştirin."
                    )

        if shape in ("cylinder", "exothermic"):
            neck_diameter = max(
                diameter * 0.5, t_attach * 0.8, hs.t_section_mm * 0.8, 5.0
            )
            neck_height = max(t_attach * 0.5, hs.t_section_mm * 0.5, 3.0)
        else:
            neck_diameter = 0.0
            neck_height = 0.0

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

        if shape == "chill":
            reason_parts.append("ince cidarlı küçük hotspot -> çıkıcı (chill) önerildi")
        elif shape == "exothermic":
            reason_parts.append("normal besleyici parçaya sığmadı -> ekzotermik mini besleyici önerildi")
        elif shape == "cylinder":
            reason_parts.append("konvansiyonel silindirik besleyici önerildi")

        if infeasible:
            reason_parts.append(
                "önerilen besleyici/çıkıcı parça geometrisine sığmıyor; kullanıcı kararı gerekiyor"
            )

        reason = "; ".join(reason_parts) if reason_parts else "önlem amaçlı"
        reason += (
            f" | bağlantı: ({surface_pos[0]:.1f}, {surface_pos[1]:.1f}, {surface_pos[2]:.1f}) mm, "
            f"normal=({normal[0]:.2f},{normal[1]:.2f},{normal[2]:.2f})"
        )

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
                exothermic=exothermic,
                infeasible=infeasible,
                warning=warning,
            )
        )

    return proposals
