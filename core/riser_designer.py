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
    """Cylinder with H = h2d * D.  Bottom face is attached to the casting and
    therefore does not cool; only side and top surfaces contribute to A."""
    h2d = float(height_to_diameter)
    # V = pi D^2 H / 4
    # A = pi D H + pi D^2 / 4
    # M = V/A = ((h2d / 4) * D) / (h2d + 0.25)
    ratio = (h2d / 4.0) / (h2d + 0.25)
    return m_mm / ratio


def _cylinder_m_for_d(d_mm: float, height_to_diameter: float = 1.0) -> float:
    """Geometric modulus M of a cylinder with H = h2d * D (bottom attached)."""
    h2d = float(height_to_diameter)
    ratio = (h2d / 4.0) / (h2d + 0.25)
    return d_mm * ratio


def _cylinder_volume_mm3(d_mm: float, height_to_diameter: float = 1.0) -> float:
    h = height_to_diameter * d_mm
    return np.pi * (d_mm ** 2) * h / 4.0


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


def _sfsa_riser_volume_cm3(shape_factor: float, casting_volume_cm3: float) -> float:
    """Bishop–Pellini / SFSA empirical relation for plate-like sections.

    VR / VC = 2.51 * SF^(-0.74)
    where SF = (L + W) / T and VR / VC is riser-to-casting volume ratio.
    """
    sf = max(float(shape_factor), 1.5)
    ratio = 2.51 * (sf ** -0.74)
    return ratio * casting_volume_cm3


def _feeding_zone_geometry(
    result: AnalysisResult,
    hs: HotSpot,
    threshold_ratio: float = 0.75,
) -> Tuple[float, float, float, float, float, float]:
    """Find the connected high-modulus region around a hotspot and compute
    plate-like dimensions, volume and shape factor.

    Returns (t_mm, w_mm, l_mm, shape_factor, V_c_cm3, M_zone_max_mm).
    """
    dx = result.dx_mm
    origin = result.origin_mm
    part_mask = result.grid == BodyType.PART
    sdf = result.sdf

    if result.curvature_mean is not None and result.curvature_mean.size == sdf.size:
        sf_field = np.clip(1.0 - result.curvature_mean * sdf, 0.77, 3.0)
    else:
        sf_field = np.ones_like(sdf)
    M_mod = sdf / sf_field

    m_threshold = max(hs.m_value_mm * threshold_ratio, 1.0)
    zone_mask = part_mask & (M_mod >= m_threshold)

    # 26-connectivity for a natural blob
    structure = np.ones((3, 3, 3), dtype=int)
    labeled, num = ndimage.label(zone_mask, structure=structure)

    hs_vox = np.round((hs.position_mm - origin) / dx).astype(int)
    hs_vox = np.clip(hs_vox, 0, np.array(sdf.shape) - 1)

    label_id = 0
    if num > 0:
        label_id = int(labeled[hs_vox[0], hs_vox[1], hs_vox[2]])

    if label_id == 0:
        # Fallback: sphere around the hotspot of radius ~2M
        radius_vox = int(np.ceil(2.0 * hs.m_value_mm / dx))
        fallback_mask = np.zeros_like(part_mask)
        ranges = []
        for i in range(3):
            lo = max(0, hs_vox[i] - radius_vox)
            hi = min(sdf.shape[i], hs_vox[i] + radius_vox + 1)
            ranges.append((lo, hi))
        x = np.arange(ranges[0][0], ranges[0][1])[:, None, None]
        y = np.arange(ranges[1][0], ranges[1][1])[None, :, None]
        z = np.arange(ranges[2][0], ranges[2][1])[None, None, :]
        dist2 = (
            ((x - hs_vox[0]) * dx) ** 2
            + ((y - hs_vox[1]) * dx) ** 2
            + ((z - hs_vox[2]) * dx) ** 2
        )
        fallback_mask[
            ranges[0][0]:ranges[0][1],
            ranges[1][0]:ranges[1][1],
            ranges[2][0]:ranges[2][1],
        ] = dist2 < (2.0 * hs.m_value_mm) ** 2
        fallback_mask &= part_mask
        comp_mask = fallback_mask
    else:
        comp_mask = labeled == label_id

    comp_vox = np.argwhere(comp_mask)
    if comp_vox.size == 0:
        # Last resort single-voxel estimate
        t = w = l = max(2.0 * hs.m_value_mm, dx)
        V_c_cm3 = (dx ** 3) / 1000.0
        M_zone_max = hs.m_value_mm
        sf = 2.0
        return t, w, l, sf, V_c_cm3, M_zone_max

    dims_mm = (comp_vox.max(axis=0) - comp_vox.min(axis=0) + 1) * dx
    dims_sorted = np.sort(dims_mm)
    t, w, l = float(dims_sorted[0]), float(dims_sorted[1]), float(dims_sorted[2])

    V_c_mm3 = float(comp_mask.sum()) * (dx ** 3)
    V_c_cm3 = V_c_mm3 / 1000.0
    M_zone_max = float(M_mod[comp_mask].max()) if comp_mask.any() else hs.m_value_mm
    sf = (l + w) / max(t, 1e-3)
    return t, w, l, sf, V_c_cm3, M_zone_max


def propose_risers(
    result: AnalysisResult, alloy: Alloy, existing_riser_count: int = 0
) -> List[RiserProposal]:
    """Generate concrete, geometry-aware riser / pad / chill / exothermic mini-riser
    proposals for failing hot spots.

    Strategy:
      * If a riser already exists and all hot spots are fed, nothing to do.
      * For each failing hot spot, find the connected high-modulus feeding zone
        and size the riser using both the Caine modulus rule and the
        Bishop–Pellini / SFSA shape-factor volume ratio.
      * If the resulting conventional riser is larger than the part itself
        (>30% of part volume or >0.5 of the smallest part dimension), do not
        present giant modulus/diameter numbers; instead recommend a mini
        exothermic riser or chill and hide the infeasible dimensions.
      * Small, isolated hot spots on thin walls remain chill candidates.
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
    h2d = 1.5  # height / diameter ratio for a side-mounted cylindrical feeder
    exo_yield = getattr(alloy, "exothermic_volume_yield", 0.45)
    exo_mod_factor = getattr(alloy, "exothermic_modulus_factor", 1.5)

    ratio = (h2d / 4.0) / (h2d + 0.25)

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

        # Feeding zone geometry and SFSA shape-factor sizing.
        t_zone, w_zone, l_zone, sf_zone, V_c_cm3, M_zone_max = _feeding_zone_geometry(
            result, hs
        )

        # Caine modulus requirement.
        m_required = max(k_mod * hs.m_value_mm, 3.0)

        # Diameter required by the modulus rule (bottom-attached cylinder).
        D_modulus = m_required / ratio

        # Volume required by the SFSA shape-factor / volume-ratio rule.
        V_sfsa_cm3 = _sfsa_riser_volume_cm3(sf_zone, V_c_cm3)
        V_sfsa_mm3 = V_sfsa_cm3 * 1000.0
        D_volume = (4.0 * V_sfsa_mm3 / (np.pi * h2d)) ** (1.0 / 3.0)

        # Use the larger of the two: must satisfy both modulus and feed volume.
        D = max(D_modulus, D_volume)
        H = h2d * D
        V = _cylinder_volume_mm3(D, h2d) / 1000.0  # cm³
        M_riser = _cylinder_m_for_d(D, h2d)

        # Feasibility envelope: a riser should not exceed the feeding-zone volume
        # by too much, nor should it grow beyond a global fraction of the part.
        max_diameter_mm = max(t_zone * 2.0, part_smallest_dim_mm * 0.5, 20.0)
        max_volume_cm3 = min(0.30 * part_volume_cm3, max(1.5 * V_c_cm3, 0.0))

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
            D, H, V = diameter, height, volume
            M_riser = 0.0

            if D > max_diameter_mm or V > max_volume_cm3:
                infeasible = True
                warning = (
                    "Önerilen çıkıcı (chill) parça geometrisine sığmıyor. "
                    "Mini exotermik besleyici veya farklı bir soğutucu yerleştirmek gerekebilir; "
                    "bölgeyi kalınlaştırmak veya geometriyi değiştirmek de düşünülebilir."
                )
        else:
            if D > max_diameter_mm or V > max_volume_cm3:
                # Try a mini exothermic riser first.
                m_exo_physical = m_required / exo_mod_factor
                D_exo = m_exo_physical / ratio
                H_exo = h2d * D_exo
                V_exo_mm3 = _cylinder_volume_mm3(D_exo, h2d)
                V_exo = V_exo_mm3 * exo_yield / 1000.0

                if D_exo <= max_diameter_mm and V_exo <= max_volume_cm3:
                    shape = "exothermic"
                    exothermic = True
                    D, H, V = D_exo, H_exo, V_exo
                    M_riser = _cylinder_m_for_d(D, h2d)
                else:
                    # Try a chill of the largest feasible size.
                    D_chill = max(1.5 * t_zone, 2.0 * hs.m_value_mm, 12.0)
                    H_chill = D_chill
                    V_chill = (np.pi * D_chill ** 2 * H_chill / 4.0) / 1000.0

                    if D_chill <= max_diameter_mm and V_chill <= max_volume_cm3:
                        shape = "chill"
                        D, H, V = D_chill, H_chill, V_chill
                        M_riser = 0.0
                    else:
                        infeasible = True
                        D = H = V = 0.0
                        M_riser = 0.0
                        warning = (
                            "Konvansiyonel, mini exotermik veya çıkıcı (chill) besleyici "
                            "parça geometrisine sığmıyor. Öneri: bölgeye soğutucu (chill) ekleyin, "
                            "kalın bölgeyi inceltin, geçiş yarıçapını büyütün veya parçayı yeniden tasarlayın."
                        )

        if shape in ("cylinder", "exothermic"):
            neck_diameter = max(
                D * 0.5, t_attach * 0.8, hs.t_section_mm * 0.8, 5.0
            )
            neck_height = max(t_attach * 0.5, hs.t_section_mm * 0.5, 3.0)
        else:
            neck_diameter = 0.0
            neck_height = 0.0

        placement = surface_pos + normal * (H / 2.0)

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
            reason_parts.append("çıkıcı (chill) önerildi")
        elif shape == "exothermic":
            reason_parts.append("konvansiyonel besleyici sığmadı -> mini exotermik besleyici önerildi")
        elif shape == "cylinder":
            reason_parts.append("konvansiyonel silindirik besleyici önerildi")

        if infeasible:
            reason_parts.append(
                "önerilen besleyici/çıkıcı parça geometrisine sığmıyor; kullanıcı kararı gerekiyor"
            )

        reason = "; ".join(reason_parts) if reason_parts else "önlem amaçlı"
        reason += (
            f" | bağlantı: ({surface_pos[0] / 10.0:.1f}, {surface_pos[1] / 10.0:.1f}, {surface_pos[2] / 10.0:.1f}) cm, "
            f"normal=({normal[0]:.2f},{normal[1]:.2f},{normal[2]:.2f})"
        )

        proposals.append(
            RiserProposal(
                target_hotspot_index=idx,
                target_hotspot_position_mm=np.array(hs.position_mm, dtype=float),
                placement_mm=placement,
                reason=reason,
                m_required_mm=float(M_riser if M_riser > 0 else m_required),
                shape=shape,
                diameter_mm=float(D),
                height_mm=float(H),
                volume_cm3=float(V),
                neck_diameter_mm=float(neck_diameter),
                neck_height_mm=float(neck_height),
                exothermic=exothermic,
                infeasible=infeasible,
                warning=warning,
            )
        )

    return proposals
