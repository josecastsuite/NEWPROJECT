"""Automatic riser design / recommendation engine for JoseCast Analyzer."""

from typing import List

import numpy as np

from core.materials import Alloy
from core.types import AnalysisResult, HotSpot, RiserProposal


def _riser_volume_sphere(diameter_mm: float) -> float:
    return (np.pi / 6.0) * (diameter_mm ** 3) / 1000.0  # cm³


def _diameter_for_sphere_m(m_mm: float) -> float:
    return 6.0 * m_mm


def _cylinder_diameter_for_m(m_mm: float, height_to_diameter: float = 1.0) -> float:
    # Cylinder with H = h2d * D, top cooling only (bottom attached to casting)
    # M = V/A = (pi D^2 H / 4) / (pi D H + pi D^2 / 4)
    # Solve for D given M and h2d.
    h2d = float(height_to_diameter)
    # A/D^2 term, V/D^3 term => M/D = (h2d/4) / (h2d + 0.25)
    ratio = (h2d / 4.0) / (h2d + 0.25)
    return m_mm / ratio


def propose_risers(result: AnalysisResult, alloy: Alloy, existing_riser_count: int = 0) -> List[RiserProposal]:
    """Generate concrete riser proposals for hot spots that cannot be fed.

    Strategy:
      * If a riser already exists and is large enough, do not propose a new one.
      * For every failing hot spot (feed_ok == False or darcy_ok == False or heuvers_ok == False)
        build a sphere or cylinder riser above the hot spot.
      * Riser modulus = max(1.2 * hot_spot_M, 1.2 * hot_spot_M + distance_penalty).
      * Placement is on top of the hot spot (z + t_section/2 + D/2).
    """
    proposals: List[RiserProposal] = []
    if not result.hotspots:
        return proposals

    # If there is at least one riser and all hot spots are fed, nothing to do.
    if existing_riser_count > 0 and all(hs.feed_ok for hs in result.hotspots):
        return proposals

    for idx, hs in enumerate(result.hotspots):
        if hs.feed_ok and hs.darcy_ok and hs.heuvers_ok and existing_riser_count > 0:
            continue

        # Required modulus: Chvorinov-modulus rule + distance penalty.
        distance_penalty = 0.0
        if hs.max_feeding_distance_mm > 0 and hs.dist_to_riser_mm > hs.max_feeding_distance_mm:
            distance_penalty = 0.15 * (hs.dist_to_riser_mm - hs.max_feeding_distance_mm)
        elif not np.isfinite(hs.dist_to_riser_mm) or hs.dist_to_riser_mm > 9999:
            distance_penalty = 2.0

        # Feeding resistance further increases the required riser size.
        resistance_penalty = 0.05 * hs.feeding_cost

        m_required = 1.2 * hs.m_value_mm + distance_penalty + resistance_penalty
        m_required = max(m_required, 1.2 * hs.m_value_mm + 1.0)

        # Use a spherical side/top riser by default; switch to cylinder for very large M.
        if m_required <= 25.0:
            shape = "sphere"
            diameter = _diameter_for_sphere_m(m_required)
            height = diameter
        else:
            shape = "cylinder"
            diameter = _cylinder_diameter_for_m(m_required, height_to_diameter=1.0)
            height = diameter

        volume = _riser_volume_sphere(diameter)
        if shape == "cylinder":
            volume = (np.pi * (diameter ** 2) * height / 4.0) / 1000.0

        # Neck: connect riser to casting, smaller than riser diameter.
        neck_diameter = max(diameter * 0.4, hs.t_section_mm * 0.8, 5.0)
        neck_height = max(hs.t_section_mm * 0.5, 3.0)

        placement = np.array(hs.position_mm, dtype=float)
        placement[2] += (hs.t_section_mm / 2.0) + (diameter / 2.0)

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
