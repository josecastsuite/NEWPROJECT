"""STEP file loader based on CadQuery + Trimesh."""

import os
import re
from typing import List, Optional

import cadquery as cq
import numpy as np
import trimesh

from core.types import Body


def _detect_step_unit(path: str) -> str:
    """Parse the STEP file for a length unit declaration.

    Common AP203/214/242 STEP files encode the unit via
    ``SI_UNIT( .MILLI., .METRE. )``.  If no explicit length unit is found,
    ``mm`` is returned as the safe default.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception:
        return "mm"

    # Look for LENGTH_UNIT lines that contain an SI_UNIT; the prefix before
    # .METRE. tells us the scale (MILLI -> mm, CENTI -> cm, etc.).
    for line in text.splitlines():
        if "LENGTH_UNIT" in line and "SI_UNIT" in line and ".METRE." in line.upper():
            m = re.search(
                r"SI_UNIT\s*\(\s*(?:\.([A-Z]+)\.|([\$\*]))\s*,\s*\.METRE\.\s*\)",
                line,
                re.IGNORECASE,
            )
            if not m:
                # No prefix or omitted: assume metre.
                m = re.search(r"SI_UNIT\s*\(\s*\.METRE\.\s*\)", line, re.IGNORECASE)
                if m:
                    return "m"
                continue
            prefix = (m.group(1) or "").upper()
            if prefix == "MILLI":
                return "mm"
            if prefix == "CENTI":
                return "cm"
            if prefix == "DECI":
                return "dm"
            if prefix == "MICRO":
                return "um"
            # No prefix / omitted -> metre.
            return "m"

    # Non-SI conversion units (e.g. inch) are not SI prefixes.
    if "CONVERSION_BASED_UNIT" in text and "INCH" in text.upper():
        return "inch"

    return "mm"


def _choose_mm_scale(mesh: trimesh.Trimesh, step_unit: str) -> float:
    """Return a scale factor that converts ``mesh`` units to millimetres.

    CadQuery/OCCT normally returns coordinates in millimetres, but some STEP
    exporters write values in metres even though the file declares metres, or
    use conversion-based units (e.g. mils).  We try the declared unit and its
    reciprocal and pick the one that lands the bounding-box diagonal in a
    plausible casting size range.
    """
    unit_to_mm = {
        "um": 0.001,
        "mm": 1.0,
        "cm": 10.0,
        "dm": 100.0,
        "m": 1000.0,
        "inch": 25.4,
    }
    diag = float(np.linalg.norm(mesh.bounding_box.extents))
    if diag <= 1e-12:
        return 1.0

    candidates = [1.0]
    declared = unit_to_mm.get(step_unit, 1.0)
    if declared not in (1.0,):
        candidates.extend([declared, 1.0 / declared])

    best = 1.0
    best_score = float("inf")
    for s in candidates:
        scaled = diag * s
        # A typical casting is between 10 mm and 10 000 mm; prefer values near
        # the middle of that range.
        if 10.0 <= scaled <= 10000.0:
            score = abs(scaled - 500.0)
        else:
            score = float("inf")
        if score < best_score:
            best_score = score
            best = s

    # If no candidate lands in the plausible range, still prefer a smaller model
    # for metre/millimetre ambiguity: a 0.4 m model loaded as 0.4 mm must grow,
    # whereas a 400 mm model loaded as 400 mm must stay.
    if best_score == float("inf") and step_unit == "m":
        if diag < 1.0:
            best = 1000.0
        elif diag > 10000.0:
            best = 0.001

    return best


def load_step(path: str, tolerance: Optional[float] = None, angular_tolerance: float = 0.1) -> List[Body]:
    """Return a list of Body objects, one per solid in the STEP file.

    All coordinates are converted to millimetres regardless of the unit declared
    inside the STEP file, so downstream voxel and area calculations are correct.

    Parameters
    ----------
    tolerance : float | None
        Linear deflection for tessellation in millimetres. If None, it is chosen
        as 0.1% of the solid diagonal, clamped between 0.05 and 2.0 mm. Smaller = finer.
    angular_tolerance : float
        Angular deflection for tessellation (default 0.1 rad).
    """
    step_unit = _detect_step_unit(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"STEP dosyası bulunamadı: {path}")

    wp = cq.importers.importStep(path)

    # Try several ways to get the list of individual solids.
    solids = []
    val = wp.val() if callable(wp.val) else getattr(wp, "val", None)
    if hasattr(val, "Solids"):
        solids = list(val.Solids())
    elif hasattr(val, "__iter__"):
        solids = list(val)
    elif hasattr(wp, "solids"):
        solids = list(wp.solids().vals())
    else:
        # Fallback: one object
        solids = [val]

    bodies: List[Body] = []
    for i, solid in enumerate(solids):
        if solid is None:
            continue

        # Choose a tessellation tolerance that is 0.1% of the solid size.
        # The tolerance is expressed in the units returned by CadQuery (usually
        # millimetres when the STEP file is well-formed).
        tol = tolerance
        if tol is None:
            try:
                bb = solid.BoundingBox()
                diag = max(bb.xmax - bb.xmin, bb.ymax - bb.ymin, bb.zmax - bb.zmin)
                tol = max(0.05, min(diag * 0.001, 2.0))
            except Exception:
                tol = 0.5

        try:
            vertices, triangles = solid.tessellate(tol, angular_tolerance)
        except Exception:
            # If a compound is passed, it may not support tessellate directly
            vertices, triangles = solid.toTris().tessellate(tol, angular_tolerance)

        if len(triangles) == 0:
            continue

        # CadQuery / OCCT sometimes returns Vector objects instead of plain tuples.
        def _to_xyz(v):
            return (float(v.X), float(v.Y), float(v.Z)) if hasattr(v, "X") else (float(v.x), float(v.y), float(v.z))

        vertices = np.array([_to_xyz(v) for v in vertices], dtype=np.float64)
        triangles = np.array([
            [int(t[0]), int(t[1]), int(t[2])] for t in triangles
        ], dtype=np.int32)

        # SolidWorks / CAD programs may produce duplicate vertices
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=True)
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()

        if len(mesh.faces) == 0:
            continue

        # Auto-scale to millimetres based on physical plausibility.  Most STEP
        # exporters write coordinates in mm; some write metres.  We try the
        # declared unit and its reciprocal and keep the scale that puts the model
        # in a sensible casting size range (10 mm .. 10 000 mm).
        scale = _choose_mm_scale(mesh, step_unit)
        raw_diag = float(np.linalg.norm(mesh.bounding_box.extents))
        print(
            f"[STEP_LOADER] {os.path.basename(path)} solid {i + 1}: "
            f"detected_unit={step_unit}, raw_diag={raw_diag:.3f}, chosen_scale={scale}",
            flush=True,
        )
        if abs(scale - 1.0) > 1e-9:
            mesh.apply_scale(scale)
            mesh.merge_vertices()
            mesh.remove_unreferenced_vertices()

        try:
            volume_mm3 = float(mesh.volume)
        except Exception:
            # If the triangulation is not watertight, fall back to the convex
            # hull volume so downstream code still has a sensible order of
            # magnitude.
            try:
                hull = mesh.convex_hull
                volume_mm3 = float(hull.volume)
            except Exception:
                volume_mm3 = 0.0
        volume_cm3 = volume_mm3 / 1000.0
        center = mesh.center_mass if mesh.is_watertight else mesh.centroid

        # Solid name from label if available
        name = getattr(solid, "label", None) or f"Body_{i + 1}"

        bodies.append(
            Body(
                index=i,
                name=name,
                vertices=mesh.vertices,
                faces=mesh.faces,
                mesh=mesh,
                volume_cm3=volume_cm3,
                center=center,
                source_unit=step_unit,
            )
        )

    return bodies
