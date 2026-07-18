"""STEP file loader based on CadQuery + Trimesh."""

import os
from typing import List, Optional

import cadquery as cq
import numpy as np
import trimesh

from core.types import Body


def load_step(path: str, tolerance: Optional[float] = None, angular_tolerance: float = 0.1) -> List[Body]:
    """Return a list of Body objects, one per solid in the STEP file.

    Parameters
    ----------
    tolerance : float | None
        Linear deflection for tessellation. If None, it is chosen as 0.1% of
        the solid diagonal, clamped between 0.05 and 2.0 mm. Smaller = finer.
    angular_tolerance : float
        Angular deflection for tessellation (default 0.1 rad).
    """
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

        volume_mm3 = mesh.volume
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
            )
        )

    return bodies
