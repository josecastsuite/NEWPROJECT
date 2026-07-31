"""Tetrahedral 3-D mesh generation for gating bodies.

Each gate body (sprue, runner, ingate, distributor, sprue throat, pouring basin,
curufluk) is meshed independently with TetGen (via meshpy).  The boundary faces
are classified as

* INLET   – face that connects the gate to its upstream neighbour
* OUTLET  – face(s) that connect the gate to its downstream neighbour(s)
* WALL    – outer surface in contact with the mold sand

The classification uses the directed gating topology from ``core.gating`` when
available, otherwise it falls back to the gravity ``up`` direction.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh

from core.types import Body, BodyType


_GATING_TYPES = {
    BodyType.SPRUE,
    BodyType.SPRUE_THROAT,
    BodyType.RUNNER,
    BodyType.DISTRIBUTOR,
    BodyType.INGATE,
    BodyType.CURUFLUK,
    BodyType.POURING_BASIN,
    BodyType.FILTER,
}


def _is_gating_body(body: Body) -> bool:
    return body.body_type in _GATING_TYPES


def _face_normal_area(
    nodes: np.ndarray, face: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Return the (non-unit) normal and area of a triangular face.

    The normal points outward if the face vertices are ordered consistently with
    the tetrahedron they belong to.
    """
    a = nodes[face[1]] - nodes[face[0]]
    b = nodes[face[2]] - nodes[face[0]]
    cross = np.cross(a, b)
    area = 0.5 * float(np.linalg.norm(cross))
    return cross, area


@dataclass
class GateMesh:
    """Tetrahedral mesh of one gate body with classified boundary faces."""

    body_name: str
    body_type: BodyType
    nodes: np.ndarray  # (n_nodes, 3) in mm
    tets: np.ndarray  # (n_tets, 4) int64
    cell_centers: np.ndarray  # (n_tets, 3)
    cell_volumes: np.ndarray  # (n_tets,)
    boundary_faces: np.ndarray  # (n_bnd, 3) indices into nodes
    boundary_normals: np.ndarray  # (n_bnd, 3) unit outward normals
    boundary_areas: np.ndarray  # (n_bnd,)
    boundary_tags: np.ndarray  # (n_bnd,) values: 0 WALL, 1 INLET, 2 OUTLET
    body: Optional[Body] = None  # reference to the source Body
    body_mesh: Optional[trimesh.Trimesh] = None  # cleaned surface mesh for SDF queries
    inlet_faces: np.ndarray = field(init=False)
    outlet_faces: np.ndarray = field(init=False)
    wall_faces: np.ndarray = field(init=False)

    def __post_init__(self):
        self.inlet_faces = np.where(self.boundary_tags == 1)[0]
        self.outlet_faces = np.where(self.boundary_tags == 2)[0]
        self.wall_faces = np.where(self.boundary_tags == 0)[0]


def _classify_boundary_faces(
    nodes: np.ndarray,
    boundary_faces: np.ndarray,
    body: Body,
    parent_body: Optional[Body] = None,
    child_bodies: Optional[List[Body]] = None,
    up_direction: Optional[np.ndarray] = None,
    tol_mm: float = 2.0,
) -> np.ndarray:
    """Tag boundary faces as WALL, INLET or OUTLET.

    The primary cue is the expected flow direction through the gate body.  This
    direction is inferred from the parent/child gating topology when available:
    flow runs from the parent to the body, or from the body to its children.
    Faces whose outward normal is strongly aligned with the flow are outlets,
    faces whose normal is strongly opposed are inlets, and the remainder are
    walls.  If the topology does not give an axis aligned with any face, the
    ``up`` direction (gravity up) is used as a fallback for a vertical element.
    """
    n_bnd = boundary_faces.shape[0]
    tags = np.zeros(n_bnd, dtype=np.int64)

    centroids = nodes[boundary_faces].mean(axis=1)
    normals = np.zeros_like(centroids)
    for i, f in enumerate(boundary_faces):
        a = nodes[f[1]] - nodes[f[0]]
        b = nodes[f[2]] - nodes[f[0]]
        n = np.cross(a, b)
        norm = float(np.linalg.norm(n)) + 1e-12
        normals[i] = n / norm

    # Candidate flow axes: parent->body and body->child.
    candidates: List[np.ndarray] = []
    parent = parent_body
    children = child_bodies or []
    if parent is not None and getattr(parent, "center", None) is not None:
        candidates.append(np.asarray(parent.center, dtype=float))
    for child in children:
        if child is not None and getattr(child, "center", None) is not None:
            candidates.append(np.asarray(child.center, dtype=float))
            break

    if up_direction is None:
        up = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        up = np.asarray(up_direction, dtype=float)
    up = up / (np.linalg.norm(up) + 1e-12)

    body_center = np.asarray(getattr(body, "center", centroids.mean(axis=0)), dtype=float)

    def _set_axis(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        proj = centroids @ axis
        n_dot = normals @ axis
        return proj, n_dot

    # Try each candidate flow axis and keep the one that best matches two end
    # caps (at least one inlet and one outlet with |n · axis| > 0.5).
    best_tags: Optional[np.ndarray] = None
    best_score = -1.0
    axes: List[np.ndarray] = []
    if parent is not None and candidates:
        axes.append(body_center - candidates[0])
    for child_center in candidates[1:]:
        axes.append(child_center - body_center)
    # Gravity fallback: flow is downward, so axis is -up.
    axes.append(-up)

    for axis in axes:
        proj, n_dot = _set_axis(axis)
        p_range = float(proj.max() - proj.min()) + 1e-12
        margin = 0.2 * p_range
        trial = tags.copy()

        inlet_mask = (n_dot < -0.5) & (proj < proj.min() + margin)
        if inlet_mask.any():
            trial[inlet_mask] = 1
        else:
            unassigned = np.where(trial == 0)[0]
            if unassigned.size:
                trial[unassigned[np.argmin(n_dot[unassigned])]] = 1

        outlet_mask = (n_dot > 0.5) & (proj > proj.max() - margin)
        if outlet_mask.any():
            trial[outlet_mask] = 2
        else:
            unassigned = np.where(trial == 0)[0]
            if unassigned.size:
                trial[unassigned[np.argmax(n_dot[unassigned])]] = 2

        score = float((trial == 1).sum() + (trial == 2).sum())
        # Prefer axes where the chosen faces are true end caps (large |dot|).
        if (trial == 1).any():
            score += abs(float(n_dot[trial == 1].mean()))
        if (trial == 2).any():
            score += abs(float(n_dot[trial == 2].mean()))
        if score > best_score:
            best_score = score
            best_tags = trial

    if best_tags is not None:
        return best_tags

    # Ultimate fallback: mark the extreme projections along ``up``.
    proj, n_dot = _set_axis(up)
    tags[np.argmax(proj)] = 1
    tags[np.argmin(proj)] = 2
    return tags


def _extract_boundary_faces(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Return the unique boundary triangular faces of a tetrahedral mesh.

    A face belongs to exactly one tetrahedron -> boundary; to two -> internal.
    """
    # All faces of all tets, with vertices sorted per face for comparison.
    faces = np.concatenate(
        [
            tets[:, [0, 1, 2]],
            tets[:, [0, 1, 3]],
            tets[:, [0, 2, 3]],
            tets[:, [1, 2, 3]],
        ]
    )
    faces_sorted = np.sort(faces, axis=1)
    unique_faces, inverse, counts = np.unique(
        faces_sorted, axis=0, return_inverse=True, return_counts=True
    )
    boundary_mask = counts[inverse] == 1
    boundary_sorted = faces_sorted[boundary_mask]

    # Restore outward winding.  For each boundary face find its tetra and flip
    # the winding so the normal points away from the tetra interior.
    face_to_tet: Dict[Tuple[int, ...], int] = {}
    for tet_i, tet in enumerate(tets):
        for face in (
            tet[[0, 1, 2]],
            tet[[0, 1, 3]],
            tet[[0, 2, 3]],
            tet[[1, 2, 3]],
        ):
            key = tuple(sorted(face.tolist()))
            face_to_tet[key] = tet_i

    oriented = []
    for f in boundary_sorted:
        key = tuple(sorted(f.tolist()))
        tet_i = face_to_tet[key]
        tet = tets[tet_i]
        missing = [v for v in tet if v not in f][0]
        fc = f.copy()
        a = nodes[fc[1]] - nodes[fc[0]]
        b = nodes[fc[2]] - nodes[fc[0]]
        n = np.cross(a, b)
        to_missing = nodes[missing] - nodes[fc[0]]
        if n @ to_missing > 0:
            fc[1], fc[2] = fc[2], fc[1]
        oriented.append(fc)
    return np.asarray(oriented, dtype=np.int64)


def _tet_volume(nodes: np.ndarray, tet: np.ndarray) -> float:
    a = nodes[tet[1]] - nodes[tet[0]]
    b = nodes[tet[2]] - nodes[tet[0]]
    c = nodes[tet[3]] - nodes[tet[0]]
    return abs(float(np.dot(a, np.cross(b, c)))) / 6.0


def build_gate_mesh(
    body: Body,
    parent_body: Optional[Body] = None,
    child_bodies: Optional[List[Body]] = None,
    max_volume_mm3: Optional[float] = None,
    max_edge_length_mm: Optional[float] = None,
    up_direction: Optional[Sequence[float]] = None,
) -> GateMesh:
    """Create a tetrahedral mesh for one gate body.

    Parameters
    ----------
    body
        A body whose ``body_type`` is one of the gating types.
    parent_body
        Upstream neighbour in the gating graph.
    child_bodies
        Downstream neighbour(s) in the gating graph.
    max_volume_mm3
        Maximum tetrahedron volume in mm³.  If ``None`` a sensible default based
        on the body bounding box is used.
    max_edge_length_mm
        Optional maximum edge length constraint.
    up_direction
        Fallback ``up`` vector when no topology is supplied.

    Returns
    -------
    GateMesh
    """
    if not _is_gating_body(body):
        raise ValueError(f"Body {body.name} is not a gating body: {body.body_type}")

    mesh = body.mesh
    if mesh is None or len(mesh.faces) == 0:
        raise ValueError(f"Body {body.name} has no surface mesh")

    # Make a clean watertight copy.
    clean = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    clean.merge_vertices(merge_tex=False, merge_norm=False)
    clean.fix_normals()
    clean.fill_holes()

    # Surface decimation keeps the tetrahedral mesh fast while preserving the
    # gate geometry.  Only decimate overly dense gate bodies.  If the optional
    # fast-simplification package is missing, skip decimation and let TetGen
    # deal with the denser input.
    _MAX_GATE_FACES: int = 1200
    _TARGET_GATE_FACES: int = 1000
    if len(clean.faces) > _MAX_GATE_FACES:
        try:
            clean = clean.simplify_quadric_decimation(face_count=_TARGET_GATE_FACES)
            clean.merge_vertices(merge_tex=False, merge_norm=False)
            clean.fix_normals()
            clean.fill_holes()
        except Exception:
            pass

    verts = clean.vertices.astype(float)
    faces = clean.faces.astype(np.int32)  # meshpy expects 0-based

    # meshpy/TetGen is only needed when a 3-D gate mesh is actually built.
    from meshpy.tet import MeshInfo, Options, build

    info = MeshInfo()
    info.set_points(verts.tolist())
    info.set_facets(faces.tolist())

    # TetGen 'p' builds a constrained Delaunay tetrahedralisation of the PLC.
    # No quality/Steiner refinement is used; the surface decimation above
    # already controls the tet count and keeps the solve tractable.
    options = Options("p")
    kwargs: Dict[str, object] = {"options": options}

    # meshpy/TetGen internally restores the previous LC_NUMERIC locale, which can
    # fail on some systems.  Force the C locale around the call and restore it
    # afterwards.
    import locale

    old_locale = locale.setlocale(locale.LC_ALL, None)
    try:
        locale.setlocale(locale.LC_ALL, "C")
        tet = build(info, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"TetGen failed for gate body {body.name}: {exc}") from exc
    finally:
        try:
            locale.setlocale(locale.LC_ALL, old_locale)
        except locale.Error:
            pass

    nodes = np.asarray(tet.points, dtype=float)
    tets = np.asarray(tet.elements, dtype=np.int64)

    # meshpy may produce degenerate tets; filter them out.
    valid = tets.min(axis=1) >= 0
    tets = tets[valid]

    cell_centers = nodes[tets].mean(axis=1)
    cell_volumes = np.array([_tet_volume(nodes, t) for t in tets], dtype=float)

    boundary_faces = _extract_boundary_faces(nodes, tets)
    tags = _classify_boundary_faces(
        nodes,
        boundary_faces,
        body,
        parent_body=parent_body,
        child_bodies=child_bodies,
        up_direction=up_direction,
    )

    boundary_normals = np.zeros((len(boundary_faces), 3), dtype=float)
    boundary_areas = np.zeros(len(boundary_faces), dtype=float)
    for i, f in enumerate(boundary_faces):
        n, area = _face_normal_area(nodes, f)
        boundary_normals[i] = n / (np.linalg.norm(n) + 1e-12)
        boundary_areas[i] = area

    return GateMesh(
        body_name=body.name,
        body_type=body.body_type,
        nodes=nodes,
        tets=tets,
        cell_centers=cell_centers,
        cell_volumes=cell_volumes,
        boundary_faces=boundary_faces,
        boundary_normals=boundary_normals,
        boundary_areas=boundary_areas,
        boundary_tags=tags,
        body=body,
        body_mesh=clean,
    )


def build_all_gate_meshes(
    bodies: List[Body],
    topology: Optional[Dict[str, object]] = None,
    up_direction: Optional[Sequence[float]] = None,
    default_max_volume_mm3: Optional[float] = None,
) -> Dict[str, GateMesh]:
    """Create a ``GateMesh`` for every gating body in ``bodies``.

    ``topology`` is the dictionary returned by ``core.gating._build_gating_topology``.
    If omitted a simple gravity-based classification is used.
    """
    body_by_name = {b.name: b for b in bodies}
    meshes: Dict[str, GateMesh] = {}

    parent_map: Dict[str, Optional[Body]] = {}
    children_map: Dict[str, List[Body]] = {}
    if topology is not None:
        parent_names = topology.get("parent", {})
        children_names = topology.get("children", {})
        for name, parent in parent_names.items():
            parent_map[name] = body_by_name.get(parent) if parent else None
        for name, children in children_names.items():
            children_map[name] = [body_by_name.get(c) for c in children if c in body_by_name]

    for body in bodies:
        if not _is_gating_body(body):
            continue
        parent = parent_map.get(body.name)
        children = children_map.get(body.name, [])
        meshes[body.name] = build_gate_mesh(
            body,
            parent_body=parent,
            child_bodies=children,
            max_volume_mm3=default_max_volume_mm3,
            up_direction=up_direction,
        )
    return meshes
