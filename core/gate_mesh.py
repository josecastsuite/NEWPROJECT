"""Tetrahedral 3-D mesh generation for gating bodies.

Each gate body (sprue, runner, ingate, distributor, sprue throat, pouring basin,
curufluk) is meshed independently with the gmsh Python API.  The boundary
faces are classified as

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


def _safe_fill_holes(mesh: trimesh.Trimesh, max_iterations: int = 20) -> None:
    """Fill mesh holes iteratively, capping the loop to avoid rare hangs."""
    for _ in range(max_iterations):
        if mesh.is_watertight:
            break
        mesh.fill_holes()


def _decimate_surface(mesh: trimesh.Trimesh, target_faces: int = 1000) -> trimesh.Trimesh:
    """Reduce ``mesh`` to approximately ``target_faces`` faces.

    First tries trimesh's quadric decimation (fast-simplification).  If that
    package is missing or the decimation produces an unusable mesh, fall back to
    a coarse voxelised marching-cubes remesh.  The fallback is robust and does
    not require optional dependencies.
    """
    if len(mesh.faces) <= target_faces:
        return mesh

    try:
        dec = mesh.simplify_quadric_decimation(face_count=target_faces)
        if dec is not None and len(dec.faces) > 0 and len(dec.faces) <= int(target_faces * 1.5):
            return dec
    except Exception:
        pass

    # Fallback: marching cubes on a coarse voxel grid.
    surface_area = float(mesh.area)
    if surface_area <= 1e-12:
        return mesh

    extents = mesh.bounding_box.extents
    min_extent = float(np.asarray(extents).min()) if extents is not None else 0.0

    # Estimate pitch from target face count.  Marching cubes typically produces
    # ~2 * area / pitch^2 faces for a closed surface, so overshoot pitch a bit.
    pitch = max(0.01, (surface_area / (target_faces * 2.0)) ** 0.5)
    if min_extent > 0.0:
        pitch = min(pitch, min_extent / 4.0)

    best = mesh
    best_count = len(mesh.faces)
    for _ in range(5):
        try:
            voxelized = mesh.voxelized(pitch=pitch)
            mc = getattr(voxelized, "marching_cubes", None)
            if mc is not None and len(mc.faces) > 0:
                if abs(len(mc.faces) - target_faces) < abs(best_count - target_faces):
                    best = mc
                    best_count = len(mc.faces)
                if best_count <= target_faces:
                    break
            # Coarsen / refine for next iteration.
            if best_count > target_faces:
                pitch *= 1.4
            else:
                pitch *= 0.8
        except Exception:
            pitch *= 1.5

    if best is not mesh:
        best.merge_vertices(merge_tex=False, merge_norm=False)
        best.fix_normals()
        _safe_fill_holes(best)
    return best


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


def _tet_mesh_gmsh(
    verts: np.ndarray,
    faces: np.ndarray,
    max_volume_mm3: Optional[float] = None,
    max_edge_length_mm: Optional[float] = None,
    model_name: str = "gate",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a tetrahedral mesh with the gmsh Python API.

    Returns ``(nodes, tets)`` where ``nodes`` has shape ``(n_nodes, 3)`` and
    ``tets`` has shape ``(n_tets, 4)`` 0-based indices.  The surface is supplied
    as a closed triangle soup; gmsh creates a volume mesh from it via a
    discrete surface import, surface classification and volume meshing.
    """
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        # Delaunay tetrahedralisation; disable automatic curvature-based
        # refinement so the already-decimated surface is respected.
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        for opt in (
            "Mesh.MeshSizeFromCurvature",
            "Mesh.MeshSizeFromParamPoints",
            "Mesh.MeshSizeFromPoints",
        ):
            try:
                gmsh.option.setNumber(opt, 0.0)
            except Exception:
                pass

        gmsh.model.add(model_name)

        # Import the closed surface as a discrete entity.  gmsh tags are 1-based.
        surf_tag = 1
        gmsh.model.addDiscreteEntity(2, surf_tag)
        n_nodes = len(verts)
        node_tags = np.arange(1, n_nodes + 1, dtype=np.int64)
        gmsh.model.mesh.addNodes(2, surf_tag, node_tags, verts.astype(float).ravel())

        tri_tags = np.arange(1, len(faces) + 1, dtype=np.int64)
        tri_nodes = (faces.astype(np.int64).ravel() + 1)
        gmsh.model.mesh.addElementsByType(surf_tag, 2, tri_tags, tri_nodes)

        # Set a global maximum element size before volume meshing.
        size_max = float("inf")
        if max_edge_length_mm is not None and max_edge_length_mm > 0:
            size_max = max_edge_length_mm
        if max_volume_mm3 is not None and max_volume_mm3 > 0:
            size_from_vol = float(max_volume_mm3) ** (1.0 / 3.0)
            size_max = min(size_max, size_from_vol)
        if np.isfinite(size_max) and size_max > 0:
            gmsh.option.setNumber("Mesh.MeshSizeMax", size_max)

        # Classify the discrete surface, create geometry entities and mesh the
        # enclosed volume.  If the surface is not perfectly closed, gmsh will
        # raise an exception here and the caller (build_gate_mesh) logs it.
        gmsh.model.mesh.classifySurfaces(40.0 * np.pi / 180.0, True, True)
        gmsh.model.mesh.createGeometry()
        gmsh.model.geo.synchronize()

        surfs = [tag for (_, tag) in gmsh.model.getEntities(2)]
        if not surfs:
            raise RuntimeError("gmsh could not classify the gate surface")
        sl = gmsh.model.geo.addSurfaceLoop(surfs)
        vol = gmsh.model.geo.addVolume([sl])
        gmsh.model.geo.synchronize()

        gmsh.model.mesh.generate(3)

        # Retrieve nodes.  getNodes returns 1-based tags and flat coordinate array.
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = node_coords.reshape(-1, 3)
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

        # Retrieve 4-node tetrahedra (element type 4 in gmsh).
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=3, tag=vol)
        tet_type = 4
        if tet_type in elem_types:
            idx = int(elem_types.tolist().index(tet_type))
            raw_tets = elem_node_tags[idx].reshape(-1, 4)
            tets = np.array(
                [[tag_to_idx[int(t)] for t in tet] for tet in raw_tets],
                dtype=np.int64,
            )
        else:
            tets = np.empty((0, 4), dtype=np.int64)

        nodes = np.asarray(node_coords, dtype=float)
        return nodes, tets
    finally:
        gmsh.finalize()


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

    # Make a clean watertight copy and decimate it so TetGen can build a
    # tetrahedral mesh quickly.  The decimation routine has a voxel/marching-cubes
    # fallback so missing ``fast-simplification`` no longer leaves a dense
    # surface for TetGen to hang on.
    _MAX_GATE_FACES: int = 1200
    _TARGET_GATE_FACES: int = 1000
    clean = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    clean.merge_vertices(merge_tex=False, merge_norm=False)
    # Remove duplicate/degenerate faces; these can make gmsh's PLC complain
    # about intersecting segments/facets even when the mesh is "watertight".
    if len(clean.faces) > 0:
        # Drop faces with zero / negative area.
        mask = np.asarray(clean.nondegenerate_faces(), dtype=bool)
        clean.update_faces(mask)
        # Drop duplicate face rows (same vertex set, any order).
        sorted_faces = np.sort(clean.faces, axis=1)
        _, unique_idx = np.unique(sorted_faces, axis=0, return_index=True)
        clean.faces = np.asarray(clean.faces, dtype=np.int64)[np.sort(unique_idx)]
        clean.remove_unreferenced_vertices()
    clean.fix_normals()
    _safe_fill_holes(clean)

    if len(clean.faces) > _MAX_GATE_FACES:
        clean = _decimate_surface(clean, target_faces=_TARGET_GATE_FACES)
        if len(clean.faces) > 0:
            mask = np.asarray(clean.nondegenerate_faces(), dtype=bool)
            clean.update_faces(mask)
            sorted_faces = np.sort(clean.faces, axis=1)
            _, unique_idx = np.unique(sorted_faces, axis=0, return_index=True)
            clean.faces = np.asarray(clean.faces, dtype=np.int64)[np.sort(unique_idx)]
            clean.remove_unreferenced_vertices()
        clean.merge_vertices(merge_tex=False, merge_norm=False)
        clean.fix_normals()
        _safe_fill_holes(clean)

    verts = clean.vertices.astype(float)
    faces = clean.faces.astype(np.int32)

    # 3-B gate mesh is built with gmsh.  This is more robust than meshpy/TetGen
    # on Windows and avoids the fast_simplification / access-violation issues.
    nodes, tets = _tet_mesh_gmsh(
        verts,
        faces,
        max_volume_mm3=max_volume_mm3,
        max_edge_length_mm=max_edge_length_mm,
        model_name=f"gate_{body.name}",
    )

    # gmsh may produce degenerate tets; filter them out.
    valid = (
        (tets[:, 0] >= 0)
        & (tets[:, 1] >= 0)
        & (tets[:, 2] >= 0)
        & (tets[:, 3] >= 0)
        & (tets[:, 0] < len(nodes))
        & (tets[:, 1] < len(nodes))
        & (tets[:, 2] < len(nodes))
        & (tets[:, 3] < len(nodes))
    )
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
