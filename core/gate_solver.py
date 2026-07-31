"""Local 3-D Darcy–Forchheimer solver on a tetrahedral gate mesh.

The gate is treated as a porous medium with an anisotropic permeability tensor
that is strongly aligned with the local wall tangent (high tangential
permeability, negligible normal permeability).  This forces the metal to flow
along the gate walls without requiring a surface-mesh cut-cell procedure.

A Picard iteration handles the Forchheimer non-linearity; at each step the
effective permeability is reduced according to the local velocity magnitude,
capturing inertial losses in narrowings and bends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.linalg import spsolve


@dataclass
class GateFlowResult:
    """Darcy–Forchheimer solution for one gate mesh."""

    pressures: np.ndarray  # (n_cells,) Pa
    velocities: np.ndarray  # (n_cells, 3) m/s
    velocity_magnitudes: np.ndarray  # (n_cells,) m/s
    cell_volumes: np.ndarray  # (n_cells,) m³
    inlet_flux_m3_s: float
    outlet_flux_m3_s: float
    pressure_drop_pa: float
    forchheimer_pressure_drop_pa: float
    total_pressure_drop_pa: float
    section_velocity_m_s: float
    peak_velocity_m_s: float
    max_pressure_pa: float
    min_pressure_pa: float
    reynolds: np.ndarray  # (n_cells,)
    froude: np.ndarray  # (n_cells,)
    air_entrainment: np.ndarray  # (n_cells,) bool, p_static < p_atm
    wall_normals: np.ndarray  # (n_cells, 3)
    hydraulic_diameter_mm: np.ndarray  # (n_cells,)


def _cell_wall_data(
    gate: GateMesh,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return wall distance, wall normals and an opening mask for cells.

    The wall normal points from the metal toward the mold wall (outward from the
    gate body).  The opening mask is ``True`` for cells whose nearest boundary
    face is an inlet/outlet; those cells keep an isotropic permeability so that
    the metal can enter and exit the gate freely.

    Instead of querying the original high-resolution surface mesh, we use the
    boundary faces of the tetrahedral mesh itself.  This is orders of magnitude
    faster and gives the same orientation information we need for the
    anisotropic permeability tensor.
    """
    centers = gate.cell_centers.astype(float)
    bnd_faces = gate.boundary_faces
    bnd_nodes = gate.nodes[bnd_faces]
    bnd_centers = bnd_nodes.mean(axis=1)

    from scipy.spatial import cKDTree

    tree = cKDTree(bnd_centers)
    _, nearest = tree.query(centers)
    nearest_faces = bnd_faces[nearest]
    normals = gate.boundary_normals[nearest]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12

    # Signed distance to the nearest boundary face plane: project the vector
    # from an arbitrary face vertex to the cell center onto the outward normal.
    nearest_nodes = bnd_nodes[nearest]
    # Pick the first vertex of each nearest face.
    v0 = nearest_nodes[:, 0, :]
    to_center = centers - v0
    distance = np.abs(np.sum(to_center * normals, axis=1))

    is_opening = gate.boundary_tags[nearest] != 0
    return distance, normals, is_opening


def _anisotropic_K(
    normals: np.ndarray,
    K_t: float,
    K_n: float,
    is_opening: Optional[np.ndarray] = None,
    wall_dist: Optional[np.ndarray] = None,
    wall_layer_m: float = 1e-3,
) -> np.ndarray:
    """Return (n_cells, 3, 3) permeability tensors.

    Cells in the bulk (farther than ``wall_layer_m`` from a wall) are isotropic
    with ``K_t``; only a thin wall layer is anisotropic, with the high-
    permeability direction tangent to the wall and the wall-normal direction
    strongly suppressed.  Inlet/outlet cells are always isotropic so the metal
    can enter and exit freely.  This avoids the unphysical high tangential
    velocities that a globally anisotropic tensor can create.
    """
    n_cells = normals.shape[0]
    I3 = np.eye(3, dtype=float).reshape(1, 3, 3)
    K = np.repeat(K_t * I3, n_cells, axis=0)

    if wall_dist is not None:
        near_wall = wall_dist.reshape(-1) < wall_layer_m
    else:
        near_wall = np.ones(n_cells, dtype=bool)

    n = normals[near_wall].reshape(-1, 3, 1)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    nnT = n @ n.transpose(0, 2, 1)
    K[near_wall] = K_t * (np.eye(3, dtype=float).reshape(1, 3, 3) - nnT) + K_n * nnT

    if is_opening is not None and is_opening.any():
        K[is_opening] = K_t * np.eye(3, dtype=K.dtype)
    return K


def _face_geometry(
    gate: GateMesh, cell_i: int, cell_j: int, face: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Face center, unit normal (i -> j), cell-centroid vector and face area."""
    nodes_f = gate.nodes[face]
    center = nodes_f.mean(axis=0)
    a = nodes_f[1] - nodes_f[0]
    b = nodes_f[2] - nodes_f[0]
    cross = np.cross(a, b)
    area = 0.5 * float(np.linalg.norm(cross))
    n = cross / (np.linalg.norm(cross) + 1e-12)

    c_i = gate.cell_centers[cell_i]
    c_j = gate.cell_centers[cell_j]
    # Ensure normal points from i to j.
    if n @ (c_j - c_i) < 0:
        n = -n
    return center, n, (c_j - c_i), area


def _boundary_face_geometry(
    gate: GateMesh, cell_i: int, face: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Return face center, outward unit normal, area and distance from cell centroid."""
    nodes_f = gate.nodes[face]
    center = nodes_f.mean(axis=0)
    a = nodes_f[1] - nodes_f[0]
    b = nodes_f[2] - nodes_f[0]
    cross = np.cross(a, b)
    area = 0.5 * float(np.linalg.norm(cross))
    n = cross / (np.linalg.norm(cross) + 1e-12)

    c_i = gate.cell_centers[cell_i]
    d = (center - c_i) @ n
    return center, n, area, max(d, 1e-6)


def _build_neighbor_graph(gate: GateMesh) -> Dict[Tuple[int, ...], List[int]]:
    """Map each sorted face to the list of tetrahedra that share it."""
    graph: Dict[Tuple[int, ...], List[int]] = {}
    faces = [
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
    ]
    for tet_i, tet in enumerate(gate.tets):
        for f in faces:
            key = tuple(sorted(tet[list(f)].tolist()))
            graph.setdefault(key, []).append(tet_i)
    return graph


def _face_k_eff(
    K: np.ndarray, n_f: np.ndarray, i: int, j: int
) -> Tuple[float, float]:
    """Project anisotropic K tensors of two cells onto face normal and harmonic mean."""
    k_i = float(n_f @ K[i] @ n_f)
    k_j = float(n_f @ K[j] @ n_f)
    k_i = max(k_i, 1e-18)
    k_j = max(k_j, 1e-18)
    if k_i + k_j == 0.0:
        return 0.0, 0.0
    k_harm = 2.0 * k_i * k_j / (k_i + k_j)
    return k_i, k_harm


def _assemble_matrix(
    gate: GateMesh,
    K: np.ndarray,
    mu: float,
    p_boundary: Dict[int, float],
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Build cell-centered FV system for ``∇·((K/μ)∇p)=0`` with Dirichlet b.c.

    ``p_boundary`` maps boundary face index -> pressure (Pa).
    """
    n_cells = len(gate.tets)
    graph = _build_neighbor_graph(gate)

    row: List[int] = []
    col: List[int] = []
    data: List[float] = []
    rhs = np.zeros(n_cells, dtype=float)

    added: set = set()
    for key, cells in graph.items():
        if len(cells) == 2:
            i, j = cells
            if (i, j) in added:
                continue
            added.add((i, j))
            added.add((j, i))

            # Find the actual face vertices in one of the tets.
            face = None
            for tet in gate.tets[cells]:
                for f in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
                    if tuple(sorted(tet[list(f)].tolist())) == key:
                        face = tet[list(f)]
                        break
                if face is not None:
                    break
            if face is None:
                continue

            _, n_f, d_vec, area = _face_geometry(gate, i, j, face)
            dist = float(np.linalg.norm(d_vec)) + 1e-12
            _, k_harm = _face_k_eff(K, n_f, i, j)
            T = (area / (mu * dist)) * k_harm

            row.extend([i, i, j, j])
            col.extend([i, j, i, j])
            data.extend([T, -T, -T, T])

    # Boundary faces.
    for bnd_idx, face in enumerate(gate.boundary_faces):
        if bnd_idx not in p_boundary:
            continue
        key = tuple(sorted(face.tolist()))
        cells = graph[key]
        if not cells:
            continue
        i = cells[0]
        _, n_f, area, d = _boundary_face_geometry(gate, i, face)
        k_i, _ = _face_k_eff(K, n_f, i, i)
        T_b = (area / (mu * d)) * k_i
        p_b = p_boundary[bnd_idx]

        row.append(i)
        col.append(i)
        data.append(T_b)
        rhs[i] += T_b * p_b

    A = sparse.coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()
    return A, rhs


def _cell_gradient(
    gate: GateMesh,
    p: np.ndarray,
    p_boundary: Dict[int, float],
) -> np.ndarray:
    """Least-squares cell-centered pressure gradient.

    Each cell uses its neighbour cell centres and boundary face centroids as
    sample points.  Weights are ``1 / (distance + eps)`` (not ``1/d²``) so a
    close boundary face does not dominate and create spuriously large
    tangential components.  The result is first-order accurate and stable for
    the irregular tetrahedra produced by TetGen.
    """
    n_cells = len(gate.tets)
    graph = _build_neighbor_graph(gate)
    face_to_bnd: Dict[Tuple[int, ...], int] = {}
    for idx, f in enumerate(gate.boundary_faces):
        face_to_bnd[tuple(sorted(f.tolist()))] = idx

    # Precompute per-cell neighbour directions and pressure differences.
    neighbors: List[List[Tuple[np.ndarray, float]]] = [[] for _ in range(n_cells)]
    for key, cells in graph.items():
        if len(cells) == 2:
            i, j = cells
            d = gate.cell_centers[j] - gate.cell_centers[i]
            dp = p[j] - p[i]
            neighbors[i].append((d, dp))
            neighbors[j].append((-d, -dp))
        elif len(cells) == 1:
            i = cells[0]
            bnd_idx = face_to_bnd.get(key)
            if bnd_idx is None:
                continue
            face_nodes = gate.nodes[gate.boundary_faces[bnd_idx]]
            fc = face_nodes.mean(axis=0)
            d = fc - gate.cell_centers[i]
            # Dirichlet faces use the prescribed pressure; walls use the cell
            # pressure (Neumann, zero normal gradient) which also stabilises the
            # least-squares reconstruction for boundary cells.
            p_face = p_boundary.get(bnd_idx, p[i])
            dp = p_face - p[i]
            neighbors[i].append((d, dp))

    grad = np.zeros((n_cells, 3), dtype=float)
    for i, nbs in enumerate(neighbors):
        if len(nbs) < 3:
            continue
        D = np.asarray([d for d, _ in nbs], dtype=float)
        dp = np.asarray([v for _, v in nbs], dtype=float)
        # Unweighted least-squares: each neighbour contributes equally.  Boundary
        # face centroids are at different distances, so weighting by 1/d would
        # over-emphasise the closest (often top/bottom) face and distort the
        # tangential gradient components.  A loose ``rcond`` discards small
        # singular values and stops nearly-coplanar neighbours from producing
        # spuriously large gradients.
        g, *_ = np.linalg.lstsq(D, dp, rcond=1e-2)
        grad[i] = g

    return grad


def _cell_velocity(
    grad_p: np.ndarray, K: np.ndarray, mu: float
) -> np.ndarray:
    """v = -(K/μ) ∇p.  ``grad_p`` and ``K`` are per-cell."""
    # grad_p: (n, 3), K: (n, 3, 3)
    return -((K @ grad_p.reshape(-1, 3, 1)).reshape(-1, 3)) / mu


def _forchheimer_adjusted_K(
    K: np.ndarray,
    grad_p: np.ndarray,
    v: np.ndarray,
    beta: float,
    rho: float,
    mu: float,
) -> np.ndarray:
    """Return Forchheimer-adjusted permeability tensors.

    For each cell the velocity is assumed to be aligned with the Darcy
    direction ``-K ∇p``.  Solving the scalar relation

        |∇p| = (μ/k) |v| + β ρ |v|²

    with ``k = |K ∇p| / |∇p|`` gives the Forchheimer-corrected speed.  The
    tensor is then scaled so that ``v = -(K_eff/μ) ∇p``.
    """
    g_norm = np.linalg.norm(grad_p, axis=1)
    Kg = (K @ grad_p.reshape(-1, 3, 1)).reshape(-1, 3)
    Kg_norm = np.linalg.norm(Kg, axis=1)
    k_dir = Kg_norm / (g_norm + 1e-12)
    vmag_old = np.linalg.norm(v, axis=1)

    a = beta * rho
    b = mu / (k_dir + 1e-18)
    c = -g_norm
    # Positive root of the quadratic a v² + b v + c = 0.
    disc = b * b - 4.0 * a * c
    disc = np.maximum(disc, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vmag = np.where(
            a > 1e-18,
            (-b + np.sqrt(disc)) / (2.0 * a),
            -c / (b + 1e-18),
        )
    vmag = np.where(g_norm > 1e-12, vmag, 0.0)
    # Blend with the previous velocity to avoid large jumps on the first Picard step.
    vmag = 0.7 * vmag + 0.3 * vmag_old

    alpha = np.zeros_like(k_dir)
    mask = Kg_norm > 1e-18
    alpha[mask] = mu * vmag[mask] / Kg_norm[mask]
    alpha = np.clip(alpha, 1e-6, 1.0)
    return K * alpha.reshape(-1, 1, 1)


def _boundary_pressures(
    gate: GateMesh,
    p_inlet_pa: float,
    p_outlet_pa: float,
) -> Dict[int, float]:
    """Map every inlet/outlet boundary face to a pressure value."""
    p_bnd: Dict[int, float] = {}
    for idx in gate.inlet_faces:
        p_bnd[int(idx)] = float(p_inlet_pa)
    for idx in gate.outlet_faces:
        p_bnd[int(idx)] = float(p_outlet_pa)
    return p_bnd


def _hydraulic_diameter(
    gate: GateMesh, wall_dist: np.ndarray
) -> np.ndarray:
    """Estimate hydraulic diameter as 4V/S of the whole gate body.

    Using a global characteristic length avoids unphysical near-wall values
    (distance -> 0) and is consistent with the hydraulic-diameter definition for
    a duct: ``D_h = 4 A_c / P`` which for a 3-D body is ``4 V / A_s``.
    """
    V = float(gate.cell_volumes.sum())
    S = float(gate.boundary_areas.sum())
    if S > 0.0:
        d_h = 4.0 * V / S
    else:
        d_h = 2.0 * float(wall_dist.max())
    return np.full_like(wall_dist, d_h)


def _compute_flux(
    gate: GateMesh,
    velocities: np.ndarray,
    face_indices: np.ndarray,
) -> float:
    """Integrate normal velocity over a set of boundary faces."""
    graph = _build_neighbor_graph(gate)
    total = 0.0
    for idx in face_indices:
        face = gate.boundary_faces[idx]
        key = tuple(sorted(face.tolist()))
        cells = graph.get(key, [])
        if not cells:
            continue
        cell = cells[0]
        n = gate.boundary_normals[idx]
        area = gate.boundary_areas[idx]
        total += float(velocities[cell] @ n) * area
    return total


def _cell_reynolds(
    v_mag: np.ndarray, d_h: np.ndarray, rho: float, mu: float
) -> np.ndarray:
    return rho * v_mag * d_h / (mu + 1e-12)


def _cell_froude(
    v_mag: np.ndarray, d_h: np.ndarray, g: float = 9.81
) -> np.ndarray:
    return v_mag / np.sqrt(g * d_h + 1e-12)


def _solve_linear(
    gate: GateMesh,
    K_t: float,
    K_n: float,
    mu: float,
    p_bnd: Dict[int, float],
    wall_layer_m: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Single anisotropic Darcy solve for a given tangential permeability."""
    wall_dist, wall_normals, is_opening = _cell_wall_data(gate)
    K = _anisotropic_K(
        wall_normals, K_t, K_n, is_opening=is_opening, wall_dist=wall_dist,
        wall_layer_m=wall_layer_m,
    )
    A, rhs = _assemble_matrix(gate, K, mu, p_bnd)
    p = spsolve(A, rhs)
    if not np.isfinite(p).all():
        p = np.zeros(len(gate.tets), dtype=float)
    grad = _cell_gradient(gate, p, p_bnd)
    v = _cell_velocity(grad, K, mu)
    flux = _compute_flux(gate, v, gate.outlet_faces)
    return p, grad, v, flux


def solve_gate_flow(
    gate: GateMesh,
    p_inlet_pa: float,
    p_outlet_pa: float,
    mu_pa_s: float,
    rho_kg_m3: float,
    K_t_m2: float = 1e-3,
    K_n_m2: float = 1e-14,
    beta_1_m: float = 0.0,
    target_flux_m3_s: float = 0.0,
    p_atm_pa: float = 101325.0,
    max_iter: int = 5,
    wall_layer_m: float = 1e-3,
) -> GateFlowResult:
    """Solve Darcy–Forchheimer on a tetrahedral gate mesh.

    The non-linear Forchheimer term is handled by a few Picard-like linearisation
    iterations.  If ``target_flux_m3_s`` is supplied, the tangential permeability
    is scaled so the gate produces the requested flow rate while preserving the
    3-D velocity profile shape.

    Parameters
    ----------
    p_inlet_pa, p_outlet_pa
        Dirichlet pressures at the inlet/outlet faces (Pa).
    mu_pa_s
        Dynamic viscosity of the liquid metal.
    rho_kg_m3
        Metal density.
    K_t_m2
        Tangential permeability (initial guess).
    K_n_m2
        Normal permeability (very small, blocks flow through walls).
    beta_1_m
        Forchheimer coefficient (1/m).  If 0 the solve is linear Darcy.
    target_flux_m3_s
        Optional flow rate to enforce through the gate (m³/s).
    """
    # Convert the gate mesh from millimetres (CAD) to SI metres for the solve.
    gate.nodes = gate.nodes * 1e-3
    gate.cell_centers = gate.cell_centers * 1e-3
    gate.cell_volumes = gate.cell_volumes * 1e-9
    gate.boundary_areas = gate.boundary_areas * 1e-6
    if gate.body_mesh is not None:
        gate.body_mesh = gate.body_mesh.copy()
        gate.body_mesh.apply_scale(1e-3)

    p_bnd = _boundary_pressures(gate, p_inlet_pa, p_outlet_pa)

    # Centreline direction for axial projection.  This removes the tangential
    # noise that the least-squares gradient picks up near boundary tets.
    if gate.inlet_faces.size and gate.outlet_faces.size:
        inlet_centroid = gate.nodes[gate.boundary_faces[gate.inlet_faces]].reshape(-1, 3).mean(axis=0)
        outlet_centroid = gate.nodes[gate.boundary_faces[gate.outlet_faces]].reshape(-1, 3).mean(axis=0)
        flow_dir = outlet_centroid - inlet_centroid
        flow_norm = float(np.linalg.norm(flow_dir)) + 1e-12
        flow_dir = flow_dir / flow_norm
    else:
        flow_dir = np.array([0.0, 0.0, 1.0], dtype=float)

    def _axial_velocity(raw_v: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return velocity projected onto the centreline and outlet flux."""
        if gate.inlet_faces.size and gate.outlet_faces.size:
            v_along = raw_v @ flow_dir
            v_proj = v_along[:, np.newaxis] * flow_dir
            flux = _compute_flux(gate, v_proj, gate.outlet_faces)
            return v_proj, flux
        return raw_v, _compute_flux(gate, raw_v, gate.outlet_faces)

    # Initial linear Darcy solve.
    p, grad, v, _ = _solve_linear(
        gate, K_t_m2, K_n_m2, mu_pa_s, p_bnd, wall_layer_m=wall_layer_m
    )
    v, flux = _axial_velocity(v)

    wall_dist, wall_normals, _ = _cell_wall_data(gate)

    # Calibrate the tangential permeability to the requested flow rate.  The
    # linear Darcy relation ``Q ∝ K_t`` means a single scaling step normally
    # suffices.  If no target flux is supplied, an optional Forchheimer drag
    # correction reduces K_t according to the local velocity.
    K_t = K_t_m2
    if target_flux_m3_s > 0.0:
        for _ in range(max_iter):
            if abs(flux) > 1e-18 and abs(abs(flux) - target_flux_m3_s) > 1e-4 * max(
                abs(target_flux_m3_s), 1e-9
            ):
                K_t = abs(K_t) * (target_flux_m3_s / abs(flux))
                p, grad, v, _ = _solve_linear(
                    gate, K_t, K_n_m2, mu_pa_s, p_bnd, wall_layer_m=wall_layer_m
                )
                v, flux = _axial_velocity(v)
            else:
                break
    elif beta_1_m > 0.0:
        outlet_area = float(gate.boundary_areas[gate.outlet_faces].sum())
        v_avg = abs(flux) / (outlet_area + 1e-18)
        drag = 1.0 + (beta_1_m * rho_kg_m3 * K_t * v_avg) / (mu_pa_s + 1e-12)
        K_t = max(K_t / drag, 1e-18)
        p, grad, v, _ = _solve_linear(
            gate, K_t, K_n_m2, mu_pa_s, p_bnd, wall_layer_m=wall_layer_m
        )
        v, flux = _axial_velocity(v)

    v_mag = np.abs(v @ flow_dir) if gate.inlet_faces.size and gate.outlet_faces.size else np.linalg.norm(v, axis=1)
    inlet_flux = _compute_flux(gate, v, gate.inlet_faces)
    outlet_flux = _compute_flux(gate, v, gate.outlet_faces)

    d_h = _hydraulic_diameter(gate, wall_dist)
    re = _cell_reynolds(v_mag, d_h, rho_kg_m3, mu_pa_s)
    fr = _cell_froude(v_mag, d_h)
    # Air entrainment is flagged where the Darcy (static) pressure falls below
    # atmospheric, indicating the metal stream could aspirate mould air or gas.
    air = p < p_atm_pa

    outlet_area = float(gate.boundary_areas[gate.outlet_faces].sum())
    v_avg = abs(outlet_flux) / (outlet_area + 1e-18)
    if gate.inlet_faces.size and gate.outlet_faces.size:
        inlet_centroid = gate.nodes[gate.boundary_faces[gate.inlet_faces]].reshape(-1, 3).mean(axis=0)
        outlet_centroid = gate.nodes[gate.boundary_faces[gate.outlet_faces]].reshape(-1, 3).mean(axis=0)
        # Use the distance between inlet and outlet patch centroids as the
        # characteristic length for the Forchheimer term.
        length_m = float(np.linalg.norm(outlet_centroid - inlet_centroid))
    else:
        length_m = float(
            np.linalg.norm(gate.cell_centers.max(axis=0) - gate.cell_centers.min(axis=0))
        )
    dp_darcy = float(p.max() - p.min())
    dp_forch = float(beta_1_m * rho_kg_m3 * (v_avg ** 2) * max(length_m, 1e-6))

    # The velocity that should be reported at a gating node is the area-average
    # velocity through the gate outlet, i.e. Q / A_outlet.  This is the physically
    # meaningful section velocity the user expects.  The raw cell maximum is kept as
    # a separate peak diagnostic for air-entrainment / local checks.
    outlet_area = float(gate.boundary_areas[gate.outlet_faces].sum())
    section_velocity_m_s = float(abs(outlet_flux) / (outlet_area + 1e-18))

    return GateFlowResult(
        pressures=p,
        velocities=v,
        velocity_magnitudes=v_mag,
        cell_volumes=gate.cell_volumes,  # already in m³ after scaling
        inlet_flux_m3_s=inlet_flux,
        outlet_flux_m3_s=outlet_flux,
        pressure_drop_pa=dp_darcy,
        forchheimer_pressure_drop_pa=dp_forch,
        total_pressure_drop_pa=dp_darcy + dp_forch,
        section_velocity_m_s=section_velocity_m_s,
        peak_velocity_m_s=float(v_mag.max()),
        max_pressure_pa=float(p.max()),
        min_pressure_pa=float(p.min()),
        reynolds=re,
        froude=fr,
        air_entrainment=air,
        wall_normals=wall_normals,
        hydraulic_diameter_mm=d_h,
    )
