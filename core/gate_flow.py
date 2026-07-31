"""Hybrid 3-D gate mesh / global voxel Darcy coupling.

For every gating body (sprue, runner, distributor, ingate, ...) a local
anisotropic Darcy–Forchheimer solve is run on a tetrahedral mesh generated
from the original STEP geometry.  Boundary pressures are taken from the
global voxel Darcy field and the flow rate through each gate is taken from the
global gating graph so that local 3-D velocities and the global filling time
remain consistent.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from core.gate_mesh import GateMesh, build_gate_mesh
from core.gate_solver import GateFlowResult, solve_gate_flow
from core.gating import _build_gating_topology
from core.types import Body, BodyType, GatingNode


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


def _body_by_name(bodies: List[Body]) -> Dict[str, Body]:
    out: Dict[str, Body] = {}
    for b in bodies:
        if b.name in out:
            continue
        out[b.name] = b
    return out


def _target_flux_per_body(
    gating_nodes: List[GatingNode],
    bodies: List[Body],
    default_q_m3_s: float = 0.0,
) -> Dict[str, float]:
    """Return the total flow rate (m³/s) that should pass through each body.

    Flows are accumulated from the global gating node graph: every node whose
    upstream body is ``name`` contributes its ``flow_rate_m3_s``.
    """
    q: Dict[str, float] = {}
    for node in gating_nodes:
        if node.name is None or " → " not in node.name:
            continue
        up = node.name.split(" → ")[0]
        q[up] = q.get(up, 0.0) + max(float(node.flow_rate_m3_s or 0.0), 0.0)

    # Sources that do not appear as upstream in a node still need the default Q.
    for b in bodies:
        if b.body_type in _GATING_TYPES and b.name not in q:
            q[b.name] = max(default_q_m3_s, 0.0)
    return q


def _pressure_interpolator(
    p: np.ndarray,
    origin_mm: np.ndarray,
    dx_mm: float,
) -> RegularGridInterpolator:
    """Build a trilinear interpolator for the dimensionless Darcy pressure."""
    nz, ny, nx = p.shape
    z = float(origin_mm[0]) + np.arange(nz) * float(dx_mm)
    y = float(origin_mm[1]) + np.arange(ny) * float(dx_mm)
    x = float(origin_mm[2]) + np.arange(nx) * float(dx_mm)
    return RegularGridInterpolator(
        (z, y, x),
        p.astype(float),
        bounds_error=False,
        fill_value=0.0,
        method="linear",
    )


def _face_centroid(gate: GateMesh, face_indices: np.ndarray) -> np.ndarray:
    """Centroid of a set of boundary faces (indices into gate.nodes)."""
    if face_indices.size == 0:
        return gate.cell_centers.mean(axis=0)
    pts = gate.nodes[gate.boundary_faces[face_indices]]
    return pts.reshape(-1, 3).mean(axis=0)


def _boundary_pressure(
    gate: GateMesh,
    interp: RegularGridInterpolator,
    pressure_drop_pa: float,
    p_atm_pa: float,
    face_indices: np.ndarray,
) -> float:
    """Interpolate absolute pressure at a boundary patch of the gate mesh."""
    c = _face_centroid(gate, face_indices)
    p_dim = float(interp(c.reshape(1, 3))[0])
    return p_atm_pa + float(p_dim) * float(pressure_drop_pa)


def _beta_from_geometry(
    gate: GateMesh,
    base_beta_1_m: float = 1.0,
    narrow_factor: float = 1.0,
) -> float:
    """Estimate a Forchheimer coefficient from local cross-section variation.

    Narrow regions receive a larger β because inertial losses scale with
    1/hydraulic-diameter.  The gate outlet area is used as the reference.
    """
    outlet_area = float(gate.boundary_areas[gate.outlet_faces].sum())
    if outlet_area <= 1e-18:
        return base_beta_1_m
    a_ratio = float(gate.boundary_areas.max()) / outlet_area
    return base_beta_1_m * (1.0 + narrow_factor * (a_ratio - 1.0))


def solve_gate_flows(
    bodies: List[Body],
    grid: np.ndarray,
    origin_mm: np.ndarray,
    dx_mm: float,
    p_dim: np.ndarray,
    pressure_drop_pa: float,
    mu_pa_s: float,
    rho_kg_m3: float,
    gating_nodes: List[GatingNode],
    Q_user_m3_s: float = 0.0,
    p_atm_pa: float = 101325.0,
    K_t_m2: float = 1e-6,
    K_n_m2: float = 1e-14,
    beta_1_m: float = 0.0,
    max_volume_mm3: Optional[float] = None,
    wall_layer_m: float = 1e-3,
    gravity_vector: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, GateFlowResult], List[GatingNode]]:
    """Solve local 3-D Darcy–Forchheimer on all gating bodies.

    Returns
    -------
    results
        ``{body_name: GateFlowResult}`` for every successfully meshed gate.
    updated_gating_nodes
        The same ``gating_nodes`` list with velocities/areas/flow rates
        overwritten for gates where a 3-D mesh solve was performed.
    """
    if not bodies or not gating_nodes:
        return {}, list(gating_nodes)

    topology = _build_gating_topology(bodies)
    body_map = _body_by_name(bodies)
    q_target = _target_flux_per_body(gating_nodes, bodies, Q_user_m3_s)
    interp = _pressure_interpolator(p_dim, origin_mm, dx_mm)

    results: Dict[str, GateFlowResult] = {}

    parent_map: Dict[str, Optional[Body]] = {}
    children_map: Dict[str, List[Body]] = {}
    parent_names = topology.get("parent", {})
    children_names = topology.get("children", {})
    for name, parent in parent_names.items():
        parent_map[name] = body_map.get(parent)
    for name, children in children_names.items():
        children_map[name] = [body_map.get(c) for c in children if c in body_map]

    # ``up_direction`` is the direction opposite to gravity and is used as a
    # fallback to orient inlet (top) / outlet (bottom) faces.
    if gravity_vector is not None:
        up = -np.asarray(gravity_vector, dtype=float)
        norm = float(np.linalg.norm(up)) + 1e-12
        if norm > 0.0:
            up = up / norm
        else:
            up = np.array([0.0, 0.0, 1.0])
    else:
        up = np.array([0.0, 0.0, 1.0])

    for body in bodies:
        if body.body_type not in _GATING_TYPES:
            continue
        q = q_target.get(body.name, 0.0)
        if q <= 0.0:
            q = Q_user_m3_s
        print(f"[GATE_MESH] processing {body.name} type={body.body_type} q={q}", flush=True)
        try:
            gate = build_gate_mesh(
                body,
                parent_body=parent_map.get(body.name),
                child_bodies=children_map.get(body.name, []),
                max_volume_mm3=max_volume_mm3,
                up_direction=up,
            )
        except Exception as exc:
            print(f"[GATE_MESH] build failed {body.name}: {exc}", flush=True)
            continue

        p_in = _boundary_pressure(
            gate, interp, pressure_drop_pa, p_atm_pa, gate.inlet_faces
        )
        p_out = _boundary_pressure(
            gate, interp, pressure_drop_pa, p_atm_pa, gate.outlet_faces
        )
        # Ensure a positive driving pressure in spite of interpolation noise.
        if p_in <= p_out:
            p_in = max(p_in, p_out + 1.0)

        beta = beta_1_m if beta_1_m > 0.0 else _beta_from_geometry(gate)

        try:
            result = solve_gate_flow(
                gate,
                p_inlet_pa=p_in,
                p_outlet_pa=p_out,
                mu_pa_s=mu_pa_s,
                rho_kg_m3=rho_kg_m3,
                K_t_m2=K_t_m2,
                K_n_m2=K_n_m2,
                beta_1_m=beta,
                target_flux_m3_s=q,
                p_atm_pa=p_atm_pa,
                wall_layer_m=wall_layer_m,
            )
        except Exception as exc:
            print(f"[GATE_MESH] solve failed {body.name}: {exc}", flush=True)
            continue

        results[body.name] = result

        # Update gating nodes that leave this gate body.
        total_downstream_q = 0.0
        for node in gating_nodes:
            if node.name is None or " → " not in node.name:
                continue
            up_name = node.name.split(" → ")[0]
            if up_name != body.name:
                continue
            total_downstream_q += max(float(node.flow_rate_m3_s or 0.0), 0.0)

        outlet_flux = max(abs(result.outlet_flux_m3_s), 1e-18)
        for node in gating_nodes:
            if node.name is None or " → " not in node.name:
                continue
            up_name = node.name.split(" → ")[0]
            if up_name != body.name:
                continue
            frac = 1.0
            if total_downstream_q > 1e-18:
                frac = max(float(node.flow_rate_m3_s or 0.0), 0.0) / total_downstream_q
            node_q = outlet_flux * frac
            area_m2 = max(float(node.section_area_cm2 or 0.0) * 1e-4, 1e-18)
            node.flow_rate_m3_s = float(node_q)
            node.velocity_m_s = float(node_q / area_m2)

    return results, list(gating_nodes)
