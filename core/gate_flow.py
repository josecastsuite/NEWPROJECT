"""Hybrid 3-D gate mesh / global voxel Darcy coupling.

For every gating body (sprue, runner, distributor, ingate, ...) a local
anisotropic Darcy–Forchheimer solve is run on a tetrahedral mesh generated
from the original STEP geometry.  Boundary pressures are taken from the
global voxel Darcy field and the flow rate through each gate is taken from the
global gating graph so that local 3-D velocities and the global filling time
remain consistent.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

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
    nx, ny, nz = p.shape
    x = float(origin_mm[0]) + np.arange(nx) * float(dx_mm)
    y = float(origin_mm[1]) + np.arange(ny) * float(dx_mm)
    z = float(origin_mm[2]) + np.arange(nz) * float(dx_mm)
    return RegularGridInterpolator(
        (x, y, z),
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


def _geometric_fallback_result(
    body: Body,
    q_m3_s: float,
    rho_kg_m3: float,
    mu_pa_s: float,
    p_atm_pa: float = 101325.0,
) -> GateFlowResult:
    """Return a minimal ``GateFlowResult`` when gmsh cannot build a tet mesh.

    The throat area is estimated from the oriented bounding-box as the product of
    the two shorter extents; velocity is ``Q / A``.  This lets the gating-node
    contact velocities and reports stay populated even if the 3-D mesher fails on
    a complex gate body.
    """
    q = max(float(q_m3_s), 1e-18)
    if body.mesh is not None and len(body.mesh.faces) > 0:
        extents = np.asarray(body.mesh.bounding_box_oriented.extents, dtype=float)
    else:
        extents = np.ones(3, dtype=float)
    sorted_e = np.sort(extents)
    area_mm2 = max(sorted_e[0] * sorted_e[1], 1e-6)
    length_mm = max(sorted_e[2], 1e-6)
    area_m2 = area_mm2 * 1e-6
    length_m = length_mm * 1e-3
    v = q / area_m2

    # Approximate hydraulic diameter from a rectangular cross-section with
    # side lengths a and b.
    a = max(sorted_e[0] * 1e-3, 1e-6)
    b = max(sorted_e[1] * 1e-3, 1e-6)
    p = 2.0 * (a + b)
    d_h = (4.0 * area_m2) / p if p > 1e-12 else 2.0 * min(a, b)

    re = float(rho_kg_m3 * v * d_h / max(mu_pa_s, 1e-12))
    g_eff = 9.80665
    fr = float(v / (np.sqrt(g_eff * d_h) + 1e-12))

    return GateFlowResult(
        pressures=np.array([p_atm_pa], dtype=float),
        velocities=np.zeros((1, 3), dtype=float),
        velocity_magnitudes=np.array([v], dtype=float),
        cell_volumes=np.array([area_m2 * length_m], dtype=float),
        inlet_flux_m3_s=q,
        outlet_flux_m3_s=q,
        pressure_drop_pa=0.0,
        forchheimer_pressure_drop_pa=0.0,
        total_pressure_drop_pa=0.0,
        section_velocity_m_s=v,
        peak_velocity_m_s=v,
        max_pressure_pa=p_atm_pa,
        min_pressure_pa=p_atm_pa,
        reynolds=np.array([re], dtype=float),
        froude=np.array([fr], dtype=float),
        air_entrainment=np.array([False], dtype=bool),
        wall_normals=np.array([[0.0, 0.0, 1.0]], dtype=float),
        hydraulic_diameter_mm=np.array([d_h * 1000.0], dtype=float),
    )


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
    # meshpy/tet is heavy and can crash on some Windows installs; load it only
    # when the 3-D gate mesh solver is actually requested.
    from core.gate_mesh import build_gate_mesh

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
        gate = None
        result: Optional[GateFlowResult] = None
        try:
            gate = build_gate_mesh(
                body,
                parent_body=parent_map.get(body.name),
                child_bodies=children_map.get(body.name, []),
                max_volume_mm3=max_volume_mm3,
                up_direction=up,
            )
        except Exception as exc:
            print(
                f"[GATE_MESH] {body.name} 3-D mesh failed ({exc}); "
                "using geometric fallback for this body.",
                flush=True,
            )

        if gate is not None:
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

        if result is None:
            result = _geometric_fallback_result(body, q, rho_kg_m3, mu_pa_s, p_atm_pa)

        results[body.name] = result

        # The 3-D mesh computes the total flow rate through the gate.  The
        # per-contact velocity is the mesh flux distributed over each contact
        # (from the gating graph) divided by that contact's geometric area.  This
        # is the physically consistent contact-surface velocity and is what
        # labels/reports display.
        outlet_nodes: List[GatingNode] = []
        inlet_nodes: List[GatingNode] = []
        for node in gating_nodes:
            if node.name is None or " → " not in node.name:
                continue
            parts = [p.strip() for p in node.name.split(" → ")]
            if parts[0] == body.name:
                outlet_nodes.append(node)
            if parts[-1] == body.name:
                inlet_nodes.append(node)

        def _set_contact_velocities(
            nodes: List[GatingNode], total_flux_m3_s: float
        ) -> List[float]:
            total_node_q = sum(
                max(float(n.flow_rate_m3_s or 0.0), 0.0) for n in nodes
            )
            scale = 1.0
            if total_node_q > 1e-18 and total_flux_m3_s > 1e-18:
                scale = float(total_flux_m3_s) / total_node_q
            velocities: List[float] = []
            for n in nodes:
                node_q = max(float(n.flow_rate_m3_s or 0.0), 0.0) * scale
                area_m2 = max(float(n.section_area_cm2 or 0.0) * 1e-4, 1e-18)
                v = node_q / area_m2
                if v > 1e-12:
                    n.max_velocity_m_s = float(v)
                    velocities.append(v)
            return velocities

        # The outlet flux is the mesh-computed, target-enforced total flow rate.
        # Use it for both inlet and outlet contacts; inlet flux from the cell
        # centred field can be noisy near the boundary.
        flux = abs(result.outlet_flux_m3_s)
        if flux <= 1e-18:
            flux = max(
                sum(
                    max(float(n.flow_rate_m3_s or 0.0), 0.0)
                    for n in outlet_nodes + inlet_nodes
                ),
                1e-18,
            )

        outlet_velocities = _set_contact_velocities(outlet_nodes, flux)
        _set_contact_velocities(inlet_nodes, flux)

        if outlet_velocities:
            result.section_velocity_m_s = float(np.mean(outlet_velocities))

    return results, list(gating_nodes)
