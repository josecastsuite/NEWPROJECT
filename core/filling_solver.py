"""3-D Darcy-flow based filling solver for JoseCast.

This module computes a pressure-driven, gravity-assisted velocity field through
the gating system and casting cavity on the existing voxel grid.  It is a
lightweight CFD-style approximation (Darcy / Hele-Shaw) that respects the
actual 3-D geometry and continuity, without the installation and setup burden
of a full OpenFOAM/VOF pipeline.

High-level usage:

    result = solve_filling_flow(
        grid=result.grid,
        origin=result.origin_mm,
        dx=result.dx_mm,
        casting_params=casting_params,
        alloy=alloy,
    )

`FillingResult` contains section-averaged velocities, the ingate contact
velocity and an optional per-voxel fill-time estimate.
"""
import heapq
import os
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pyvista as pv
import trimesh
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
from scipy.sparse import linalg as spla

from core.config import FlowConfig
from core.gate_flow import solve_gate_flows
from core.materials import MOLDS, MoldMaterial
from core.types import Body, BodyType, FillingResult, GatingNode, GatingVelocityError
from core.voxelizer import build_voxel_grid, compute_face_fractions

_FLOW_CFG = FlowConfig()


def _downsample_grid(
    grid: np.ndarray,
    origin: np.ndarray,
    dx: float,
    max_cells: int = _FLOW_CFG.max_solver_cells,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Downsample a body-type grid with nearest-neighbour interpolation.

    The solver memory/time is driven by the number of cavity (non-empty) cells,
    so the limit is applied to the cavity count rather than the dense grid size.
    """
    nx, ny, nz = grid.shape
    cavity = grid != BodyType.EMPTY
    n_cavity = int(cavity.sum())
    if n_cavity <= max_cells:
        return grid.copy(), origin.copy(), dx

    # Choose an integer factor that brings the cavity count below the limit.
    factor = int(np.ceil((n_cavity / max_cells) ** (1.0 / 3.0)))
    factor = max(2, factor)

    new_shape = (
        max(1, nx // factor),
        max(1, ny // factor),
        max(1, nz // factor),
    )
    zoom = (new_shape[0] / nx, new_shape[1] / ny, new_shape[2] / nz)
    grid_c = ndimage.zoom(grid, zoom, order=0, mode="nearest")
    # Nearest-neighbour body IDs are preserved.
    dx_c = dx * (nx / new_shape[0])
    origin_c = origin.copy()
    # Origin is kept at the same physical corner; zoom handles sampling.
    return grid_c.astype(grid.dtype), origin_c, dx_c


def _geodesic_distance_field(
    cavity_mask: np.ndarray, inlet_mask: np.ndarray
) -> np.ndarray:
    """26-neighbour geodesic distance from ``inlet_mask`` within ``cavity_mask``.

    Uses vectorised sparse-graph construction so 120 000-cell LBM grids do not
    hang in Python list append loops.
    """
    shape = cavity_mask.shape
    node_id = np.full(shape, -1, dtype=np.int64)
    n_nodes = int(cavity_mask.sum())
    if n_nodes == 0:
        return np.full(shape, np.inf, dtype=np.float64)
    node_id[cavity_mask] = np.arange(n_nodes, dtype=np.int64)
    nz, ny, nx = shape

    src_list: List[np.ndarray] = []
    dst_list: List[np.ndarray] = []
    w_list: List[np.ndarray] = []
    # 13 unique neighbour offsets; csr_graph below is undirected.
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                if not (
                    dz > 0 or (dz == 0 and dy > 0) or (dz == 0 and dy == 0 and dx > 0)
                ):
                    continue
                if dz >= 0:
                    sz = slice(0, nz - dz)
                    dz_s = slice(dz, nz)
                else:
                    sz = slice(-dz, nz)
                    dz_s = slice(0, nz + dz)
                if dy >= 0:
                    sy = slice(0, ny - dy)
                    dy_s = slice(dy, ny)
                else:
                    sy = slice(-dy, ny)
                    dy_s = slice(0, ny + dy)
                if dx >= 0:
                    sx = slice(0, nx - dx)
                    dx_s = slice(dx, nx)
                else:
                    sx = slice(-dx, nx)
                    dx_s = slice(0, nx + dx)
                src = node_id[sz, sy, sx]
                dst = node_id[dz_s, dy_s, dx_s]
                valid = (src >= 0) & (dst >= 0)
                if not valid.any():
                    continue
                src_list.append(src[valid])
                dst_list.append(dst[valid])
                weight = float(np.linalg.norm([dz, dy, dx]))
                w_list.append(np.full(valid.sum(), weight, dtype=np.float64))

    if src_list:
        rows = np.concatenate(src_list)
        cols = np.concatenate(dst_list)
        data = np.concatenate(w_list)
    else:
        rows = cols = data = np.empty(0, dtype=np.float64)
    graph = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    inlet_nodes = node_id[inlet_mask]
    inlet_nodes = inlet_nodes[inlet_nodes >= 0]
    geodesic = np.full(shape, np.inf, dtype=np.float64)
    if inlet_nodes.size > 0:
        dists = csgraph.dijkstra(
            graph, indices=inlet_nodes, directed=False, return_predecessors=False
        )
        min_dist = np.min(dists, axis=0)
        min_dist = np.where(np.isfinite(min_dist), min_dist, np.inf)
        geodesic[cavity_mask] = min_dist
    return geodesic


def _recommend_filter(
    gating_nodes: List[Any], Q_m3_s: float, alloy: Any
) -> Optional[str]:
    """Suggest a ceramic filter size/location when the runner flow is turbulent.

    Uses a simple Darcy-Forchheimer-style loss estimate.  The filter area is
    sized so the face velocity stays below a metal-dependent limit.
    """
    if not gating_nodes or Q_m3_s <= 1e-12:
        return None
    # Prefer the first RUNNER node; fall back to SPRUE_THROAT, then first node.
    target_node = None
    for n in gating_nodes:
        if "RUNNER" in n.body_type:
            target_node = n
            break
    if target_node is None:
        for n in gating_nodes:
            if "SPRUE_THROAT" in n.body_type or "SPRUE" in n.body_type:
                target_node = n
                break
    if target_node is None:
        target_node = gating_nodes[0]

    v_node = max(
        target_node.max_velocity_m_s
        if target_node.max_velocity_m_s > 1e-12
        else target_node.velocity_m_s,
        1e-6,
    )
    rho = float(getattr(alloy, "rho_liquid_kg_m3", 2700.0))
    # Al: face velocity through filter should stay < ~0.5 m/s.
    v_filter_max = 0.45
    A_f = Q_m3_s / v_filter_max
    D_f = np.sqrt(4.0 * A_f / np.pi) * 1000.0  # mm
    thickness_mm = 15.0
    ppi = 20
    # Rough pressure drop coefficient for a 20 PPI ceramic filter.
    K_filter = 4.0 + 0.05 * ppi
    dp_pa = 0.5 * rho * (v_node ** 2) * K_filter
    loc = target_node.name or target_node.body_type
    return (
        f"Seramik filtre önerisi: '{loc}' bölgesine Ø{D_f:.0f} mm, "
        f"{ppi} PPI, {thickness_mm:.0f} mm kalınlık; "
        f"yaklaşık {dp_pa:.0f} Pa ek basınç düşümü, "
        f"yüzey hızı ~{v_filter_max:.2f} m/s."
    )


def _flow_refined_grid(
    bodies: List[Body],
    casting_params,
    desired_dx_mm: float = _FLOW_CFG.desired_dx_mm,
    max_cells: int = _FLOW_CFG.max_solver_cells,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """Re-voxelise the casting bodies for the Darcy flow solve.

    The flow solve needs a grid fine enough to resolve gate cross-sections
    (desired_dx_mm) while the PCG solver can still handle the cavity cell count
    (max_cells).  If bodies are unavailable the caller falls back to the
    supplied grid.
    """
    if not bodies:
        return None, None, None
    from core.voxelizer import build_voxel_grid

    max_size = 0.0
    for b in bodies:
        size = float(np.max(b.mesh.bounds[1] - b.mesh.bounds[0]))
        if size > max_size:
            max_size = size
    if max_size <= 0.0:
        return None, None, None

    # Target dimension for the desired voxel pitch, capped by the memory budget.
    target_dim = max(160, int(np.ceil(max_size / desired_dx_mm)))
    # Total grid cells scale as target_dim^3.  Use a larger safety factor so
    # the desired dx (e.g. 0.8 mm) is not prematurely clipped; _downsample_grid
    # enforces the cavity-cell budget afterwards.
    max_dim = int(np.floor((max_cells * 8.0) ** (1.0 / 3.0)))
    target_dim = min(target_dim, max_dim)
    if target_dim < 160:
        target_dim = 160

    gvec = getattr(casting_params, "gravity_vector", (0.0, 0.0, -1.0))
    try:
        grid, _, origin, dx, _ = build_voxel_grid(
            bodies,
            target_dim=target_dim,
            gravity_vector=gvec,
            conservative=False,
            progress_callback=None,
        )
        return grid, origin, dx
    except Exception as exc:
        print(f"[Darcy solver] flow re-voxelization failed: {exc}; using supplied grid")
        return None, None, None


def _cavity_and_solid_masks(grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return cavity (mold cavity incl. gating+part) and solid masks."""
    cavity = grid != BodyType.EMPTY
    solid = ~cavity
    return cavity, solid


def _resample_to_grid(
    src: np.ndarray,
    src_origin: np.ndarray,
    src_dx: float,
    dst_shape: Tuple[int, int, int],
    dst_origin: np.ndarray,
    dst_dx: float,
    fill_value: float = 0.0,
    order: int = 1,
) -> np.ndarray:
    """Tri-linearly resample a cell-centred 3-D scalar from one regular grid to another.

    The two grids may have different origins, voxel pitches and shapes; the
    physical coordinates of the destination cell centres are mapped back to the
    source index frame and ``map_coordinates`` is used.
    """
    nz, ny, nx = dst_shape
    zc = dst_origin[0] + (np.arange(nz) + 0.5) * dst_dx
    yc = dst_origin[1] + (np.arange(ny) + 0.5) * dst_dx
    xc = dst_origin[2] + (np.arange(nx) + 0.5) * dst_dx
    zz, yy, xx = np.meshgrid(zc, yc, xc, indexing="ij")
    coords = np.stack(
        [
            (zz - src_origin[0]) / src_dx - 0.5,
            (yy - src_origin[1]) / src_dx - 0.5,
            (xx - src_origin[2]) / src_dx - 0.5,
        ],
        axis=0,
    )
    return ndimage.map_coordinates(
        src, coords, order=order, mode="nearest", cval=fill_value
    )


def _ensure_dirichlet_per_component(
    cavity: np.ndarray,
    dirichlet: np.ndarray,
    dirichlet_value: np.ndarray,
    g: np.ndarray,
    origin: np.ndarray,
    dx: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Guarantee every connected cavity component has at least one Dirichlet cell.

    Isolated pockets (e.g. a disconnected riser) would otherwise make the
    Laplacian matrix singular.  We set the highest (most upstream) cell of each
    unassigned component to atmospheric pressure p=0.
    """
    # Use the same 6-face connectivity as the Darcy matrix to avoid declaring
    # diagonally-touching cells as one component when the matrix has no edge.
    structure = np.zeros((3, 3, 3), dtype=int)
    structure[1, 1, 1] = 1
    structure[0, 1, 1] = structure[2, 1, 1] = 1
    structure[1, 0, 1] = structure[1, 2, 1] = 1
    structure[1, 1, 0] = structure[1, 1, 2] = 1
    labeled, n = ndimage.label(cavity, structure=structure)
    if n <= 1:
        return dirichlet, dirichlet_value
    proj = _projection_along(cavity.shape, origin, dx, g)
    out_d = dirichlet.copy()
    out_v = dirichlet_value.copy()
    for label in range(1, n + 1):
        comp = labeled == label
        if not (out_d & comp).any():
            idx = np.unravel_index(np.argmax(np.where(comp, proj, -np.inf)), comp.shape)
            out_d[idx] = True
            out_v[idx] = 0.0
    return out_d, out_v


def _gravity_unit(g: Tuple[float, float, float]) -> np.ndarray:
    g = np.asarray(g, dtype=np.float64)
    norm = float(np.linalg.norm(g))
    if norm == 0.0:
        g = np.array([0.0, 0.0, -1.0])
    else:
        g = g / norm
    return g


def _projection_along(
    shape: Tuple[int, int, int],
    origin: np.ndarray,
    dx: float,
    g: np.ndarray,
) -> np.ndarray:
    """Return the scalar projection of each cell centre onto direction -g."""
    nx, ny, nz = shape
    ix = np.arange(nx) * dx + origin[0] + dx / 2.0
    iy = np.arange(ny) * dx + origin[1] + dx / 2.0
    iz = np.arange(nz) * dx + origin[2] + dx / 2.0
    X, Y, Z = np.meshgrid(ix, iy, iz, indexing="ij")
    coords = np.stack([X, Y, Z], axis=-1)
    return -np.tensordot(coords, g, axes=[[-1], [0]])


def _roll_axis_from_g(g: np.ndarray) -> Optional[int]:
    """Return the dominant axis closest to +/-g, or None if diagonal."""
    abs_g = np.abs(g)
    if abs_g.max() < 0.5:
        return None
    return int(np.argmax(abs_g))


def _find_boundary_cells_along(
    mask: np.ndarray,
    g: np.ndarray,
    side: str,
) -> np.ndarray:
    """Return the exact upstream (side='up') or downstream (side='down')
    face cells of a boolean mask for an arbitrary normalised gravity vector.

    A cell is a face cell only if it has at least one 26-neighbour in the
    requested half-space (against gravity for 'up', with gravity for 'down')
    that lies outside the mask or outside the grid.  The connected component
    containing the extreme projection onto that direction is kept, so the
    selected area is not artificially inflated by a tolerance band.
    """
    out = np.zeros_like(mask, dtype=bool)
    if not mask.any():
        return out

    shape = mask.shape
    # 26-neighbour offsets.
    offsets = [
        (di, dj, dk)
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
        for dk in (-1, 0, 1)
        if not (di == 0 and dj == 0 and dk == 0)
    ]

    # Projection of each voxel centre onto -g in voxel units.  Only the
    # ordering matters; the upstream direction maximises this projection.
    ii, jj, kk = np.indices(shape, dtype=np.float64)
    proj = -(ii * g[0] + jj * g[1] + kk * g[2])

    # A face cell has a neighbour in the requested flow half-space that is
    # not part of the mask.  side='up'  -> flow comes from -g, so we look
    # at offsets with d·g < 0.  side='down' -> flow goes with g, d·g > 0.
    directional = np.zeros_like(mask, dtype=bool)
    for di, dj, dk in offsets:
        dot = float(di * g[0] + dj * g[1] + dk * g[2])
        if side == "up" and dot >= 0:
            continue
        if side == "down" and dot <= 0:
            continue

        # We need rolled[c] = mask[c + d], which is np.roll(mask, -d).
        rolled = np.roll(mask, (-di, -dj, -dk), axis=(0, 1, 2))
        # Zero the slices that wrapped around from the opposite border.
        if di > 0:
            rolled[-di:, :, :] = False
        elif di < 0:
            rolled[: abs(di), :, :] = False
        if dj > 0:
            rolled[:, -dj:, :] = False
        elif dj < 0:
            rolled[:, : abs(dj), :] = False
        if dk > 0:
            rolled[:, :, -dk:] = False
        elif dk < 0:
            rolled[:, :, : abs(dk)] = False

        directional |= (mask & (~rolled))

    if not directional.any():
        return out

    if side == "up":
        limit = float(proj[directional].max())
        seed = directional & (proj >= limit - 1e-12)
    else:
        limit = float(proj[directional].min())
        seed = directional & (proj <= limit + 1e-12)

    labeled, num = ndimage.label(directional, structure=np.ones((3, 3, 3), dtype=int))
    if num == 0:
        return out
    seed_labels = np.unique(labeled[seed])
    seed_labels = seed_labels[seed_labels != 0]
    if seed_labels.size == 0:
        return out
    out = np.isin(labeled, seed_labels)
    return out


def _select_inlet_cells(
    grid: np.ndarray,
    cavity: np.ndarray,
    g: np.ndarray,
    section_key: str,
) -> Tuple[np.ndarray, str]:
    """Select the inlet (upstream) boundary cells from the user-selected section."""
    key_lower = (section_key or "SPRUE").upper()
    type_map = {
        "SPRUE": BodyType.SPRUE,
        "SPRUE_BASE": BodyType.SPRUE,
        "SPRUE_THROAT": BodyType.SPRUE_THROAT,
        "POURING_BASIN": BodyType.POURING_BASIN,
        "RUNNER": BodyType.RUNNER,
        "DISTRIBUTOR": BodyType.DISTRIBUTOR,
        "CURUFLUK": BodyType.CURUFLUK,
        "INGATE": BodyType.INGATE,
        "FILTER": BodyType.FILTER,
    }
    body_type = type_map.get(key_lower, BodyType.SPRUE)

    mask = (grid == body_type) & cavity
    # For SPRUE_THROAT, if no dedicated throat body exists, fall back to the
    # top of the SPRUE body below it.
    if key_lower == "SPRUE_THROAT" and not mask.any():
        mask = (grid == BodyType.SPRUE) & cavity
    chosen_name = BodyType(body_type).name

    if not mask.any():
        # Fall back to any SPRUE / POURING_BASIN / RUNNER.
        for bt, name in (
            (BodyType.SPRUE, "SPRUE"),
            (BodyType.POURING_BASIN, "POURING_BASIN"),
            (BodyType.RUNNER, "RUNNER"),
        ):
            candidate = (grid == bt) & cavity
            if candidate.any():
                mask = candidate
                chosen_name = name
                body_type = bt
                break
        else:
            mask = cavity.copy()
            chosen_name = "CAVITY"

    inlet = _find_boundary_cells_along(mask, g, side="up") & cavity
    if not inlet.any():
        # Last resort: the single highest cell of the mask along -g.
        proj = _projection_along(grid.shape, np.zeros(3), 1.0, g)
        mask_idx = np.argwhere(mask)
        if len(mask_idx) > 0:
            best = tuple(mask_idx[np.argmax(proj[mask])])
            inlet = np.zeros_like(mask, dtype=bool)
            inlet[best] = True
    return inlet, chosen_name


def _select_vent_cells(
    grid: np.ndarray,
    cavity: np.ndarray,
    g: np.ndarray,
) -> np.ndarray:
    """Select vent cells: top of an open RISER only.

    Parça, meme, yolluk, sagu, döküm hunisi ve curufluk üst yüzeyleri hiçbir
    kalıp tipinde otomatik vent sayılmaz. Kum kalıplarda yüzeysel hava kaçışı
    `permeability_proxy` ile LBM sonrası düzeltilir, açık sınır olarak değil.
    """
    riser_mask = (grid == BodyType.RISER) & cavity
    if not riser_mask.any():
        return np.zeros_like(cavity, dtype=bool)
    return _find_boundary_cells_along(riser_mask, g, side="up") & cavity


def _select_lbm_outlet_cells(
    grid: np.ndarray,
    cavity: np.ndarray,
    g: np.ndarray,
    mold=None,
) -> np.ndarray:
    """Select LBM air-escape cells (same physical rule as _select_vent_cells)."""
    return _select_vent_cells(grid, cavity, g)


def _build_laplace_matrix(
    cavity: np.ndarray,
    dirichlet: np.ndarray,
    dirichlet_value: np.ndarray,
    permeability: Optional[np.ndarray] = None,
    dx: float = 1.0,
    face_fractions: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    viscosity_pa_s: float = 1.0,
) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Build a 7-point Darcy matrix on the cavity with Dirichlet cells.

    Solid neighbours are treated as zero-flux (Neumann) boundaries.  Face
    conductance is ``K_face * f_A / (mu * dx**2)`` where ``K_face`` is the
    harmonic mean of the two adjacent cell permeabilities, ``mu`` is the
    dynamic viscosity, and ``f_A`` is the FAVOR fractional face area.  This
    makes narrow gates locally more resistive and removes the staircase area
    bloat on curved geometry.
    """
    flat_idx = np.full(cavity.shape, -1, dtype=np.int32)
    flat_idx[cavity] = np.arange(int(cavity.sum()))
    n_unknowns = int(cavity.sum())

    rows_parts: List[np.ndarray] = []
    cols_parts: List[np.ndarray] = []
    data_parts: List[np.ndarray] = []
    rhs = np.zeros(n_unknowns, dtype=np.float64)
    diag = np.zeros(n_unknowns, dtype=np.float64)

    if permeability is None:
        permeability = np.ones(cavity.shape, dtype=np.float64)
    mu = max(float(viscosity_pa_s), 1e-9)
    K = np.maximum(permeability, 1e-18) / mu
    dx2 = float(dx) * float(dx)

    if face_fractions is None:
        nz, ny, nx = cavity.shape
        f_A_z = np.ones((nz + 1, ny, nx), dtype=np.float64)
        f_A_y = np.ones((nz, ny + 1, nx), dtype=np.float64)
        f_A_x = np.ones((nz, ny, nx + 1), dtype=np.float64)
    else:
        f_A_z, f_A_y, f_A_x = face_fractions

    def _add_faces(cur_flat, nb_flat, cur_dir, nb_dir, cur_val, nb_val, K_face):
        valid = (cur_flat >= 0) & (nb_flat >= 0)
        if not valid.any():
            return
        cur_idx = cur_flat[valid]
        nb_idx = nb_flat[valid]
        cdir = cur_dir[valid]
        ndir = nb_dir[valid]
        cv = cur_val[valid]
        nv = nb_val[valid]
        w = K_face[valid] / dx2

        # both non-Dirichlet: symmetric off-diagonals, both diagonals accumulate
        both = (~cdir) & (~ndir)
        if both.any():
            c = cur_idx[both]
            n = nb_idx[both]
            wb = w[both]
            rows_parts.append(np.concatenate([c, n]))
            cols_parts.append(np.concatenate([n, c]))
            data_parts.append(np.concatenate([wb, wb]))
            np.add.at(diag, c, wb)
            np.add.at(diag, n, wb)

        # cur non-Dirichlet, nb Dirichlet
        c_nd_nb_d = (~cdir) & ndir
        if c_nd_nb_d.any():
            c = cur_idx[c_nd_nb_d]
            wb = w[c_nd_nb_d]
            np.add.at(diag, c, wb)
            np.add.at(rhs, c, -wb * nv[c_nd_nb_d])

        # cur Dirichlet, nb non-Dirichlet (the symmetric contribution from
        # the other side of the same face).
        c_d_nb_nd = cdir & (~ndir)
        if c_d_nb_nd.any():
            n = nb_idx[c_d_nb_nd]
            wb = w[c_d_nb_nd]
            np.add.at(diag, n, wb)
            np.add.at(rhs, n, -wb * cv[c_d_nb_nd])

    # z-faces (axis 0) -- interior face indices 1..nz-1 of f_A_z
    Kz = 2.0 * K[:-1] * K[1:] / (K[:-1] + K[1:]) * f_A_z[1:-1]
    _add_faces(
        flat_idx[:-1], flat_idx[1:],
        dirichlet[:-1], dirichlet[1:],
        dirichlet_value[:-1], dirichlet_value[1:], Kz,
    )
    # y-faces (axis 1)
    Ky = 2.0 * K[:, :-1] * K[:, 1:] / (K[:, :-1] + K[:, 1:]) * f_A_y[:, 1:-1, :]
    _add_faces(
        flat_idx[:, :-1], flat_idx[:, 1:],
        dirichlet[:, :-1], dirichlet[:, 1:],
        dirichlet_value[:, :-1], dirichlet_value[:, 1:], Ky,
    )
    # x-faces (axis 2)
    Kx = 2.0 * K[:, :, :-1] * K[:, :, 1:] / (K[:, :, :-1] + K[:, :, 1:]) * f_A_x[:, :, 1:-1]
    _add_faces(
        flat_idx[:, :, :-1], flat_idx[:, :, 1:],
        dirichlet[:, :, :-1], dirichlet[:, :, 1:],
        dirichlet_value[:, :, :-1], dirichlet_value[:, :, 1:], Kx,
    )

    unknown = np.arange(n_unknowns, dtype=np.int32)
    dirichlet_unknowns = dirichlet[cavity]
    if dirichlet_unknowns.any():
        rows_parts.append(unknown[dirichlet_unknowns])
        cols_parts.append(unknown[dirichlet_unknowns])
        data_parts.append(np.ones(dirichlet_unknowns.sum(), dtype=np.float64))
        rhs[dirichlet_unknowns] = dirichlet_value[cavity][dirichlet_unknowns]

    non_dirichlet = ~dirichlet_unknowns
    if non_dirichlet.any():
        rows_parts.append(unknown[non_dirichlet])
        cols_parts.append(unknown[non_dirichlet])
        data_parts.append(-diag[non_dirichlet])

    rows = np.concatenate(rows_parts) if rows_parts else np.empty(0, dtype=np.int64)
    cols = np.concatenate(cols_parts) if cols_parts else np.empty(0, dtype=np.int64)
    data = np.concatenate(data_parts) if data_parts else np.empty(0, dtype=np.float64)
    A = csr_matrix((data, (rows, cols)), shape=(n_unknowns, n_unknowns))
    return A, rhs, flat_idx, dirichlet_unknowns, dirichlet_value[cavity]


def _solve_pressure_cpp(core, A_spd, b_spd, n_red):
    """Try the C++ AMGCL BiCGStab+AMG pressure solver."""
    try:
        # Ensure CSR arrays are contiguous and typed for nanobind/AMGCL.
        A = A_spd.astype(np.float64, copy=False)
        if not A.has_sorted_indices:
            A.sort_indices()
        indptr = A.indptr.astype(np.int64)
        indices = A.indices.astype(np.int64)
        data = A.data.astype(np.float64)
        rhs = np.ascontiguousarray(b_spd, dtype=np.float64)

        max_iter = max(1000, min(3000, n_red + 500))
        x = core.solve_pressure(indptr, indices, data, rhs, max_iter, 1e-7, 1e-12)

        # Guard against an unconverged or nonsensical return.
        if not np.isfinite(x).all():
            raise RuntimeError("C++ pressure solution contains non-finite values")
        resid = np.linalg.norm(A.dot(x) - rhs) / (np.linalg.norm(rhs) + 1e-18)
        if resid > 1e-3:
            raise RuntimeError(f"C++ pressure residual too large: {resid}")
        return x
    except Exception as exc:
        print(f"[Darcy solver] C++ AMGCL fallback: {exc}")
        return None


def _solve_pressure(
    A: csr_matrix,
    rhs: np.ndarray,
    dirichlet_unknowns: np.ndarray,
    dirichlet_value: np.ndarray,
) -> np.ndarray:
    """Solve the sparse SPD pressure system with PCG + a Jacobi or ILU(0) preconditioner.

    The interior discretisation matrix is a weighted graph Laplacian that is
    symmetric negative-definite once Dirichlet (inlet/vent) cells are fixed.
    We therefore solve the reduced system for the non-Dirichlet unknowns on
    ``-A``, which is symmetric positive-definite.  The direct ``spsolve``
    factorisation is only used as a last-resort safety net for very small
    systems because it can consume huge amounts of RAM on million-cell grids.
    """
    n = A.shape[0]
    is_dir = np.asarray(dirichlet_unknowns, dtype=bool)
    red = ~is_dir

    p = np.empty(n, dtype=np.float64)
    if is_dir.any():
        p[is_dir] = dirichlet_value[is_dir]
    else:
        # No Dirichlet fixed: pressure is defined up to a constant; anchor one
        # cell to make the reduced system SPD.
        is_dir = np.zeros(n, dtype=bool)
        is_dir[0] = True
        p[is_dir] = 0.0
        red[0] = False

    def _cg_solve(A_spd, b_spd, n_red):
        # 0) C++ AMGCL solver (OpenVDB build path) when available.
        if os.environ.get("JOSECAST_USE_CPP_DARCY", "1") != "0":
            try:
                from core.cpp_bridge import JOSECAST_CORE
                if JOSECAST_CORE is not None:
                    x = _solve_pressure_cpp(JOSECAST_CORE, A_spd, b_spd, n_red)
                    if x is not None:
                        return x
            except Exception as exc:
                print(f"[Darcy solver] C++ AMGCL exception: {exc}")

        # 1) Jacobi-preconditioned CG.
        try:
            diag = A_spd.diagonal()
            safe_diag = np.where(np.abs(diag) > 1e-18, diag, 1.0)
            inv_diag = 1.0 / safe_diag
            M = spla.LinearOperator((n_red, n_red), matvec=lambda x: inv_diag * x)
            x, info = spla.cg(
                A_spd, b_spd, M=M, atol=0.0, rtol=1e-5, maxiter=min(2000, n_red + 500)
            )
            if info == 0:
                return x
            print(f"[Darcy solver] PCG+Jacobi failed info={info}")
        except Exception as exc:
            print(f"[Darcy solver] PCG+Jacobi exception: {exc}")

        # 2) ILU(0) preconditioned CG for moderate systems.
        if n_red <= 200_000:
            try:
                A_csc = A_spd.tocsc()
                ilu = spla.spilu(
                    A_csc,
                    drop_tol=1e-6,
                    fill_factor=1.5,
                    diag_pivot_thresh=0.0,
                    options={"ColPerm": "NATURAL"},
                )
                M = spla.LinearOperator((n_red, n_red), matvec=ilu.solve)
                x, info = spla.cg(
                    A_spd, b_spd, M=M, atol=0.0, rtol=1e-8, maxiter=min(5000, n_red + 1000)
                )
                if info == 0:
                    return x
                print(f"[Darcy solver] PCG+ILU failed info={info}")
            except Exception as exc:
                print(f"[Darcy solver] PCG+ILU exception: {exc}")

        # 3) Direct solve only as a safety net for small systems.
        if n_red < 100_000:
            try:
                x = spla.spsolve(A_spd, b_spd)
                if not np.isfinite(x).all():
                    print("[Darcy solver] spsolve produced non-finite values")
                    x = np.zeros_like(b_spd)
                return x
            except Exception as exc:
                print(f"[Darcy solver] spsolve failed: {exc}")

        # 4) Last resort: zero pressure.
        print("[Darcy solver] all pressure solvers failed; returning zero pressure")
        return np.zeros_like(b_spd)

    if red.any():
        A_red = A[red][:, red]
        A_spd = -A_red
        b_spd = -rhs[red]
        p[red] = _cg_solve(A_spd, b_spd, int(red.sum()))

    return p


def _face_velocities(
    p: np.ndarray,
    cavity: np.ndarray,
    dx: float,
    permeability: Optional[np.ndarray] = None,
    viscosity_pa_s: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute staggered Darcy face velocities u = -(K_face/mu) * dp/dx, etc.

    Velocities are zero on faces adjacent to a solid cell.  ``permeability`` is
    cell-centred; face values are the harmonic mean of the two adjacent cells.
    """
    mu = max(float(viscosity_pa_s), 1e-9)
    if permeability is None:
        Kz = Ky = Kx = 1.0 / mu
    else:
        K = np.maximum(permeability, 1e-18) / mu
        Kz = 2.0 * K[:-1] * K[1:] / (K[:-1] + K[1:])
        Ky = 2.0 * K[:, :-1] * K[:, 1:] / (K[:, :-1] + K[:, 1:])
        Kx = 2.0 * K[:, :, :-1] * K[:, :, 1:] / (K[:, :, :-1] + K[:, :, 1:])

    u = np.zeros((p.shape[0] + 1, p.shape[1], p.shape[2]), dtype=np.float64)
    v = np.zeros((p.shape[0], p.shape[1] + 1, p.shape[2]), dtype=np.float64)
    w = np.zeros((p.shape[0], p.shape[1], p.shape[2] + 1), dtype=np.float64)

    # interior faces
    u[1:-1, :, :] = -Kz * (p[1:] - p[:-1]) / dx
    v[:, 1:-1, :] = -Ky * (p[:, 1:] - p[:, :-1]) / dx
    w[:, :, 1:-1] = -Kx * (p[:, :, 1:] - p[:, :, :-1]) / dx

    u_valid = np.zeros(u.shape, dtype=bool)
    u_valid[1:-1, :, :] = cavity[:-1] & cavity[1:]
    u *= u_valid

    v_valid = np.zeros(v.shape, dtype=bool)
    v_valid[:, 1:-1, :] = cavity[:, :-1] & cavity[:, 1:]
    v *= v_valid

    w_valid = np.zeros(w.shape, dtype=bool)
    w_valid[:, :, 1:-1] = cavity[:, :, :-1] & cavity[:, :, 1:]
    w *= w_valid
    return u, v, w


def _cell_velocity_magnitude(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """Interpolate staggered face velocities to cell centres and return magnitude."""
    ux = 0.5 * (u[:-1] + u[1:])
    vy = 0.5 * (v[:, :-1] + v[:, 1:])
    wz = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
    return np.sqrt(ux * ux + vy * vy + wz * wz)


def _inlet_face_area_m2(
    source: np.ndarray,
    cavity: np.ndarray,
    dx: float,
    face_fractions: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> float:
    """Real open area (m²) of the faces separating the source mask from the rest of the cavity."""
    if face_fractions is None:
        nz, ny, nx = cavity.shape
        f_A_z = np.ones((nz + 1, ny, nx), dtype=np.float64)
        f_A_y = np.ones((nz, ny + 1, nx), dtype=np.float64)
        f_A_x = np.ones((nz, ny, nx + 1), dtype=np.float64)
    else:
        f_A_z, f_A_y, f_A_x = face_fractions
    area = dx * dx
    A = 0.0

    # z-faces (axis 0)
    left = source[:-1] & ~source[1:] & cavity[1:]
    right = source[1:] & ~source[:-1] & cavity[:-1]
    a_z = f_A_z[1:-1] * area
    A += float(a_z[left | right].sum())

    # y-faces (axis 1)
    down = source[:, :-1] & ~source[:, 1:] & cavity[:, 1:]
    up = source[:, 1:] & ~source[:, :-1] & cavity[:, :-1]
    a_y = f_A_y[:, 1:-1, :] * area
    A += float(a_y[down | up].sum())

    # x-faces (axis 2)
    back = source[:, :, :-1] & ~source[:, :, 1:] & cavity[:, :, 1:]
    front = source[:, :, 1:] & ~source[:, :, :-1] & cavity[:, :, :-1]
    a_x = f_A_x[:, :, 1:-1] * area
    A += float(a_x[back | front].sum())

    return A


def _mesh_surface_pv(mesh: trimesh.Trimesh) -> Optional[pv.PolyData]:
    """Return a PyVista surface of a trimesh for inside/outside tests."""
    if mesh is None or len(mesh.faces) == 0:
        return None
    try:
        faces = np.asarray(mesh.faces)
        if faces.ndim != 2 or faces.shape[1] != 3:
            faces = mesh.triangles
        n_faces = faces.shape[0]
        if n_faces == 0:
            return None
        faces_arr = np.hstack(
            [np.full((n_faces, 1), 3, dtype=faces.dtype), faces]
        ).ravel()
        return pv.PolyData(
            np.asarray(mesh.vertices, dtype=np.float64), faces_arr
        )
    except Exception:
        return None


def _mesh_throat_area_m2(
    mesh: trimesh.Trimesh,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    body_centroid: Optional[np.ndarray] = None,
    n_sweep: int = 5,
) -> float:
    """Return the throat cross-section area (m²) of ``mesh`` near ``plane_origin``.

    The plane is perpendicular to ``plane_normal`` (the local flow direction).  To
    avoid cutting the body only at a sharp corner or end cap, the plane is swept
    a short distance *into* the body (toward ``body_centroid``) and the largest
    valid polygon area is returned.
    """
    if mesh is None or len(mesh.faces) == 0:
        return 0.0
    n = np.asarray(plane_normal, dtype=np.float64)
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-18:
        n = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        n = n / n_norm

    origin = np.asarray(plane_origin, dtype=np.float64)

    # Ensure the sweep goes into the body, not out into empty space.
    if body_centroid is not None:
        to_centroid = np.asarray(body_centroid, dtype=np.float64) - origin
        if float(np.dot(to_centroid, n)) < -1e-12:
            n = -n

    bbox = np.asarray(mesh.bounds, dtype=np.float64)
    diag = float(np.linalg.norm(bbox[1] - bbox[0]))
    # Move far enough into the body to skip a possible open end-cap, but not
    # more than a small fraction of the body diagonal.
    L = max(diag * 0.05, 0.5)
    if n_sweep <= 1:
        steps = [0.0]
    else:
        steps = np.linspace(0.0, L, n_sweep)

    best = 0.0
    for s in steps:
        try:
            section = mesh.section(plane_origin=origin + s * n, plane_normal=n)
            if section is None:
                continue
            path2d = section.to_2D()
            if isinstance(path2d, tuple):
                path2d = path2d[0]
            area_mm2 = float(path2d.area)
            if area_mm2 > best:
                best = area_mm2
        except Exception:
            continue
    return float(best * 1e-6)


def _first_valid_throat_area_m2(
    mesh: trimesh.Trimesh,
    plane_origin: np.ndarray,
    normals: List[np.ndarray],
    body_centroid: Optional[np.ndarray] = None,
    n_sweep: int = 5,
) -> Tuple[float, Optional[np.ndarray]]:
    """Return the first valid throat area (m²) and the normal that produced it.

    The list ``normals`` is ordered from most-physical (e.g. the body's own
    entry-to-exit flow axis) to geometric fallbacks.  The first direction that
    produces a non-zero section is used, so we do not accidentally pick a smaller,
    non-flow-aligned slice of the body.  ``body_centroid`` makes the sweep move
    into the body instead of out into empty space.
    """
    for n in normals:
        n = np.asarray(n, dtype=np.float64)
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-12:
            continue
        a = _mesh_throat_area_m2(
            mesh, plane_origin, n / n_norm, body_centroid=body_centroid, n_sweep=n_sweep
        )
        if a > 1e-18:
            return a, n / n_norm
    return 0.0, None


def _build_mesh_contacts(
    comp_body: Dict[int, Body],
    comp_meta: Dict[int, Tuple[BodyType, str]],
    comp_centroids: Dict[int, np.ndarray],
    g: np.ndarray,
    part_id: int,
    max_gap_mm: float = 0.5,
    max_query: int = 5000,
    verbose: bool = True,
) -> List[Dict]:
    """Build the contact graph directly from CAD mesh proximity.

    Every pair of bodies whose surface points are within ``max_gap_mm`` is
    considered a gating contact.  The contact centroid is the midpoint of the
    closest source/target surface pair, so the downstream area calculation starts
    at the real geometric interface instead of a coarse voxel face centre.

    Flow orientation is decided in three steps:
      1. Any contact involving the part: the non-part body is upstream.
      2. Otherwise the weighted average contact normal on each body is compared
         with gravity; the body whose contact surface points downward is upstream.
      3. If the normals are ambiguous (typical for side contacts), fall back to
         the gating-type hierarchy and, as a last resort, to the higher centroid
         along the reverse gravity direction.
    """
    ids = sorted(
        cid
        for cid, b in comp_body.items()
        if b is not None and b.mesh is not None and len(b.mesh.faces) > 0
    )
    contacts: List[Dict] = []
    if len(ids) < 2:
        return contacts

    g_vec = np.asarray(g, dtype=np.float64)
    g_norm = float(np.linalg.norm(g_vec))
    g_u = g_vec / g_norm if g_norm > 1e-12 else np.array([0.0, 0.0, -1.0])

    def _rank(cid: int) -> float:
        return float(-np.dot(comp_centroids.get(cid, np.zeros(3)), g_u))

    type_order = {
        BodyType.POURING_BASIN: 0,
        BodyType.SPRUE_THROAT: 1,
        BodyType.SPRUE: 2,
        BodyType.RUNNER: 3,
        BodyType.DISTRIBUTOR: 4,
        BodyType.CURUFLUK: 5,
        BodyType.FILTER: 6,
        BodyType.INGATE: 7,
        BodyType.RISER: 8,
        BodyType.PART: 9,
        BodyType.EMPTY: 10,
        BodyType.CORE: 11,
        BodyType.COOLING_SPRUE: 12,
    }

    def _order(cid: int) -> int:
        bt = comp_meta.get(cid, (BodyType.EMPTY, ""))[0]
        return type_order.get(bt, 10)

    for i, j in combinations(ids, 2):
        mesh_a = comp_body[i].mesh
        mesh_b = comp_body[j].mesh
        if mesh_a is mesh_b:
            continue
        name_a = comp_meta.get(i, (BodyType.EMPTY, f"Body_{i}"))[1]
        name_b = comp_meta.get(j, (BodyType.EMPTY, f"Body_{j}"))[1]

        sep = np.maximum(0.0, mesh_a.bounds[0] - mesh_b.bounds[1]).max()
        sep = max(sep, float(np.maximum(0.0, mesh_b.bounds[0] - mesh_a.bounds[1]).max()))
        bbox_sep = float(np.linalg.norm(np.maximum(0.0, np.maximum(mesh_a.bounds[0] - mesh_b.bounds[1], mesh_b.bounds[0] - mesh_a.bounds[1]))))

        if bbox_sep > max_gap_mm:
            if verbose:
                print(
                    f"[CONTACT_DISCOVERY] {name_a} <-> {name_b}: "
                    f"bbox_sep={bbox_sep:.2f} mm (> {max_gap_mm} mm) -> skip",
                    flush=True,
                )
            continue

        # Query the smaller mesh against the larger one; its face centres are
        # denser near the contact patch.
        if float(mesh_a.area) > float(mesh_b.area):
            source, target = mesh_b, mesh_a
            source_cid, target_cid = j, i
        else:
            source, target = mesh_a, mesh_b
            source_cid, target_cid = i, j

        centers = np.asarray(source.triangles_center, dtype=np.float64)
        face_indices = np.arange(centers.shape[0], dtype=np.int64)

        if centers.shape[0] > max_query:
            step = centers.shape[0] / max_query
            sample_idx = (np.arange(max_query) * step).astype(int)
            sample_idx[-1] = min(sample_idx[-1], centers.shape[0] - 1)
            local_centers = centers[sample_idx]
            local_face_indices = face_indices[sample_idx]
        else:
            local_centers = centers
            local_face_indices = face_indices

        try:
            closest, dist, tgt_face_idx = trimesh.proximity.closest_point(target, local_centers)
        except Exception as exc:
            if verbose:
                print(
                    f"[CONTACT_DISCOVERY] {name_a} <-> {name_b}: "
                    f"closest_point failed ({exc}) -> skip",
                    flush=True,
                )
            continue

        min_idx = int(np.argmin(dist))
        d_min = float(dist[min_idx])
        if d_min > max_gap_mm:
            if verbose:
                print(
                    f"[CONTACT_DISCOVERY] {name_a} <-> {name_b}: "
                    f"d_min={d_min:.3f} mm (> {max_gap_mm} mm) -> skip",
                    flush=True,
                )
            continue

        # Collect all sampled source faces within the contact tolerance to build a
        # robust patch-averaged normal.  This smooths out single-triangle noise on
        # side contacts.
        dist_arr = np.asarray(dist, dtype=np.float64)
        patch_mask = dist_arr <= max_gap_mm
        if not patch_mask.any():
            patch_mask[min_idx] = True

        src_faces = local_face_indices[patch_mask]
        tgt_faces = tgt_face_idx[patch_mask].astype(np.int64)
        src_areas = source.area_faces[src_faces] if len(src_faces) > 0 else np.ones(1)
        tgt_areas = target.area_faces[tgt_faces] if len(tgt_faces) > 0 else np.ones(1)
        src_normals = source.face_normals[src_faces]
        tgt_normals = target.face_normals[tgt_faces]
        if src_areas.sum() > 1e-18:
            n_src = np.sum(src_normals * src_areas[:, None], axis=0) / src_areas.sum()
        else:
            n_src = src_normals.mean(axis=0) if src_normals.size else g_u
        if tgt_areas.sum() > 1e-18:
            n_tgt = np.sum(tgt_normals * tgt_areas[:, None], axis=0) / tgt_areas.sum()
        else:
            n_tgt = tgt_normals.mean(axis=0) if tgt_normals.size else -g_u
        n_src = n_src / (np.linalg.norm(n_src) + 1e-18)
        n_tgt = n_tgt / (np.linalg.norm(n_tgt) + 1e-18)

        p_src = local_centers[min_idx]
        p_tgt = np.asarray(closest[min_idx], dtype=np.float64)

        # Map source/target normals and points back to the original ids i/j.
        if source_cid == i:
            n_i, n_j = n_src, n_tgt
            p_i, p_j = p_src, p_tgt
        else:
            n_i, n_j = n_tgt, n_src
            p_i, p_j = p_tgt, p_src

        # Determine flow direction (up_id -> down_id).
        if i == part_id:
            up_id, down_id = j, i
            up_pt, down_pt = p_j, p_i
        elif j == part_id:
            up_id, down_id = i, j
            up_pt, down_pt = p_i, p_j
        else:
            dot_i = float(np.dot(n_i, g_u))
            dot_j = float(np.dot(n_j, g_u))
            if dot_i > dot_j + 0.3:
                up_id, down_id = i, j
                up_pt, down_pt = p_i, p_j
            elif dot_j > dot_i + 0.3:
                up_id, down_id = j, i
                up_pt, down_pt = p_j, p_i
            else:
                # Side/ambiguous contact: use the gating hierarchy and then rank.
                o_i = _order(i)
                o_j = _order(j)
                if o_i == o_j:
                    if _rank(i) >= _rank(j):
                        up_id, down_id = i, j
                        up_pt, down_pt = p_i, p_j
                    else:
                        up_id, down_id = j, i
                        up_pt, down_pt = p_j, p_i
                elif o_i < o_j:
                    up_id, down_id = i, j
                    up_pt, down_pt = p_i, p_j
                else:
                    up_id, down_id = j, i
                    up_pt, down_pt = p_j, p_i

        centroid = 0.5 * (up_pt + down_pt)
        direction = down_pt - up_pt
        d_norm = float(np.linalg.norm(direction))
        if d_norm > 1e-12:
            normal = direction / d_norm
        else:
            normal = g_u

        # Face-normal based outward normals for the up/down bodies give a far more
        # accurate section plane than the centroid-to-centroid vector.
        if source_cid == up_id:
            up_normal = n_src
            down_normal = n_tgt
        else:
            up_normal = n_tgt
            down_normal = n_src
        # Outward normal of the upstream body must point downstream.
        if float(np.dot(up_normal, normal)) < 0.0:
            up_normal = -up_normal
        # Outward normal of the downstream body must point upstream.
        if float(np.dot(down_normal, normal)) > 0.0:
            down_normal = -down_normal

        contacts.append(
            {
                "id1": i,
                "id2": j,
                "up_id": up_id,
                "down_id": down_id,
                "type1": comp_meta.get(i, (BodyType.EMPTY, name_a))[0],
                "type2": comp_meta.get(j, (BodyType.EMPTY, name_b))[0],
                "name1": name_a,
                "name2": name_b,
                "voxel_area_m2": 0.0,
                "centroid_mm": centroid,
                "normal": normal,
                "up_normal": tuple(float(x) for x in up_normal),
                "down_normal": tuple(float(x) for x in down_normal),
                "up_pt_mm": tuple(float(x) for x in up_pt),
                "down_pt_mm": tuple(float(x) for x in down_pt),
                # Raw per-body contact points/normals used for global reorientation.
                "p1_mm": tuple(float(x) for x in p_i),
                "p2_mm": tuple(float(x) for x in p_j),
                "n1": tuple(float(x) for x in n_i),
                "n2": tuple(float(x) for x in n_j),
                "flux_m3_s": 0.0,
            }
        )
        if verbose:
            up_name = comp_meta.get(up_id, (None, str(up_id)))[1]
            down_name = comp_meta.get(down_id, (None, str(down_id)))[1]
            print(
                f"[CONTACT_DISCOVERY] {name_a} <-> {name_b}: "
                f"bbox_sep={bbox_sep:.2f} d_min={d_min:.3f} mm -> CONTACT "
                f"centroid=({centroid[0]:.2f},{centroid[1]:.2f},{centroid[2]:.2f}) "
                f"up={up_name} down={down_name}",
                flush=True,
            )
    return contacts


def _part_reachable_undirected(contacts: List[Dict], part_id: int) -> Dict[int, bool]:
    """Return the set of contact-graph nodes in the same component as ``part_id``.

    Uses ``id1``/``id2`` (the symmetric body ids), so the result does not depend
    on the local contact orientation.
    """
    from collections import defaultdict, deque

    graph: Dict[int, List[int]] = defaultdict(list)
    for c in contacts:
        u = int(c["id1"])
        v = int(c["id2"])
        graph[u].append(v)
        graph[v].append(u)

    if part_id not in graph:
        return {part_id: True}

    reach: Dict[int, bool] = {part_id: True}
    q: deque[int] = deque([part_id])
    while q:
        cur = q.popleft()
        for nb in graph.get(cur, []):
            if nb not in reach:
                reach[nb] = True
                q.append(nb)
    return reach


def _reorient_contacts_to_source(
    contacts: List[Dict],
    source_id: int,
    comp_meta: Dict[int, Tuple[BodyType, str]],
    comp_centroids: Dict[int, np.ndarray],
    g_u: np.ndarray,
    part_id: int,
    verbose: bool = False,
) -> None:
    """Point every contact away from ``source_id`` toward the part.

    Distance from the source is computed on the undirected contact graph, so
    side/T-junction contacts are oriented consistently with the main flow path
    instead of the local face normal.
    """
    if not contacts or source_id is None:
        return

    from collections import defaultdict, deque

    graph: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for idx, c in enumerate(contacts):
        u = int(c["id1"])
        v = int(c["id2"])
        graph[u].append((v, idx))
        graph[v].append((u, idx))

    if source_id not in graph:
        return

    dist: Dict[int, int] = {source_id: 0}
    q: deque[int] = deque([source_id])
    while q:
        cur = q.popleft()
        for nb, _ in graph.get(cur, []):
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                q.append(nb)

    for c in contacts:
        u = int(c["id1"])
        v = int(c["id2"])

        # The part is always the sink; any non-part body feeding it is upstream.
        if u == part_id:
            up_id, down_id = v, u
        elif v == part_id:
            up_id, down_id = u, v
        else:
            du = dist.get(u)
            dv = dist.get(v)
            if du is None or dv is None:
                continue
            if du == dv:
                # Leave the local orientation when the graph distance is ambiguous.
                continue
            up_id, down_id = (u, v) if du < dv else (v, u)

        if c.get("up_id") == up_id and c.get("down_id") == down_id:
            continue

        # Swap raw points/normals to match the new up/down assignment.
        p_up = np.array(c["p1_mm"] if up_id == u else c["p2_mm"], dtype=np.float64)
        p_down = np.array(c["p2_mm"] if down_id == v else c["p1_mm"], dtype=np.float64)
        n_up_raw = np.array(c["n1"] if up_id == u else c["n2"], dtype=np.float64)
        n_down_raw = np.array(c["n2"] if down_id == v else c["n1"], dtype=np.float64)

        direction = p_down - p_up
        d_norm = float(np.linalg.norm(direction))
        if d_norm > 1e-12:
            normal = direction / d_norm
        else:
            normal = np.asarray(g_u, dtype=np.float64)

        # Upstream body normal must point downstream; downstream body normal upstream.
        if float(np.dot(n_up_raw, normal)) < 0.0:
            n_up_raw = -n_up_raw
        if float(np.dot(n_down_raw, normal)) > 0.0:
            n_down_raw = -n_down_raw

        c["up_id"] = up_id
        c["down_id"] = down_id
        c["up_pt_mm"] = tuple(float(x) for x in p_up)
        c["down_pt_mm"] = tuple(float(x) for x in p_down)
        c["up_normal"] = tuple(float(x) for x in n_up_raw)
        c["down_normal"] = tuple(float(x) for x in n_down_raw)
        c["centroid_mm"] = tuple(float(x) for x in (0.5 * (p_up + p_down)))
        c["normal"] = normal

        if verbose:
            up_name = comp_meta.get(up_id, (None, str(up_id)))[1]
            down_name = comp_meta.get(down_id, (None, str(down_id)))[1]
            print(
                f"[CONTACT_DISCOVERY] reoriented: up={up_name} down={down_name}",
                flush=True,
            )


def _section_area_mm2(
    mesh: trimesh.Trimesh,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
) -> Tuple[float, int]:
    """Return the area (mm²) of a mesh cross-section at the given plane.

    The resulting planar contour is converted to 2-D and its total area is
    returned; ``n_loops`` is the number of closed contours found.
    """
    if mesh is None or len(mesh.faces) == 0:
        return 0.0, 0
    try:
        origin = np.asarray(plane_origin, dtype=np.float64)
        normal = np.asarray(plane_normal, dtype=np.float64)
        n_norm = float(np.linalg.norm(normal))
        if n_norm < 1e-12:
            normal = np.array([0.0, 0.0, -1.0])
        else:
            normal = normal / n_norm
        sec = mesh.section(plane_origin=origin, plane_normal=normal)
        if sec is None or len(sec.entities) == 0:
            return 0.0, 0
        to_2d = getattr(sec, "to_2D", None)
        if to_2d is None:
            path_2d, _ = sec.to_planar()
        else:
            path_2d, _ = to_2d()
        area = float(path_2d.area) if hasattr(path_2d, "area") else 0.0
        n_loops = len(path_2d.polygons_full) if hasattr(path_2d, "polygons_full") else 0
        return float(area), int(n_loops)
    except Exception:
        return 0.0, 0


def _body_principal_axis(mesh: trimesh.Trimesh, contact_centroid: np.ndarray) -> np.ndarray:
    """Return the unit OBB axis of ``mesh`` most aligned with the contact direction.

    Kept as a fallback; prefer :func:`_body_flow_axis` for gating contact area.
    """
    if mesh is None or len(mesh.faces) == 0:
        return np.array([0.0, 0.0, -1.0])
    try:
        obb = mesh.bounding_box_oriented
        axes = np.asarray(obb.primitive.transform[:3, :3], dtype=np.float64)
        body_c = np.asarray(mesh.centroid, dtype=np.float64)
        direction = np.asarray(contact_centroid, dtype=np.float64) - body_c
        n_dir = float(np.linalg.norm(direction))
        if n_dir < 1e-12:
            direction = np.array([0.0, 0.0, -1.0])
        else:
            direction = direction / n_dir
        dots = np.dot(axes.T, direction)
        idx = int(np.argmax(np.abs(dots)))
        axis = np.asarray(axes[:, idx], dtype=np.float64)
        n = float(np.linalg.norm(axis))
        if n < 1e-12:
            axis = np.array([0.0, 0.0, -1.0])
        else:
            axis = axis / n
        if float(np.dot(axis, direction)) < 0.0:
            axis = -axis
        return axis
    except Exception:
        return np.array([0.0, 0.0, -1.0])


def _body_flow_axis(
    mesh: trimesh.Trimesh,
    contact_centroid: np.ndarray,
    surface_normal: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return the unit flow/throat axis of ``mesh`` at the contact.

    The flow axis is the direction along which metal moves through the body,
    i.e. the axis whose perpendicular cross-section is the real throat/opening.
    - Long runners/sprues -> long OBB axis.
    - Short disk-like throats/caps -> short OBB axis.
    - Transitional shapes -> axis best aligned with the surface normal or the
      direction toward the contact centroid.
    """
    if mesh is None or len(mesh.faces) == 0:
        return np.array([0.0, 0.0, -1.0])
    try:
        obb = mesh.bounding_box_oriented
        axes = np.asarray(obb.primitive.transform[:3, :3], dtype=np.float64)
        extents = np.asarray(obb.primitive.extents, dtype=np.float64)
        body_c = np.asarray(mesh.centroid, dtype=np.float64)
    except Exception:
        return np.array([0.0, 0.0, -1.0])

    if extents.max() < 1e-12:
        return np.array([0.0, 0.0, -1.0])

    order = np.argsort(extents)
    short_idx, mid_idx, long_idx = int(order[0]), int(order[1]), int(order[2])
    r_min = extents[short_idx] / extents[long_idx]
    r_mid = extents[mid_idx] / extents[long_idx]

    # Direction from body centroid to the contact point.
    to_contact = np.asarray(contact_centroid, dtype=np.float64) - body_c
    n_to = float(np.linalg.norm(to_contact))
    if n_to > 1e-12:
        to_contact = to_contact / n_to
    else:
        to_contact = np.array([0.0, 0.0, -1.0])

    # Surface normal at the contact point on this body, if available.
    n = np.asarray(surface_normal, dtype=np.float64) if surface_normal is not None else None
    if n is not None:
        n_norm = float(np.linalg.norm(n))
        if n_norm > 1e-12:
            n = n / n_norm
        else:
            n = None

    def _aligned_index(direction: np.ndarray) -> Tuple[int, float]:
        cos_vals = np.abs(axes.T @ direction)
        idx = int(np.argmax(cos_vals))
        return idx, float(cos_vals[idx])

    # Disk-like: one short extent and two long, similar extents.
    if r_min < 0.25 and r_mid > 0.7:
        idx, cos = _aligned_index(n if n is not None else to_contact)
        # If the normal/contact points along the short axis, the face is the throat.
        if idx == short_idx and cos > 0.6:
            return _signed_axis(axes[:, short_idx], to_contact)
        # Otherwise the disk is being fed from its edge: throat is still the face.
        return _signed_axis(axes[:, short_idx], to_contact)

    # Rod-like: one long extent and two short, similar extents.
    if r_mid < 0.45 and r_min < 0.45:
        return _signed_axis(axes[:, long_idx], to_contact)

    # Transitional: use surface normal if it clearly aligns with one OBB axis,
    # otherwise fall back to the direction toward the contact, then the long axis.
    if n is not None:
        idx, cos = _aligned_index(n)
        if cos > 0.6:
            return _signed_axis(axes[:, idx], to_contact)
    idx, cos = _aligned_index(to_contact)
    if cos > 0.6:
        return _signed_axis(axes[:, idx], to_contact)
    return _signed_axis(axes[:, long_idx], to_contact)


def _signed_axis(axis: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Return ``axis`` oriented to point toward ``direction``."""
    axis = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return np.array([0.0, 0.0, -1.0])
    axis = axis / n
    if float(np.dot(axis, direction)) < 0.0:
        axis = -axis
    return axis


def _throat_area_along_axis_mm2(
    mesh: trimesh.Trimesh,
    contact_centroid: np.ndarray,
    axis: Optional[np.ndarray] = None,
    sweep_mm: Optional[Tuple[float, ...]] = None,
    plateau_tol: float = 0.15,
    plateau_window: int = 2,
    mode: str = "first",
    min_area_mm2: float = 1.0,
) -> float:
    """Full cross-sectional area (mm²) of ``mesh`` perpendicular to ``axis``.

    ``mode='first'`` (default): starting at ``contact_centroid`` and moving inward,
    return the first stable (plateau) cross-sectional area.  This is the right
    behaviour for a contact where the entry may be a sliver and the full face is
    a short distance inside.

    ``mode='min'``: sweep the whole supplied range and return the smallest
    non-sliver cross-sectional area.  This is used to find the geometric throat of
    an isolated source body.
    """
    if mesh is None or len(mesh.faces) == 0:
        return 0.0
    try:
        if axis is None:
            axis = _body_principal_axis(mesh, contact_centroid)
        axis = np.asarray(axis, dtype=np.float64)
        n = float(np.linalg.norm(axis))
        if n < 1e-12:
            axis = np.array([0.0, 0.0, -1.0])
        else:
            axis = axis / n
        # Ensure axis points from the body centroid toward the contact.
        body_c = np.asarray(mesh.centroid, dtype=np.float64)
        to_contact = np.asarray(contact_centroid, dtype=np.float64) - body_c
        if float(np.dot(to_contact, axis)) < 0.0:
            axis = -axis

        t = float(np.dot(contact_centroid - body_c, axis))
        p_axis = body_c + t * axis

        if sweep_mm is None:
            verts_t = (mesh.vertices.astype(np.float64) - body_c).dot(axis)
            L = float(verts_t.max() - verts_t.min())
            max_inward = min(5.0, max(1.5, 0.12 * L))
            sweep_mm = tuple(
                sorted(set(list(np.linspace(0.0, -max_inward, 25)) + [-0.02, -0.05, -0.1, -0.2, -0.5, -1.0, -2.0, -3.0]))
            )
        else:
            sweep_mm = tuple(sorted(sweep_mm))

        areas = []
        for s in sweep_mm:
            area, _ = _section_area_mm2(mesh, p_axis + axis * s, axis)
            areas.append(float(area))

        max_area = max(areas) if areas else 0.0
        if max_area <= 1e-9:
            return 0.0

        if mode == "min":
            threshold = max(max_area * 0.01, min_area_mm2)

            def is_plateau(window):
                if any(a <= threshold for a in window):
                    return False
                max_a = max(window)
                if max_a <= 1e-9:
                    return False
                return all(abs(a - max_a) / max_a <= plateau_tol for a in window)

            # First sustained plateau from the inlet end.
            first = None
            for i in range(len(areas) - plateau_window):
                window = areas[i : i + plateau_window + 1]
                if is_plateau(window):
                    first = float(np.mean(window))
                    break

            # Last sustained plateau before the closed end.
            last = None
            for i in range(len(areas) - plateau_window, -1, -1):
                window = areas[i : i + plateau_window + 1]
                if is_plateau(window):
                    last = float(np.mean(window))
                    break

            candidates = [a for a in (first, last) if a is not None]
            if candidates:
                return float(min(candidates))
            return float(max_area)

        threshold = max_area * 0.05
        # Look for the first stable area (plateau) when moving inward from the contact.
        for i in range(len(areas) - 1):
            if areas[i] <= threshold:
                continue
            window_end = min(i + plateau_window, len(areas) - 1)
            if window_end <= i:
                continue
            diffs = [abs(areas[j] - areas[i]) / max(areas[i], 1e-12) for j in range(i + 1, window_end + 1)]
            if all(d <= plateau_tol for d in diffs):
                return float(areas[i])

        # No plateau found: use the largest measured area (full cross-section) as
        # the best available estimate.
        return float(max_area)
    except Exception:
        return 0.0


def _section_polygon_union(
    mesh: trimesh.Trimesh,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    u: Optional[np.ndarray] = None,
    v: Optional[np.ndarray] = None,
    tol_mm: float = 0.2,
) -> Any:
    """Return a shapely union of the mesh section polygons projected onto a common 2-D basis, or None.

    The section is first attempted at ``plane_origin``; if that lies exactly on
    a boundary and ``mesh.section`` returns nothing, small offsets along the
    normal are tried.  The 2-D projection always uses ``plane_origin`` so the
    resulting polygons from two meshes remain in the same coordinate frame.
    """
    try:
        from shapely import geometry as geom
        from shapely.ops import unary_union
    except Exception:
        return None

    n = np.asarray(plane_normal, dtype=np.float64)
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-18:
        return None
    n = n / n_norm

    if u is None or v is None:
        arb = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u_vec = np.cross(arb, n)
        u_norm = float(np.linalg.norm(u_vec))
        if u_norm < 1e-18:
            return None
        u_vec = u_vec / u_norm
        v_vec = np.cross(n, u_vec)
    else:
        u_vec = np.asarray(u, dtype=np.float64)
        v_vec = np.asarray(v, dtype=np.float64)

    origin = np.asarray(plane_origin, dtype=np.float64)
    eps = max(tol_mm * 0.1, 0.01)

    for offset in (0.0, eps, -eps):
        try:
            section = mesh.section(
                plane_origin=origin + offset * n,
                plane_normal=n,
            )
        except Exception:
            section = None
        if section is None or getattr(section, "is_empty", True) or len(section.entities) == 0:
            continue

        polys = []
        for entity in section.entities:
            pts = getattr(entity, "points", None)
            if pts is None or len(pts) < 3:
                continue
            coords_3d = section.vertices[pts]
            coords_2d = np.column_stack(
                ((coords_3d - origin) @ u_vec, (coords_3d - origin) @ v_vec)
            )
            try:
                poly = geom.Polygon(coords_2d)
                if not poly.is_valid:
                    poly = poly.buffer(0.0)
                if poly.area > 1e-12:
                    polys.append(poly)
            except Exception:
                continue
        if polys:
            return unary_union(polys)
    return None


def _plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return orthonormal (u, v) spanning the plane with the given normal."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / (float(np.linalg.norm(n)) + 1e-18)
    arb = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(arb, n)
    u = u / (float(np.linalg.norm(u)) + 1e-18)
    v = np.cross(n, u)
    return u, v


def _contact_section_intersection(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
    centroid: np.ndarray,
    normal: np.ndarray,
    tol_mm: float = 0.2,
    verbose: bool = False,
    label: str = "",
) -> Tuple[float, float, float, str]:
    """Overlap area of the two body cross-sections on a common plane.

    Returns ``(area_mm2, area_up_mm2, area_down_mm2, method)``.
    """
    u, v = _plane_basis(normal)
    poly_a = _section_polygon_union(mesh_a, centroid, normal, u=u, v=v, tol_mm=tol_mm)
    poly_b = _section_polygon_union(mesh_b, centroid, normal, u=u, v=v, tol_mm=tol_mm)
    if poly_a is None or poly_b is None:
        return 0.0, 0.0, 0.0, ""

    try:
        inter = poly_a.intersection(poly_b)
        area = float(inter.area)
        method = "section"

        if area <= 1e-12 and tol_mm > 1e-12:
            # Near-touch: surfaces are within tolerance but the plane sections
            # do not formally overlap.  Buffer both by half the tolerance.
            inter = poly_a.buffer(tol_mm * 0.5).intersection(poly_b.buffer(tol_mm * 0.5))
            area = float(inter.area)
            method = "section(buffer)"

        if area > 1e-12:
            return float(area), float(poly_a.area), float(poly_b.area), method
    except Exception:
        pass

    return 0.0, 0.0, 0.0, ""


def _face_contact_area_single_mm2(
    source: trimesh.Trimesh,
    target: trimesh.Trimesh,
    tol_mm: float = 0.2,
    contact_centroid: Optional[np.ndarray] = None,
    max_source_faces: int = 50000,
) -> float:
    """Return the area (mm²) of source boundary faces that touch ``target``.

    A source face is part of the physical contact if its outward normal points
    toward the contact region, its centroid is within ``tol_mm`` of the target
    surface, and the target lies on the outward normal side.  This rejects side
    faces and back faces of the smaller body that happen to be geometrically
    close to the larger body.
    """
    if source is None or target is None:
        return 0.0
    if len(source.faces) == 0 or len(target.faces) == 0:
        return 0.0
    if len(source.faces) > max_source_faces:
        # Sample faces uniformly to cap memory/time for unexpected large inputs.
        step = int(np.ceil(len(source.faces) / max_source_faces))
        face_idx = np.arange(0, len(source.faces), step)
    else:
        face_idx = np.arange(len(source.faces))
    try:
        centroids = source.triangles_center[face_idx]
        normals = source.face_normals[face_idx]
        closest, _, _ = trimesh.proximity.closest_point(target, centroids)
        vec = np.asarray(closest, dtype=np.float64) - np.asarray(centroids, dtype=np.float64)
        dist = np.linalg.norm(vec, axis=1)
        dist = np.where(dist < 1e-12, 1.0, dist)
        cos = np.einsum("ij,ij->i", vec / dist[:, None], normals)

        # Only faces that point toward the contact region are candidates.
        if contact_centroid is not None:
            to_contact = np.asarray(contact_centroid, dtype=np.float64) - np.asarray(
                source.centroid, dtype=np.float64
            )
            n_to = float(np.linalg.norm(to_contact))
            if n_to > 1e-12:
                to_contact = to_contact / n_to
                facing_contact = np.einsum("ij,j->i", normals, to_contact) > 0.5
            else:
                facing_contact = np.ones(len(normals), dtype=bool)
        else:
            facing_contact = np.ones(len(normals), dtype=bool)

        # Allow slightly angled contacts but discard back-facing surfaces.
        mask = facing_contact & (dist <= tol_mm) & (cos > 0.0)
        return float(source.area_faces[face_idx[mask]].sum())
    except Exception:
        return 0.0


def _contact_surface_area_m2(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
    contact_centroid: np.ndarray,
    tol_mm: float = 0.2,
    label: str = "",
    verbose: bool = False,
    contact_normal: Optional[np.ndarray] = None,
    contact_up_pt: Optional[np.ndarray] = None,
    contact_down_pt: Optional[np.ndarray] = None,
    contact_up_normal: Optional[np.ndarray] = None,
    contact_down_normal: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, int, int]:
    """Return the real throat/contact area (m²) between two meshes.

    The contact area is the physical shared boundary patch.  It is estimated by
    finding the boundary faces of the *smaller* body that are within ``tol_mm``
    of the other body and point toward it.  This directly captures T-junctions,
    angled contacts and complex geometry without a prescribed section plane.  If
    the face-proximity estimate is unavailable or implausible, we fall back to a
    section-based throat area along the smaller body's flow axis.
    """
    if mesh_a is None or mesh_b is None:
        return 0.0, 0.0, 0.0, 0, 0
    if len(mesh_a.faces) == 0 or len(mesh_b.faces) == 0:
        return 0.0, 0.0, 0.0, 0, 0

    centroid = np.asarray(contact_centroid, dtype=np.float64)

    # Pick the smaller mesh as the source for the face proximity query.
    try:
        vol_a = float(mesh_a.bounding_box.volume) if mesh_a.bounding_box else 0.0
    except Exception:
        vol_a = 0.0
    try:
        vol_b = float(mesh_b.bounding_box.volume) if mesh_b.bounding_box else 0.0
    except Exception:
        vol_b = 0.0
    if vol_a <= 0.0 and vol_b <= 0.0:
        vol_a = float(len(mesh_a.faces))
        vol_b = float(len(mesh_b.faces))

    if vol_a <= vol_b:
        small_mesh, large_mesh = mesh_a, mesh_b
        small_pt = np.asarray(contact_up_pt, dtype=np.float64) if contact_up_pt is not None else centroid
        small_normal = contact_up_normal
    else:
        small_mesh, large_mesh = mesh_b, mesh_a
        small_pt = np.asarray(contact_down_pt, dtype=np.float64) if contact_down_pt is not None else centroid
        small_normal = contact_down_normal

    # 1) Real shared contact-patch area from boundary-face proximity.  This is the
    # most faithful for angled / T-junction contacts because it uses the actual
    # surface faces that touch.
    # Use a generous tolerance for the face-proximity query; the normal alignment
    # test filters out faces that are merely close but not actually facing the
    # other body.
    # Flow axis and stable end cross-sections of both bodies on the smaller body's
    # throat plane.  This is the physically limiting cross-section of the contact.
    small_axis = _body_flow_axis(small_mesh, small_pt, small_normal)
    a_small_end = _throat_area_along_axis_mm2(
        small_mesh, small_pt, axis=small_axis, mode="first", min_area_mm2=1.0
    )
    a_large_end = _throat_area_along_axis_mm2(
        large_mesh, small_pt, axis=small_axis, mode="first", min_area_mm2=1.0
    )

    # 1) Try to use the actual shared surface patch.  It is trustworthy only when
    # it is a plausible fraction of the smaller body's throat; otherwise the query
    # has either under- or over-captured the contact.
    a_face = _face_contact_area_single_mm2(
        small_mesh,
        large_mesh,
        tol_mm=max(tol_mm, 2.0),
        contact_centroid=centroid,
    )
    # The shared contact patch cannot be larger than the smaller body's throat.
    # A small slack of 5 % is allowed for discretisation; anything substantially
    # larger means the proximity query has captured side/back faces.
    if (
        a_face > 1e-12
        and a_small_end > 1e-12
        and (0.6 * a_small_end <= a_face <= 1.05 * a_small_end)
    ):
        area_mm2 = float(a_face)
        a_a = a_face if small_mesh is mesh_a else a_small_end
        a_b = a_small_end if small_mesh is mesh_a else a_face
        chosen = "contact_faces"
        if verbose and label:
            print(
                f"[CONTACT_AREA] {label}: centroid=({centroid[0]:.2f},{centroid[1]:.2f},{centroid[2]:.2f}), "
                f"A_up={a_a:.3f} mm², A_down={a_b:.3f} mm², "
                f"A_contact={area_mm2:.3f} mm² ({chosen}, small_end={a_small_end:.3f})",
                flush=True,
            )
        return float(area_mm2 * 1e-6), a_a, a_b, 0, 0

    if verbose and label and a_face > 1e-12:
        print(
            f"[CONTACT_AREA] {label}: a_face={a_face:.3f} mm² rejected "
            f"(small_end={a_small_end:.3f}, large_end={a_large_end:.3f} mm²)",
            flush=True,
        )

    # 2) Fallback: the smaller of the two end cross-sections on the small-body
    # throat plane.  This correctly limits the throat even for T-junctions and
    # angled contacts as long as the smaller body's flow axis is identified.
    a_a = a_small_end if small_mesh is mesh_a else a_large_end
    a_b = a_large_end if small_mesh is mesh_a else a_small_end

    candidates = [a for a in (a_small_end, a_large_end) if a > 1e-12]
    if not candidates:
        # Last-ditch: use the global throat of either body.
        axis_a = _body_flow_axis(mesh_a, centroid, contact_up_normal)
        axis_b = _body_flow_axis(mesh_b, centroid, contact_down_normal)
        a_a_throat = _throat_area_along_axis_mm2(mesh_a, centroid, axis=axis_a)
        a_b_throat = _throat_area_along_axis_mm2(mesh_b, centroid, axis=axis_b)
        candidates = [a for a in (a_a_throat, a_b_throat) if a > 1e-12]
    if not candidates:
        if verbose and label:
            print(
                f"[CONTACT_AREA] {label}: centroid=({centroid[0]:.2f},{centroid[1]:.2f},{centroid[2]:.2f}) "
                f"-> A_contact=0.0 (section+throat failed)",
                flush=True,
            )
        return 0.0, a_a, a_b, 0, 0

    area_mm2 = float(min(candidates))
    chosen = "section_min"

    if verbose and label:
        print(
            f"[CONTACT_AREA] {label}: centroid=({centroid[0]:.2f},{centroid[1]:.2f},{centroid[2]:.2f}), "
            f"A_up={a_a:.3f} mm², A_down={a_b:.3f} mm², "
            f"A_contact={area_mm2:.3f} mm² ({chosen})",
            flush=True,
        )
    return float(area_mm2 * 1e-6), a_a, a_b, 0, 0


def _nearest_other_body_centroid(source_body, bodies: List) -> np.ndarray:
    """Return the centroid of the nearest other gating body to ``source_body``."""
    try:
        src_c = np.asarray(source_body.mesh.centroid, dtype=np.float64)
    except Exception:
        return np.zeros(3)
    best = None
    best_dist = float("inf")
    for b in bodies:
        if b is source_body or b is None or getattr(b, "mesh", None) is None or len(b.mesh.faces) == 0:
            continue
        try:
            bc = np.asarray(b.mesh.centroid, dtype=np.float64)
        except Exception:
            continue
        d = float(np.linalg.norm(bc - src_c))
        if d < best_dist:
            best_dist = d
            best = bc
    return best if best is not None else src_c


def _source_body_throat_area_mm2(source_body, g_u: np.ndarray) -> float:
    """Geometric throat area (mm²) of an isolated source body.

    The sprue/runner source body is usually an extruded shape: one OBB extent is
    the outlier (short for a disk-like throat, long for a long sprue/distributor).
    The throat is the minimum stable cross-section perpendicular to that outlier
    axis, swept from the upward-facing end toward the interior.  This captures the
    small inlet of a distributor, the narrow top of a tapered sprue, the circular
    face of a disk throat, and the bottom throat of a vertical sprue.
    """
    mesh = getattr(source_body, "mesh", None)
    if mesh is None or len(mesh.faces) == 0:
        return 0.0
    try:
        obb = mesh.bounding_box_oriented
        axes = np.asarray(obb.primitive.transform[:3, :3], dtype=np.float64)
        extents = np.asarray(obb.primitive.extents, dtype=np.float64)
        if len(extents) < 3:
            return 0.0
        med = float(np.median(extents))
        # Choose the outlier extent (most different from the median).
        devs = np.abs(extents - med)
        outlier = int(np.argmax(devs))
        # If all extents are very close, there is no clear extrusion axis.
        if devs.max() < 0.15 * max(med, 1e-9):
            return 0.0
        axis = np.asarray(axes[:, outlier], dtype=np.float64)
        axis = axis / (float(np.linalg.norm(axis)) + 1e-18)
        body_c = np.asarray(mesh.centroid, dtype=np.float64)
        half = float(extents[outlier]) * 0.5
        p1 = body_c + axis * half
        p2 = body_c - axis * half
        # The source inlet is the end that is most upward (opposite to gravity).
        up = -np.asarray(g_u, dtype=np.float64)
        up = up / (float(np.linalg.norm(up)) + 1e-18)
        proj1 = float(np.dot(p1, up))
        proj2 = float(np.dot(p2, up))
        if proj1 >= proj2:
            contact_pt = p1
        else:
            contact_pt = p2
            axis = -axis
        # Sweep the whole body length inward from this end.
        sweep = tuple(float(-s) for s in np.linspace(0.0, extents[outlier], 40))
        area = _throat_area_along_axis_mm2(
            mesh, contact_pt, axis=axis, sweep_mm=sweep, mode="min", min_area_mm2=5.0
        )
        return float(area) if area > 1e-6 else 0.0
    except Exception:
        return 0.0


def _legacy_cad_source_area_m2(
    bodies: List,
    section_key: str,
    g_u: np.ndarray,
) -> float:
    """Fallback: source body throat area from the whole-body OBB minimum section."""
    from core.types import BodyType

    section_to_types = {
        "SPRUE": [BodyType.SPRUE, BodyType.POURING_BASIN],
        "SPRUE_BASE": [BodyType.SPRUE],
        "SPRUE_THROAT": [BodyType.SPRUE_THROAT, BodyType.SPRUE, BodyType.POURING_BASIN],
        "POURING_BASIN": [BodyType.POURING_BASIN],
        "RUNNER": [BodyType.RUNNER],
        "DISTRIBUTOR": [BodyType.DISTRIBUTOR],
        "CURUFLUK": [BodyType.CURUFLUK],
        "FILTER": [BodyType.FILTER],
        "INGATE": [BodyType.INGATE],
        "RISER": [BodyType.RISER],
    }
    target_types = section_to_types.get((section_key or "SPRUE_THROAT").upper(),
                                        [BodyType.SPRUE_THROAT, BodyType.SPRUE, BodyType.POURING_BASIN])
    candidates = [
        b for b in bodies
        if getattr(b, "body_type", None) in target_types
        and getattr(b, "mesh", None) is not None
        and len(b.mesh.faces) > 0
    ]
    if not candidates:
        return 0.0

    def rank(b):
        c = np.asarray(b.mesh.centroid, dtype=np.float64)
        return float(-np.dot(c, g_u))

    best_area = 0.0
    for b in sorted(candidates, key=rank, reverse=True):
        area_mm2 = _source_body_throat_area_mm2(b, g_u)
        if area_mm2 > 0.0:
            best_area = max(best_area, float(area_mm2 * 1e-6))
        if best_area > 1e-12:
            break
    return float(best_area)


def cad_source_area_m2(
    bodies: List,
    section_key: str,
    g: np.ndarray,
) -> float:
    """Return the CAD source exit/contact area (m²) for the selected velocity section.

    The source body is found by the same CAD contact graph used by the Darcy
    solver, and its effective source area is the sum of the real throat/contact
    areas of its first downstream contacts.  This makes Q = v_design * A_source
    consistent with the actual geometric choke/exit, instead of an arbitrary
    whole-body cross-section.
    """
    from core.types import BodyType

    if not bodies:
        return 0.0

    g_u = np.asarray(g, dtype=np.float64)
    g_n = float(np.linalg.norm(g_u))
    if g_n > 1e-12:
        g_u = g_u / g_n
    else:
        g_u = np.array([0.0, 0.0, -1.0])

    part_candidates = [b for b in bodies if getattr(b, "body_type", None) == BodyType.PART]
    if not part_candidates:
        part_body = max(bodies, key=lambda b: float(getattr(b, "volume_cm3", 0.0) or 0.0))
    else:
        part_body = max(part_candidates, key=lambda b: float(getattr(b, "volume_cm3", 0.0) or 0.0))

    part_id = 1
    comp_meta: Dict[int, Tuple[BodyType, str]] = {
        part_id: (BodyType.PART, getattr(part_body, "name", "Parça"))
    }
    comp_body: Dict[int, Body] = {part_id: part_body}
    try:
        part_c = np.asarray(part_body.mesh.centroid, dtype=np.float64)
    except Exception:
        part_c = np.asarray(getattr(part_body, "center", np.zeros(3)), dtype=np.float64)
    comp_centroids: Dict[int, np.ndarray] = {part_id: part_c}

    next_id = 2
    for b in bodies:
        if b is part_body:
            continue
        comp_meta[next_id] = (getattr(b, "body_type", BodyType.EMPTY), getattr(b, "name", f"Body_{next_id}"))
        comp_body[next_id] = b
        try:
            c = np.asarray(b.mesh.centroid, dtype=np.float64)
        except Exception:
            c = np.asarray(getattr(b, "center", np.zeros(3)), dtype=np.float64)
        comp_centroids[next_id] = c
        next_id += 1

    try:
        contacts = _build_mesh_contacts(
            comp_body,
            comp_meta,
            comp_centroids,
            g_u,
            part_id,
            max_gap_mm=0.5,
            max_query=5000,
            verbose=False,
        )
    except Exception:
        contacts = []

    if not contacts:
        return _legacy_cad_source_area_m2(bodies, section_key, g_u)

    source_section = (section_key or "SPRUE_THROAT").upper()
    new_meta = _reclassify_comp_meta_from_graph(
        comp_meta, comp_centroids, contacts, part_id, g_u, source_section, verbose=False
    )

    # Build the part-reachable directed graph.
    adj: Dict[int, List[int]] = {cid: [] for cid in new_meta}
    incoming: Dict[int, List[int]] = {cid: [] for cid in new_meta}
    to_part: Dict[int, bool] = {cid: False for cid in new_meta}
    for c in contacts:
        up = int(c["up_id"])
        down = int(c["down_id"])
        if down == part_id:
            to_part[up] = True
            incoming[part_id].append(up)
            continue
        adj[up].append(down)
        incoming[down].append(up)

    can_reach_part: Dict[int, bool] = {part_id: True}
    queue = [part_id]
    while queue:
        cur = queue.pop(0)
        for prev in incoming.get(cur, []):
            if prev not in can_reach_part:
                can_reach_part[prev] = True
                queue.append(prev)

    indeg = {cid: 0 for cid in new_meta}
    outdeg = {cid: 0 for cid in new_meta}
    for c in contacts:
        up = int(c["up_id"])
        down = int(c["down_id"])
        if down == part_id:
            if can_reach_part.get(up, False):
                outdeg[up] += 1
            continue
        if can_reach_part.get(up, False) and can_reach_part.get(down, False):
            outdeg[up] += 1
            indeg[down] += 1

    section_to_types = {
        "SPRUE": {BodyType.SPRUE, BodyType.POURING_BASIN},
        "SPRUE_BASE": {BodyType.SPRUE},
        "SPRUE_THROAT": {BodyType.SPRUE_THROAT, BodyType.SPRUE, BodyType.POURING_BASIN},
        "POURING_BASIN": {BodyType.POURING_BASIN},
        "RUNNER": {BodyType.RUNNER},
        "DISTRIBUTOR": {BodyType.DISTRIBUTOR},
        "CURUFLUK": {BodyType.CURUFLUK},
        "FILTER": {BodyType.FILTER},
        "INGATE": {BodyType.INGATE},
        "RISER": {BodyType.RISER},
    }
    allowed_source_types = section_to_types.get(
        source_section,
        {BodyType.SPRUE_THROAT, BodyType.SPRUE, BodyType.POURING_BASIN},
    )

    def rank(cid: int) -> float:
        return float(-np.dot(comp_centroids.get(cid, np.zeros(3)), g_u))

    non_part = [cid for cid in new_meta if cid != part_id]
    candidates = [
        cid for cid in non_part
        if indeg[cid] == 0
        and outdeg[cid] > 0
        and can_reach_part.get(cid, False)
        and not to_part.get(cid, False)
        and new_meta[cid][0] in allowed_source_types
    ]
    if not candidates:
        candidates = [
            cid for cid in non_part
            if outdeg[cid] > 0
            and can_reach_part.get(cid, False)
            and not to_part.get(cid, False)
            and new_meta[cid][0] in allowed_source_types
        ]
    if not candidates:
        candidates = [
            cid for cid in non_part
            if outdeg[cid] > 0
            and can_reach_part.get(cid, False)
            and new_meta[cid][0] in allowed_source_types
        ]

    if not candidates:
        return _legacy_cad_source_area_m2(bodies, section_key, g_u)

    source_id = max(candidates, key=rank)

    total_area_m2 = 0.0
    for c in contacts:
        if int(c["up_id"]) != source_id:
            continue
        down = int(c["down_id"])
        if down != part_id and not can_reach_part.get(down, False):
            continue
        body_up = comp_body.get(source_id)
        body_down = comp_body.get(down)
        if body_up is None or body_down is None:
            continue
        if getattr(body_up, "mesh", None) is None or getattr(body_down, "mesh", None) is None:
            continue
        area_m2, _, _, _, _ = _contact_surface_area_m2(
            body_up.mesh,
            body_down.mesh,
            np.asarray(c["centroid_mm"], dtype=np.float64),
            tol_mm=0.5,
            contact_normal=np.asarray(c["normal"], dtype=np.float64) if c.get("normal") is not None else None,
            contact_up_pt=np.asarray(c["up_pt_mm"], dtype=np.float64) if c.get("up_pt_mm") is not None else None,
            contact_down_pt=np.asarray(c["down_pt_mm"], dtype=np.float64) if c.get("down_pt_mm") is not None else None,
            contact_up_normal=np.asarray(c["up_normal"], dtype=np.float64) if c.get("up_normal") is not None else None,
            contact_down_normal=np.asarray(c["down_normal"], dtype=np.float64) if c.get("down_normal") is not None else None,
        )
        total_area_m2 += max(float(area_m2), 0.0)

    if total_area_m2 > 1e-12:
        return float(total_area_m2)

    return _legacy_cad_source_area_m2(bodies, section_key, g_u)


def _origin_inside_body(
    mesh_pv: pv.PolyData,
    centroid: np.ndarray,
    interior_dir: np.ndarray,
    body_centroid: np.ndarray,
    eps: float,
) -> Optional[np.ndarray]:
    """Find a point inside ``mesh_pv`` along the ray from ``centroid`` to ``body_centroid``.

    The contact centroid is sometimes just outside the throat body (e.g. the
    voxel contact sits on the upstream side).  This helper marches from the
    contact toward the body interior and returns a point a small ``eps``
    inside the surface.
    """
    direction = np.asarray(interior_dir, dtype=np.float64)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-18)
    dist = float(np.linalg.norm(np.asarray(body_centroid) - np.asarray(centroid)))
    if dist <= 1e-9:
        return None
    # Sample the ray and use PyVista to find the first inside point.
    n_samples = 200
    ts = np.linspace(0.0, dist, n_samples)
    points = np.asarray(centroid)[None, :] + ts[:, None] * direction[None, :]
    try:
        cloud = pv.PolyData(points)
        result = cloud.select_interior_points(surface=mesh_pv)
        sel = result["selected_points"].astype(bool)
    except Exception:
        return None
    idx = int(np.argmax(sel))
    if not sel[idx]:
        return None
    origin = points[idx] + direction * eps
    return origin


def _body_throat_area_m2(
    mesh: Optional[trimesh.Trimesh],
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
) -> Optional[float]:
    """Pure geometric cross-section, no fallback, no sweep.

    Plane passes through plane_origin, perpendicular to plane_normal (body axis).
    Returns area in m², or None if section fails.
    """
    if mesh is None or len(mesh.faces) == 0:
        return None
    origin = np.asarray(plane_origin, dtype=np.float64)
    normal = np.asarray(plane_normal, dtype=np.float64)
    n = float(np.linalg.norm(normal))
    if n < 1e-18:
        return None
    normal = normal / n
    try:
        section = mesh.section(plane_origin=origin, plane_normal=normal)
        if section is None or getattr(section, "is_empty", True):
            return None
        p2d = section.to_2D()
        if isinstance(p2d, tuple):
            p2d = p2d[0]
        if p2d is None or not getattr(p2d, "polygons_closed", None):
            return None
        polys = getattr(p2d, "polygons_closed", None)
        if not polys:
            return None
        area_mm2 = float(sum(float(p.area) for p in polys))
        if area_mm2 <= 1e-12:
            return None
        return area_mm2 * 1e-6
    except Exception:
        return None


def _inlet_flux_m3_s(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    source: np.ndarray,
    cavity: np.ndarray,
    dx: float,
    face_fractions: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> float:
    """Total flux (m³/s) leaving the source mask into the rest of the cavity.

    The face area is ``dx*dx * f_A`` so only the real open fraction contributes
    to the volumetric flow rate on curved/staircase geometry.
    """
    if face_fractions is None:
        nz, ny, nx = cavity.shape
        f_A_z = np.ones((nz + 1, ny, nx), dtype=np.float64)
        f_A_y = np.ones((nz, ny + 1, nx), dtype=np.float64)
        f_A_x = np.ones((nz, ny, nx + 1), dtype=np.float64)
    else:
        f_A_z, f_A_y, f_A_x = face_fractions
    area = dx * dx
    flux = 0.0

    # z-faces (axis 0) -- face k is between cells k-1 and k
    left_source = source[:-1] & ~source[1:] & cavity[1:]
    right_source = source[1:] & ~source[:-1] & cavity[:-1]
    uz = u[1:-1]
    a_z = f_A_z[1:-1] * area
    flux += float((uz * a_z)[left_source].sum())
    flux -= float((uz * a_z)[right_source].sum())

    # y-faces (axis 1)
    down_source = source[:, :-1] & ~source[:, 1:] & cavity[:, 1:]
    up_source = source[:, 1:] & ~source[:, :-1] & cavity[:, :-1]
    vy = v[:, 1:-1]
    a_y = f_A_y[:, 1:-1, :] * area
    flux += float((vy * a_y)[down_source].sum())
    flux -= float((vy * a_y)[up_source].sum())

    # x-faces (axis 2)
    back_source = source[:, :, :-1] & ~source[:, :, 1:] & cavity[:, :, 1:]
    front_source = source[:, :, 1:] & ~source[:, :, :-1] & cavity[:, :, :-1]
    wx = w[:, :, 1:-1]
    a_x = f_A_x[:, :, 1:-1] * area
    flux += float((wx * a_x)[back_source].sum())
    flux -= float((wx * a_x)[front_source].sum())

    return flux


def _mask_inlet_by_section(
    grid: np.ndarray,
    section_key: str,
) -> np.ndarray:
    """Return a cell mask for the section used to derive Q from user velocity."""
    key = (section_key or "SPRUE").upper()
    type_map = {
        "SPRUE": [BodyType.SPRUE, BodyType.POURING_BASIN],
        "SPRUE_BASE": [BodyType.SPRUE],
        "SPRUE_THROAT": [BodyType.SPRUE_THROAT, BodyType.SPRUE, BodyType.POURING_BASIN],
        "POURING_BASIN": [BodyType.POURING_BASIN],
        "RUNNER": [BodyType.RUNNER],
        "DISTRIBUTOR": [BodyType.DISTRIBUTOR],
        "CURUFLUK": [BodyType.CURUFLUK],
        "INGATE": [BodyType.INGATE],
        "FILTER": [BodyType.FILTER],
        "RISER": [BodyType.RISER],
    }
    types = type_map.get(key, [BodyType.SPRUE])
    mask = np.zeros_like(grid, dtype=bool)
    for bt in types:
        mask |= grid == bt
    return mask


def _section_face_cells(
    grid: np.ndarray,
    cavity: np.ndarray,
    section_key: str,
    g: np.ndarray,
    allow_fallback: bool = True,
) -> Tuple[np.ndarray, str]:
    """Return a cell mask for the representative cross-section of a section.

    For sprue/runner/distributor/curufluk/filter the upstream face is used;
    for the ingate the downstream face (interface to the part) is used.
    """
    section_mask = _mask_inlet_by_section(grid, section_key) & cavity
    used_section = (section_key or "SPRUE").upper()
    if not section_mask.any() and allow_fallback:
        # fall back to the smallest-area gating section
        candidates = [
            BodyType.SPRUE,
            BodyType.RUNNER,
            BodyType.DISTRIBUTOR,
            BodyType.INGATE,
        ]
        for bt in candidates:
            section_mask = (grid == bt) & cavity
            if section_mask.any():
                used_section = BodyType(bt).name
                break

    # Determine whether to use the upstream or downstream boundary.
    # The user-specified velocity is measured at the entrance, so for the
    # sprue throat / sprue / runner / distributor we use the upstream face.
    # For the ingate the velocity is usually quoted at the ingate exit into
    # the part, hence the downstream face.
    key = used_section.upper()
    if key in ("INGATE",):
        side = "down"
    elif key == "SPRUE_BASE":
        side = "down"
    else:
        side = "up"
    face = _find_boundary_cells_along(section_mask, g, side=side) & cavity
    if not face.any():
        face = _find_boundary_cells_along(section_mask, g, side="up") & cavity
    return face, used_section


def _compute_user_flow_rate(
    grid: np.ndarray,
    cavity: np.ndarray,
    dx_m: float,
    velocity_m_s: float,
    section_key: str,
    fill_time_s: float,
    part_volume_m3: float,
    g: np.ndarray,
    design_velocity_m_s: float = 0.0,
    design_section_key: str = "SPRUE_THROAT",
    design_area_m2: float = 0.0,
    fine_grid: Optional[np.ndarray] = None,
    fine_cavity: Optional[np.ndarray] = None,
    fine_dx_m: Optional[float] = None,
) -> Tuple[float, float, str]:
    """Convert user velocity, fill time, or design velocity into a total flow rate Q.

    No automatic fallback: if the requested section has no measurable cross-section
    and no design area is supplied, the solver stops with a clear error.
    """
    a_grid = fine_grid if fine_grid is not None else grid
    a_cavity = fine_cavity if fine_cavity is not None else cavity
    a_dx = fine_dx_m if fine_dx_m is not None else dx_m

    used_section = (section_key or "SPRUE").upper()
    design_section = (design_section_key or used_section).upper()

    if velocity_m_s > 0.0:
        if design_area_m2 > 1e-18:
            area_m2 = float(design_area_m2)
            used_section = design_section
        else:
            face, used_section = _section_face_cells(a_grid, a_cavity, section_key, g, allow_fallback=False)
            area_m2 = float(face.sum()) * (a_dx * a_dx)
        if area_m2 <= 0.0:
            raise GatingVelocityError(
                f"Giriş debisi hesaplanamadı: {used_section} kesit alanı ölçülemedi ve "
                f"tasarım alanı verilmedi."
            )
        Q = velocity_m_s * area_m2
        return Q, area_m2, used_section

    if fill_time_s > 0.0 and part_volume_m3 > 0.0:
        face, used_section = _section_face_cells(a_grid, a_cavity, section_key, g, allow_fallback=False)
        area_m2 = float(face.sum()) * (a_dx * a_dx)
        Q = part_volume_m3 / fill_time_s
        return Q, area_m2, used_section

    if design_velocity_m_s > 0.0:
        used_section = design_section
        if design_area_m2 > 1e-18:
            area_m2 = float(design_area_m2)
        else:
            face, _ = _section_face_cells(a_grid, a_cavity, design_section_key, g, allow_fallback=False)
            area_m2 = float(face.sum()) * (a_dx * a_dx)
        if area_m2 <= 0.0:
            raise GatingVelocityError(
                f"Tasarım debisi hesaplanamadı: {used_section} kesit alanı ölçülemedi."
            )
        Q = design_velocity_m_s * area_m2
        return Q, area_m2, used_section

    raise GatingVelocityError(
        "Akış girişi eksik: ne hız (ingate_velocity_m_s), ne doldurma süresi "
        "(t_fill_s), ne de tasarım hızı (design_choke_velocity_m_s) verilmemiş."
    )


def _compute_fill_time(
    vmag: np.ndarray,
    cavity: np.ndarray,
    inlet_cells: np.ndarray,
    dx_m: float,
) -> np.ndarray:
    """Approximate front-arrival time (s) for each voxel from the inlet.

    A 6-neighbour fast marching with speed |v| is used: dt = dx / (0.5*(|v_i|+|v_j|)).
    Unreached / stagnant cells are left at np.inf; non-cavity cells are 0.
    """
    shape = cavity.shape
    fill = np.full(shape, np.inf, dtype=np.float64)
    fill[inlet_cells] = 0.0
    if not inlet_cells.any():
        fill[~cavity] = 0.0
        return fill

    # Seed all inlet cells
    heap = [(0.0, int(i), int(j), int(k)) for i, j, k in zip(*np.where(inlet_cells))]
    heapq.heapify(heap)
    visited = np.zeros(shape, dtype=bool)

    # 6-neighbour front propagation on the resolved flow grid.  Face-only
    # connections keep the fast-marching heap small; the Darcy pressure solve
    # already resolved the 3-D velocity field, so this is just an arrival-time
    # estimate.
    neighbours = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]

    while heap:
        t, i, j, k = heapq.heappop(heap)
        if visited[i, j, k]:
            continue
        visited[i, j, k] = True
        if t > fill[i, j, k] + 1e-12:
            continue
        for di, dj, dk in neighbours:
            ni, nj, nk = i + di, j + dj, k + dk
            if not (0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]):
                continue
            if visited[ni, nj, nk] or not cavity[ni, nj, nk]:
                continue
            v_avg = 0.5 * (max(vmag[i, j, k], 1e-6) + max(vmag[ni, nj, nk], 1e-6))
            dist = float(np.sqrt(di * di + dj * dj + dk * dk))
            dt = (dx_m * dist) / v_avg
            t_new = t + dt
            if t_new < fill[ni, nj, nk]:
                fill[ni, nj, nk] = t_new
                heapq.heappush(heap, (t_new, ni, nj, nk))

    fill[~cavity] = 0.0
    return fill


def _compute_fill_time_graph(
    grid: np.ndarray,
    origin_mm: np.ndarray,
    dx_mm: float,
    g: np.ndarray,
    gating_nodes: List[GatingNode],
    bodies: List[Body],
    body_index: Optional[np.ndarray],
    fill_time_s: float,
) -> np.ndarray:
    """Compute voxel front-arrival times from the gating graph.

    The Darcy velocity field is not used for timing; instead, the BFS path through
    the gating graph and the already-computed node velocities / flow rates drive a
    simple 1-D plug-flow estimate in each body.  The part is filled outward from
    each ingate so that the farthest metal cell reaches exactly ``fill_time_s``.
    """
    shape = grid.shape
    fill = np.full(shape, np.inf, dtype=np.float64)
    cavity = (grid > 0) & (grid != int(BodyType.CORE))
    if not cavity.any() or fill_time_s <= 1e-12:
        fill[~cavity] = 0.0
        fill[grid == int(BodyType.CORE)] = np.inf
        return fill

    g_u = np.asarray(g, dtype=np.float64)
    if np.linalg.norm(g_u) > 1e-12:
        g_u = g_u / np.linalg.norm(g_u)
    else:
        g_u = np.array([0.0, 0.0, -1.0])

    def _fallback_fill() -> np.ndarray:
        """Last-resort radial fill from the source region when the gating graph cannot be used."""
        source_mask = np.zeros(shape, dtype=bool)
        for bt in (
            BodyType.POURING_BASIN,
            BodyType.SPRUE_THROAT,
            BodyType.SPRUE,
            BodyType.RUNNER,
            BodyType.DISTRIBUTOR,
        ):
            source_mask |= grid == int(bt)
        source_mask &= cavity
        if not source_mask.any():
            idx = np.argwhere(cavity)
            centers = origin_mm + (idx + 0.5) * dx_mm
            proj = centers @ (-g_u)
            top_n = max(1, int(0.01 * idx.shape[0]))
            top_idx = idx[np.argpartition(-proj, top_n - 1)[:top_n]]
            source_mask = np.zeros(shape, dtype=bool)
            source_mask[tuple(top_idx.T)] = True
        dist = ndimage.distance_transform_edt(~source_mask)
        dist_m = dist * dx_mm / 1000.0
        max_m = dist_m[cavity].max() if cavity.any() else 0.0
        if max_m < 1e-9:
            return np.where(cavity, fill_time_s, 0.0)
        return np.where(cavity, (dist_m / max_m) * fill_time_s, 0.0)

    if (
        not gating_nodes
        or body_index is None
        or body_index.shape != grid.shape
        or not bodies
    ):
        return _fallback_fill()

    name_to_bidx = {b.name: i for i, b in enumerate(bodies)}
    part_bidx = None
    for i, b in enumerate(bodies):
        if b.body_type == BodyType.PART:
            part_bidx = i
            break

    children: Dict[int, List[Dict]] = {}
    source_bidx: Optional[int] = None
    source_v = 0.0
    source_q = 0.0
    source_centroid = np.zeros(3, dtype=np.float64)
    ingate_branches: List[Dict] = []

    for node in gating_nodes:
        if "→" not in node.name or "→" not in node.body_type:
            continue
        up_name, down_name = [s.strip() for s in node.name.split("→")]
        up_type, down_type = [s.strip() for s in node.body_type.split("→")]
        if up_name == "Kaynak" or "SOURCE" in up_type:
            source_bidx = name_to_bidx.get(down_name)
            source_v = float(
                node.max_velocity_m_s
                if node.max_velocity_m_s > 1e-12
                else node.velocity_m_s
            )
            source_q = float(node.flow_rate_m3_s)
            source_centroid = np.asarray(node.centroid_mm, dtype=np.float64)
            continue
        up_bidx = name_to_bidx.get(up_name)
        if up_bidx is None:
            continue
        # The downstream name is the actual body name except for the part, which
        # the gating pipeline labels as "Parça".
        if down_name == "Parça" or down_type == "PART":
            down_bidx = part_bidx
        else:
            down_bidx = name_to_bidx.get(down_name)
        if down_bidx is None:
            continue
        child = {
            "bidx": down_bidx,
            "point": np.asarray(node.centroid_mm, dtype=np.float64),
            "v": float(
                node.max_velocity_m_s
                if node.max_velocity_m_s > 1e-12
                else node.velocity_m_s
            ),
            "q": float(node.flow_rate_m3_s),
        }
        children.setdefault(up_bidx, []).append(child)
        if down_bidx == part_bidx:
            ingate_branches.append(child.copy())
            ingate_branches[-1]["parent_bidx"] = up_bidx

    if source_bidx is None or part_bidx is None:
        return _fallback_fill()

    # Source entry point: highest cell along -gravity within the source body.
    src_mask = body_index == source_bidx
    src_entry = source_centroid
    if src_mask.any():
        src_idx = np.argwhere(src_mask)
        src_centers = origin_mm + (src_idx + 0.5) * dx_mm
        proj = src_centers @ (-g_u)
        top_idx = src_idx[np.argmax(proj)]
        src_entry = origin_mm + (top_idx + 0.5) * dx_mm

    t_entry: Dict[int, float] = {source_bidx: 0.0}
    entry_point: Dict[int, np.ndarray] = {source_bidx: src_entry.copy()}
    v_in: Dict[int, float] = {source_bidx: max(source_v, 1e-6)}

    # BFS over the directed gating graph (body indices).
    queue = [source_bidx]
    visited = {source_bidx}
    while queue:
        bidx = queue.pop(0)
        for child in children.get(bidx, []):
            child_bidx = child["bidx"]
            if child_bidx in visited:
                continue
            visited.add(child_bidx)
            P = child["point"]
            entry = entry_point[bidx]
            v = v_in[bidx]
            L = float(np.linalg.norm(P - entry))
            t_entry[child_bidx] = t_entry[bidx] + (L / 1000.0 / v if v > 1e-18 else 0.0)
            entry_point[child_bidx] = P.copy()
            v_in[child_bidx] = max(child["v"], 1e-6)
            queue.append(child_bidx)

    # Synchronize sibling gates fed by the same parent: wait until the metal
    # front reaches the farthest gate so they all start filling together.  This
    # avoids the visually absurd "first one gate then another" sequence.
    if t_entry:
        parent_to_children: Dict[int, List[int]] = {}
        for parent_bidx, childs in children.items():
            for child in childs:
                parent_to_children.setdefault(parent_bidx, []).append(child["bidx"])
        for child_bidx_list in parent_to_children.values():
            if len(child_bidx_list) <= 1:
                continue
            valid = [c for c in child_bidx_list if c in t_entry]
            if not valid:
                continue
            max_t = max(t_entry[c] for c in valid)
            for c in valid:
                t_entry[c] = max_t

    # If the graph never reaches the part, fall back to a radial fill so the
    # animation is not completely blank.
    if part_bidx not in visited:
        return _fallback_fill()

    # Compute cell centres once and cache per body.
    def _centres(mask: np.ndarray) -> np.ndarray:
        idx = np.argwhere(mask)
        return origin_mm + (idx + 0.5) * dx_mm, idx

    # Fill time inside each gating body (use body_index, not grid body type, so
    # graph-reclassified bodies are still timed correctly).
    gating_bidx = visited - {part_bidx}
    for bidx in gating_bidx:
        mask = body_index == bidx
        if not mask.any():
            continue
        centers, idx = _centres(mask)
        t0 = t_entry[bidx]
        v = v_in[bidx]
        childs = children.get(bidx, [])
        if not childs:
            # No downstream child: flow toward the body centroid.
            body = bodies[bidx]
            try:
                centroid = np.asarray(body.mesh.centroid, dtype=np.float64)
            except Exception:
                centroid = centers.mean(axis=0)
            axis = centroid - entry_point[bidx]
            L = float(np.linalg.norm(axis))
            if L < 1e-9:
                axis = -g_u
                L = 1.0
            dir_u = axis / L
            proj = (centers - entry_point[bidx]) @ dir_u
            proj = np.clip(proj, 0.0, L)
            fill[mask] = t0 + proj / 1000.0 / v
            continue
        t_vals = np.full(idx.shape[0], np.inf, dtype=np.float64)
        for child in childs:
            P = child["point"]
            axis = P - entry_point[bidx]
            L = float(np.linalg.norm(axis))
            if L < 1e-9:
                body = bodies[bidx]
                try:
                    centroid = np.asarray(body.mesh.centroid, dtype=np.float64)
                except Exception:
                    centroid = centers.mean(axis=0)
                axis = centroid - entry_point[bidx]
                L = float(np.linalg.norm(axis))
            if L < 1e-9:
                axis = -g_u
                L = 1.0
            dir_u = axis / L
            proj = (centers - entry_point[bidx]) @ dir_u
            proj = np.clip(proj, 0.0, L)
            t_vals = np.minimum(t_vals, t0 + proj / 1000.0 / v)
        fill[mask] = t_vals

    # The source body is treated as already full at t=0 so the initial frame is
    # visible and the animation starts from a real pour entry volume.
    if source_bidx is not None:
        fill[body_index == source_bidx] = 0.0

    # Part: outward fill from each ingate. Exclude cells that belong to bodies the
    # gating graph has reclassified as part of the gating system.
    gating_bidx_arr = np.fromiter(gating_bidx, dtype=np.int64, count=len(gating_bidx))
    on_gating = np.isin(body_index, gating_bidx_arr)
    part_mask = (grid == BodyType.PART) & ~on_gating
    if part_mask.any() and ingate_branches:
        centers, _ = _centres(part_mask)
        part_fill = np.full(centers.shape[0], np.inf, dtype=np.float64)
        for branch in ingate_branches:
            P = branch["point"]
            parent_bidx = branch.get("parent_bidx")
            if parent_bidx is None or parent_bidx not in t_entry:
                continue
            ingate_entry = entry_point[parent_bidx]
            v_ingate = v_in[parent_bidx]
            L_ingate = float(np.linalg.norm(P - ingate_entry))
            t_ingate = t_entry[parent_bidx] + (L_ingate / 1000.0 / v_ingate if v_ingate > 1e-18 else 0.0)
            dists = np.linalg.norm(centers - P, axis=1)
            max_dist = float(dists.max()) if dists.size else 0.0
            if max_dist > 1e-12 and fill_time_s > t_ingate:
                v_eff = (max_dist / 1000.0) / (fill_time_s - t_ingate)
            else:
                v_eff = max(branch["v"], 1e-6)
            part_fill = np.minimum(part_fill, t_ingate + dists / 1000.0 / v_eff)
        # Volume-correct the part so its latest cell reaches exactly fill_time_s.
        if part_fill.size:
            pmin = float(part_fill.min())
            pmax = float(part_fill.max())
            if pmax > pmin + 1e-12 and fill_time_s > pmin:
                scale = (fill_time_s - pmin) / (pmax - pmin)
                part_fill = pmin + (part_fill - pmin) * scale
            else:
                part_fill = np.full_like(part_fill, fill_time_s)
        fill[part_mask] = part_fill
    else:
        fill[part_mask] = fill_time_s

    # Risers/feeders fill after the part using the same source velocity.
    # For each riser cell, start at the fill time of the nearest already-filled
    # non-riser metal cell and add the travel distance / velocity into the riser.
    source_v_safe = max(source_v, 1e-6)
    riser_bidx = [i for i, b in enumerate(bodies) if getattr(b, "body_type", None) == BodyType.RISER]
    if riser_bidx:
        riser_mask = np.isin(body_index, np.fromiter(riser_bidx, dtype=np.int64, count=len(riser_bidx)))
        if riser_mask.any():
            non_riser_metal = cavity & ~riser_mask
            if non_riser_metal.any():
                dist, indices = ndimage.distance_transform_edt(
                    ~non_riser_metal, return_indices=True, return_distances=True
                )
                nearest_fill = fill[indices[0], indices[1], indices[2]]
                travel_m = dist[riser_mask] * dx_mm / 1000.0
                fill[riser_mask] = nearest_fill[riser_mask] + travel_m / source_v_safe
            else:
                fill[riser_mask] = fill_time_s

    fill[~cavity] = 0.0
    # Clamp any numerical overshoot in non-riser cells; risers may legitimately
    # fill after the part and therefore exceed fill_time_s.
    non_riser = cavity & ~riser_mask if riser_bidx else cavity
    fill[np.isfinite(fill) & non_riser] = np.minimum(fill[np.isfinite(fill) & non_riser], fill_time_s)
    # Any remaining metal cells that are not reached by the source graph (e.g.
    # cooling sprues that sit above the part) still need a finite time.
    inf_metal = np.isinf(fill) & cavity
    if inf_metal.any():
        fill[inf_metal] = fill_time_s

    # If the graph produced an almost-flat field (every cell is at fill_time_s),
    # the body_index / grid mapping is broken for this model.  Fall back to a
    # radial fill from the source region so the animation is still visible.
    if cavity.any():
        fmin = float(fill[cavity].min())
        fmax = float(fill[cavity].max())
        if (fmax - fmin) < 1e-3 * fill_time_s or fmin > 0.9 * fill_time_s:
            return _fallback_fill()

    return fill


def _geodesic_gating_time(
    grid: np.ndarray,
    body_index: np.ndarray,
    bodies: List[Body],
    gating_nodes: List[GatingNode],
    source_mask: np.ndarray,
    dx_mm: float,
) -> np.ndarray:
    """Geodesic arrival time (seconds) from the source through the gating system.

    Each connected gating body is treated as a conduit with a local front speed
    derived from the gating-node velocities.  The shortest-path distance within
    the metal mask, divided by the local speed, gives a realistic time for the
    metal front to travel from the source through sprues, runners and ingates.
    """
    shape = grid.shape
    T = np.full(shape, np.inf, dtype=np.float64)
    gating_types = {
        int(BodyType.SPRUE),
        int(BodyType.SPRUE_THROAT),
        int(BodyType.POURING_BASIN),
        int(BodyType.RUNNER),
        int(BodyType.DISTRIBUTOR),
        int(BodyType.INGATE),
    }
    gating_mask = (
        np.isin(grid, list(gating_types))
        & (body_index >= 0)
        & (grid != int(BodyType.CORE))
    )
    if not gating_mask.any():
        return T

    n_bodies = len(bodies)
    name_to_bidx = {b.name: i for i, b in enumerate(bodies)}
    q_sum = np.zeros(n_bodies, dtype=np.float64)
    a_sum = np.zeros(n_bodies, dtype=np.float64)
    for node in gating_nodes:
        if "→" not in node.name or "→" not in node.body_type:
            continue
        up_name, _ = [s.strip() for s in node.name.split("→")]
        up_bidx = name_to_bidx.get(up_name)
        if up_bidx is None:
            continue
        q = float(node.flow_rate_m3_s)
        v = float(
            node.max_velocity_m_s
            if node.max_velocity_m_s > 1e-12
            else node.velocity_m_s
        )
        if v <= 1e-18:
            continue
        q_sum[up_bidx] += q
        a_sum[up_bidx] += q / v

    body_speed = np.zeros(n_bodies, dtype=np.float64)
    valid = a_sum > 1e-18
    if valid.any():
        body_speed[valid] = q_sum[valid] / a_sum[valid]
    avg_speed = float(np.mean(body_speed[body_speed > 1e-18])) if np.any(body_speed > 1e-18) else 1.5
    body_speed = np.where(body_speed > 1e-18, body_speed, avg_speed)

    cell_bidx = body_index[gating_mask]
    cell_speed = body_speed[cell_bidx]
    speed_grid = np.zeros(shape, dtype=np.float64)
    speed_grid[gating_mask] = cell_speed

    idx = np.full(shape, -1, dtype=np.int64)
    idx[gating_mask] = np.arange(int(gating_mask.sum()))
    N = int(gating_mask.sum())

    source_indices = idx[source_mask & gating_mask]
    if source_indices.size == 0:
        # Use the top-most (against gravity) gating cells as fallback seed.
        return T

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    dx_m = float(dx_mm) / 1000.0

    # x-faces
    valid_x = (idx[:-1, :, :] >= 0) & (idx[1:, :, :] >= 0)
    r = idx[:-1, :, :][valid_x]
    c = idx[1:, :, :][valid_x]
    v_face = 0.5 * (speed_grid[:-1, :, :][valid_x] + speed_grid[1:, :, :][valid_x])
    w = dx_m / np.maximum(v_face, 1e-6)
    rows.extend([r, c])
    cols.extend([c, r])
    weights.extend([w, w])

    # y-faces
    valid_y = (idx[:, :-1, :] >= 0) & (idx[:, 1:, :] >= 0)
    r = idx[:, :-1, :][valid_y]
    c = idx[:, 1:, :][valid_y]
    v_face = 0.5 * (speed_grid[:, :-1, :][valid_y] + speed_grid[:, 1:, :][valid_y])
    w = dx_m / np.maximum(v_face, 1e-6)
    rows.extend([r, c])
    cols.extend([c, r])
    weights.extend([w, w])

    # z-faces
    valid_z = (idx[:, :, :-1] >= 0) & (idx[:, :, 1:] >= 0)
    r = idx[:, :, :-1][valid_z]
    c = idx[:, :, 1:][valid_z]
    v_face = 0.5 * (speed_grid[:, :, :-1][valid_z] + speed_grid[:, :, 1:][valid_z])
    w = dx_m / np.maximum(v_face, 1e-6)
    rows.extend([r, c])
    cols.extend([c, r])
    weights.extend([w, w])

    if not rows:
        return T

    rows_a = np.concatenate(rows)
    cols_a = np.concatenate(cols)
    weights_a = np.concatenate(weights)
    graph = csr_matrix((weights_a, (rows_a, cols_a)), shape=(N, N))
    dist = csgraph.dijkstra(graph, directed=False, indices=source_indices, return_predecessors=False)
    if dist.ndim == 1:
        dist = dist.reshape(1, -1)
    min_dist = np.min(dist, axis=0)
    T[gating_mask] = min_dist
    return T


def _gating_volume_time(
    grid: np.ndarray,
    body_index: np.ndarray,
    bodies: List[Body],
    origin_mm: np.ndarray,
    dx_mm: float,
    source_bidx: int,
    source_q: float,
    source_centroid: np.ndarray,
    children: Dict[int, List[Dict]],
    q_in: Dict[int, float],
    entry_point: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return gating fill times and per-body entry/exit times.

    Each gating body is treated as a plug-flow vessel: the front reaches the
    body exit after the body volume has been filled, ``t_exit = t_enter + V/Q``.
    For a common manifold (one parent, several children), all children start
    at the same ``t_exit`` of the parent, so sibling ingates open together.
    """
    shape = grid.shape
    n_bodies = len(bodies)
    dx_m = dx_mm / 1000.0
    cell_vol = dx_m ** 3
    cavity = (grid > 0) & (grid != int(BodyType.CORE))

    # Body volumes from the voxel grid.
    body_vol = np.zeros(n_bodies, dtype=np.float64)
    for b in range(n_bodies):
        body_vol[b] = float(np.sum((body_index == b) & cavity)) * cell_vol

    # Build parent map and weighted entry points for merged bodies.
    parents: Dict[int, List[Tuple[int, float]]] = {}
    entry_wsum: Dict[int, Tuple[np.ndarray, float]] = {}
    for parent, childs in children.items():
        for c in childs:
            cb = c["bidx"]
            q = float(c["q"])
            point = np.asarray(c["point"], dtype=np.float64)
            if cb is None:
                continue
            parents.setdefault(cb, []).append((parent, q))
            ptsum, wsum = entry_wsum.get(cb, (np.zeros(3, dtype=np.float64), 0.0))
            entry_wsum[cb] = (ptsum + point * q, wsum + q)
    for b, (ptsum, wsum) in entry_wsum.items():
        if wsum > 1e-18:
            entry_point[b] = ptsum / wsum
    entry_point[source_bidx] = source_centroid

    # Default exit point = body centroid; for bodies with children use the
    # flow-weighted average of the child contact points.
    exit_point: Dict[int, np.ndarray] = {}
    for b in range(n_bodies):
        idx = np.argwhere((body_index == b) & cavity)
        if idx.size:
            centers = origin_mm + (idx + 0.5) * dx_mm
            exit_point[b] = centers.mean(axis=0)
        else:
            exit_point[b] = np.zeros(3, dtype=np.float64)
    for parent, childs in children.items():
        if not childs:
            continue
        ptsum = np.zeros(3, dtype=np.float64)
        wsum = 0.0
        for c in childs:
            q = float(c["q"])
            ptsum += np.asarray(c["point"], dtype=np.float64) * q
            wsum += q
        if wsum > 1e-18:
            exit_point[parent] = ptsum / wsum

    # Through-flow per body and gating arrival times.
    t_enter = np.full(n_bodies, np.inf, dtype=np.float64)
    t_exit = np.full(n_bodies, np.inf, dtype=np.float64)

    if os.environ.get("JOSECAST_USE_CPP_GATING", "1") != "0":
        try:
            from core.cpp_bridge import JOSECAST_CORE
            if JOSECAST_CORE is not None:
                edge_rows = []
                edge_qs = []
                for parent, childs in children.items():
                    for c in childs:
                        cb = c.get("bidx")
                        if cb is None:
                            continue
                        edge_rows.append([int(parent), int(cb)])
                        edge_qs.append(float(c["q"]))
                if edge_rows:
                    edges = np.asarray(edge_rows, dtype=np.int32).reshape(-1, 2)
                    edge_q = np.asarray(edge_qs, dtype=np.float64)
                else:
                    edges = np.empty((0, 2), dtype=np.int32)
                    edge_q = np.empty(0, dtype=np.float64)
                q_in_arr = np.zeros(n_bodies, dtype=np.float64)
                for b, q in q_in.items():
                    q_in_arr[int(b)] = float(q)
                t_enter, t_exit, _ = JOSECAST_CORE.solve_gating_times(
                    int(source_bidx), float(source_q), body_vol, edges, edge_q, q_in_arr
                )
        except Exception as exc:
            print(f"[Gating] C++ solver fallback: {exc}")

    if not np.isfinite(t_exit[source_bidx]):
        # Fallback Python DFS (or if C++ path was skipped/failed).
        Q = np.zeros(n_bodies, dtype=np.float64)
        Q[source_bidx] = source_q
        for b, flows in parents.items():
            Q[b] = sum(q for _, q in flows)
        for b, q in q_in.items():
            Q[b] = max(Q[b], q)

        visited = np.zeros(n_bodies, dtype=bool)

        def _visit(bidx: int) -> None:
            if bidx < 0 or bidx >= n_bodies or visited[bidx]:
                return
            visited[bidx] = True
            if bidx == source_bidx:
                t_enter[bidx] = 0.0
            else:
                parent_exits = [t_exit[p] for p, _ in parents.get(bidx, []) if np.isfinite(t_exit[p])]
                t_enter[bidx] = float(max(parent_exits)) if parent_exits else 0.0
            t_exit[bidx] = t_enter[bidx] + body_vol[bidx] / max(Q[bidx], 1e-18)
            for c in children.get(bidx, []):
                cb = c.get("bidx")
                if cb is not None:
                    _visit(cb)

        _visit(source_bidx)

    # Per-cell gating fill time, linear from entry to exit along the local flow.
    T = np.full(shape, np.inf, dtype=np.float64)
    for b in range(n_bodies):
        if not np.isfinite(t_enter[b]) or not np.isfinite(t_exit[b]):
            continue
        mask = (body_index == b) & cavity
        if not mask.any():
            continue
        p0 = entry_point.get(b, exit_point[b])
        p1 = exit_point.get(b, exit_point[b])
        d = p1 - p0
        d2 = float(np.dot(d, d))
        idx_b = np.argwhere(mask)
        centers = origin_mm + (idx_b + 0.5) * dx_mm
        if d2 < 1e-6:
            T[mask] = t_exit[b] if b != source_bidx else t_enter[b]
            continue
        coord = ((centers - p0) @ d) / d2
        coord = np.clip(coord, 0.0, 1.0)
        T[mask] = t_enter[b] + coord * (t_exit[b] - t_enter[b])

    return T, t_enter, t_exit


def _part_fill_fast_marching(
    grid: np.ndarray,
    part_mask: np.ndarray,
    origin_mm: np.ndarray,
    dx_mm: float,
    g: np.ndarray,
    T_gating: np.ndarray,
    ingate_branches: List[Dict],
    fill_time_s: float,
    gamma: float = 0.8,
) -> np.ndarray:
    """Gravity-aware 3-D fast-marching fill time inside the part.

    Each ingate is a source seeded with its priming time from the gating graph.
    The local front speed inside the part is ``Q_total / A(s)`` where ``A(s)`` is
    the cross-sectional area of the cavity perpendicular to gravity at projection
    ``s``.  Edge weights are anisotropic: downward edges are faster, upward edges
    are slower, so the front naturally falls, spreads and rises like a real free
    surface.  The final field is linearly scaled so the last cell reaches exactly
    ``fill_time_s``.
    """
    shape = grid.shape
    cavity_all = (grid > 0) & (grid != int(BodyType.CORE))
    part = np.asarray(part_mask, dtype=bool) & cavity_all
    idx = np.argwhere(part)
    if idx.size == 0:
        return np.full(shape, np.inf, dtype=np.float64)

    node_id = np.full(shape, -1, dtype=np.int64)
    node_id[idx[:, 0], idx[:, 1], idx[:, 2]] = np.arange(idx.shape[0])
    N = idx.shape[0]

    g_u = np.asarray(g, dtype=np.float64)
    g_norm = np.linalg.norm(g_u)
    if g_norm > 1e-12:
        g_u = g_u / g_norm
    else:
        g_u = np.array([0.0, 0.0, -1.0])
    up = -g_u

    centers = origin_mm + (idx + 0.5) * dx_mm
    s = centers @ up
    bin_idx = np.round(s / max(dx_mm, 1e-9)).astype(np.int64)
    unique_bins, inverse, counts = np.unique(bin_idx, return_inverse=True, return_counts=True)
    dx_m = dx_mm / 1000.0
    area_per_bin_m2 = counts.astype(np.float64) * (dx_m ** 2)

    # Gate area from ingate data; use it as a floor so the front speed never
    # exceeds the gate velocity in a single-voxel slice.
    q_total = sum(max(float(b.get("q", 0.0)), 1e-18) for b in ingate_branches)
    v_gate_max = max(max(float(b.get("v", 1.5)), 0.1) for b in ingate_branches)
    A_gate_m2 = q_total / v_gate_max
    base_v = q_total / np.maximum(area_per_bin_m2, A_gate_m2)
    base_v_per_cell = base_v[inverse]
    # Floor avoids huge edge weights in very wide sections; ceiling is the gate
    # velocity (the front cannot outrun the metal entering the cavity).
    base_v_per_cell = np.clip(base_v_per_cell, 1e-3, v_gate_max)

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    weights: List[np.ndarray] = []

    # x-faces
    valid_x = (node_id[:-1, :, :] >= 0) & (node_id[1:, :, :] >= 0)
    r = node_id[:-1, :, :][valid_x]
    c = node_id[1:, :, :][valid_x]
    bv = 0.5 * (base_v_per_cell[r] + base_v_per_cell[c])
    # forward: +x, backward: -x
    gx = float(g_u[0])
    w_forward = dx_m / (bv * np.maximum(0.1, 1.0 + gamma * gx))
    w_back = dx_m / (bv * np.maximum(0.1, 1.0 - gamma * gx))
    rows.extend([r, c])
    cols.extend([c, r])
    weights.extend([w_forward, w_back])

    # y-faces
    valid_y = (node_id[:, :-1, :] >= 0) & (node_id[:, 1:, :] >= 0)
    r = node_id[:, :-1, :][valid_y]
    c = node_id[:, 1:, :][valid_y]
    bv = 0.5 * (base_v_per_cell[r] + base_v_per_cell[c])
    gy = float(g_u[1])
    w_forward = dx_m / (bv * np.maximum(0.1, 1.0 + gamma * gy))
    w_back = dx_m / (bv * np.maximum(0.1, 1.0 - gamma * gy))
    rows.extend([r, c])
    cols.extend([c, r])
    weights.extend([w_forward, w_back])

    # z-faces
    valid_z = (node_id[:, :, :-1] >= 0) & (node_id[:, :, 1:] >= 0)
    r = node_id[:, :, :-1][valid_z]
    c = node_id[:, :, 1:][valid_z]
    bv = 0.5 * (base_v_per_cell[r] + base_v_per_cell[c])
    gz = float(g_u[2])
    w_forward = dx_m / (bv * np.maximum(0.1, 1.0 + gamma * gz))
    w_back = dx_m / (bv * np.maximum(0.1, 1.0 - gamma * gz))
    rows.extend([r, c])
    cols.extend([c, r])
    weights.extend([w_forward, w_back])

    # Virtual source connected to part cells that touch a primed gating cell.
    # The seed time is the earliest gating arrival in the 26-neighbourhood.
    gating_seed = cavity_all & ~part & np.isfinite(T_gating) & (T_gating > -1e-9)
    if gating_seed.any():
        seed_w = np.full(shape, np.inf, dtype=np.float64)
        seed_w[gating_seed] = T_gating[gating_seed]
        min_seed_w = ndimage.minimum_filter(seed_w, size=3, mode="constant", cval=np.inf)
        seed_mask = part & np.isfinite(min_seed_w)
        seed_ids = node_id[seed_mask]
        seed_weights = min_seed_w[seed_mask]
    else:
        seed_ids = np.array([], dtype=np.int64)
        seed_weights = np.array([], dtype=np.float64)

    if seed_ids.size == 0:
        # No gating seed: use the highest part cells along -gravity.
        part_idx = np.argwhere(part)
        if part_idx.size == 0:
            return np.full(shape, np.inf, dtype=np.float64)
        centers_part = origin_mm + (part_idx + 0.5) * dx_mm
        proj = centers_part @ (-g_u)
        top_n = min(10, part_idx.shape[0])
        top = part_idx[np.argpartition(-proj, top_n - 1)[:top_n]]
        seed_ids = node_id[top[:, 0], top[:, 1], top[:, 2]]
        seed_weights = np.zeros(top_n, dtype=np.float64)

    virtual = N
    rows.append(np.full(seed_ids.size, virtual, dtype=np.int64))
    cols.append(seed_ids)
    weights.append(seed_weights)

    rows_a = np.concatenate(rows)
    cols_a = np.concatenate(cols)
    weights_a = np.concatenate(weights)
    graph = csr_matrix(
        (weights_a, (rows_a, cols_a)), shape=(virtual + 1, virtual + 1)
    )
    dist = csgraph.dijkstra(graph, directed=True, indices=virtual, return_predecessors=False)
    if dist.ndim == 1:
        dist = dist.reshape(1, -1)
    part_times = dist[0, :N]

    t_part_start = max(
        (float(b.get("t_open", 0.0)) for b in ingate_branches),
        default=0.0,
    )

    fill_part = np.full(shape, np.inf, dtype=np.float64)
    fill_part[part] = part_times
    fill_part[~part] = 0.0
    fill_part[grid == int(BodyType.CORE)] = np.inf

    if part.any():
        pm = part & np.isfinite(fill_part)
        if pm.any():
            pmin = float(fill_part[pm].min())
            pmax = float(fill_part[pm].max())
            if pmax > pmin + 1e-12 and fill_time_s > t_part_start:
                scale = (fill_time_s - t_part_start) / (pmax - pmin)
                fill_part[pm] = t_part_start + (fill_part[pm] - pmin) * scale
            else:
                fill_part[pm] = fill_time_s

    return fill_part


def _part_fill_volume_level(
    grid: np.ndarray,
    part_mask: np.ndarray,
    origin_mm: np.ndarray,
    dx_mm: float,
    g: np.ndarray,
    T_gating: np.ndarray,
    ingate_branches: List[Dict],
    fill_time_s: float,
) -> np.ndarray:
    """Exact volume-of-cavity / rising-free-surface fill time.

    The part cavity is split into connected components.  For each component the
    cells are projected onto the anti-gravity axis, the cross-sectional area
    ``A(s)`` of each layer is measured, and the layer is filled according to
    ``dt = A(s) ds / Q``.  This is the exact solution for an incompressible
    liquid rising under a prescribed inlet flow, and it naturally satisfies
    continuity in narrow/wide sections without artificial velocity caps.

    Multiple ingates are assigned to the component they feed; their opening
    times come from the gating graph and their flows add to ``Q_comp``.
    """
    shape = grid.shape
    g_u = _gravity_unit(g)
    up = -g_u
    dx_m = dx_mm / 1000.0
    area_per_cell = dx_m * dx_m
    cavity_all = (grid > 0) & (grid != int(BodyType.CORE))
    part = np.asarray(part_mask, dtype=bool) & cavity_all
    fill_part = np.full(shape, np.inf, dtype=np.float64)
    if not part.any() or not ingate_branches:
        return fill_part

    labeled, n_comp = ndimage.label(part, structure=np.ones((3, 3, 3), dtype=int))
    if n_comp == 0:
        return fill_part

    branch_q = np.array(
        [float(b.get("q", 0.0)) for b in ingate_branches], dtype=np.float64
    )
    branch_open = np.array(
        [float(b.get("t_open", 0.0)) for b in ingate_branches], dtype=np.float64
    )
    branch_points = np.array(
        [np.asarray(b.get("point", [0.0, 0.0, 0.0]), dtype=np.float64)
         for b in ingate_branches],
        dtype=np.float64,
    )
    total_q = float(max(branch_q.sum(), 1e-18))

    # Component centroids in mm (physical coordinates)
    comp_centroids_mm = np.empty((n_comp, 3), dtype=np.float64)
    for c in range(1, n_comp + 1):
        idx_c = np.argwhere(labeled == c)
        centers_c = origin_mm + (idx_c + 0.5) * dx_mm
        comp_centroids_mm[c - 1] = centers_c.mean(axis=0)

    # Assign each branch to the nearest component.
    diff = branch_points[:, None, :] - comp_centroids_mm[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)
    if dist2.size:
        branch_comp = np.argmin(dist2, axis=1)
    else:
        branch_comp = np.zeros(len(ingate_branches), dtype=np.int64)

    for c in range(1, n_comp + 1):
        mask = labeled == c
        idx_c = np.argwhere(mask)
        centers_c = origin_mm + (idx_c + 0.5) * dx_mm
        s = centers_c @ up

        # Build the step-wise inlet schedule for this component.
        assigned = branch_comp == (c - 1)
        if assigned.any():
            events = sorted(
                zip(
                    branch_open[assigned].tolist(),
                    branch_q[assigned].tolist(),
                )
            )
        else:
            events = [(0.0, total_q)]

        # Bin cells by gravity projection; each bin is one layer of thickness dx.
        bin_idx = np.floor(s / max(dx_mm, 1e-9)).astype(np.int64)
        bin_min = int(bin_idx.min())
        rel = bin_idx - bin_min
        counts = np.bincount(rel)
        A_layers = counts.astype(np.float64) * area_per_cell
        dV = A_layers * dx_m

        # Cumulative volume up to the centre of each layer.
        cumV = np.cumsum(dV) - 0.5 * dV
        cumV = np.maximum(cumV, 0.0)

        # Convert cumulative volume to time for a step-wise inlet flow.
        t_layer = _volume_to_time(cumV, events)

        rel_all = rel
        fill_part[mask] = t_layer[rel_all]

    return fill_part


def _volume_to_time(
    V: np.ndarray,
    events: List[Tuple[float, float]],
) -> np.ndarray:
    """Solve ``V = ∫ Q(t) dt`` for a piecewise-constant inlet schedule.

    ``events`` is a sorted list of ``(t_open, q)`` pairs.  At each ``t_open`` the
    flow ``q`` is added to the active flow.  ``V`` is the cumulative volume to be
    filled; the returned array is the time at which each volume has been added.
    """
    if not events:
        return np.full_like(V, np.inf)
    # Remove zero-flow events that have duplicate times; keep the earliest t_open.
    events = sorted(events)
    t_starts = np.array([e[0] for e in events], dtype=np.float64)
    q_vals = np.array([e[1] for e in events], dtype=np.float64)
    # Cumulative active flow during each interval.
    Q_cum = np.cumsum(q_vals)
    # End time of each interval (start of next event), inf for the last.
    t_ends = np.empty_like(t_starts)
    t_ends[:-1] = t_starts[1:]
    t_ends[-1] = np.inf
    # Volume capacity of each finite interval.
    dt = t_ends[:-1] - t_starts[:-1]
    capacity = Q_cum[:-1] * dt
    cum_capacity = np.cumsum(np.concatenate(([0.0], capacity)))
    # Per layer, find which interval contains the target volume.
    out = np.empty_like(V, dtype=np.float64)
    out[:] = np.inf
    for i in range(len(events)):
        if i < len(events) - 1:
            lo = cum_capacity[i]
            hi = cum_capacity[i + 1]
            in_interval = (V >= lo - 1e-15) & (V < hi)
            Q = max(Q_cum[i], 1e-18)
            out[in_interval] = t_starts[i] + (V[in_interval] - lo) / Q
        else:
            lo = cum_capacity[i]
            Q = max(Q_cum[i], 1e-18)
            in_last = V >= lo - 1e-15
            out[in_last] = t_starts[i] + (V[in_last] - lo) / Q
    return out


def _compute_fill_time_volume_layer(
    grid: np.ndarray,
    origin_mm: np.ndarray,
    dx_mm: float,
    g: np.ndarray,
    gating_nodes: List[GatingNode],
    bodies: List[Body],
    body_index: Optional[np.ndarray],
    fill_time_s: float,
    source_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Volume-aware, gravity-aligned fill-time estimator.

    The gating system is treated as a network of vessels that must each fill
    before the downstream vessel can start.  Once the ingates are primed, the
    casting cavity fills as a single rising metal level whose velocity depends
    on the local cavity cross-section ``v(z) = Q_active / A(z)``.  This gives
    a continuous, physically plausible front instead of separate expanding
    spheres from each gate.
    """
    shape = grid.shape
    fill = np.full(shape, np.inf, dtype=np.float64)
    cavity = (grid > 0) & (grid != int(BodyType.CORE))
    if not cavity.any() or fill_time_s <= 1e-12:
        fill[~cavity] = 0.0
        fill[grid == int(BodyType.CORE)] = np.inf
        return fill

    g_u = np.asarray(g, dtype=np.float64)
    g_norm = float(np.linalg.norm(g_u))
    if g_norm > 1e-12:
        g_u = g_u / g_norm
    else:
        g_u = np.array([0.0, 0.0, -1.0])
    up = -g_u

    # Fallback: gravity rising front if no gating graph.
    if not gating_nodes or body_index is None or body_index.shape != shape or not bodies:
        idx = np.argwhere(cavity)
        if idx.size:
            centers = origin_mm + (idx + 0.5) * dx_mm
            z = centers @ up
            z_min = float(z.min())
            z_max = float(z.max())
            rng = max(z_max - z_min, 1e-6)
            fill[cavity] = ((z - z_min) / rng) * fill_time_s
        fill[~cavity] = 0.0
        fill[grid == int(BodyType.CORE)] = np.inf
        return fill

    n_bodies = len(bodies)
    name_to_bidx = {b.name: i for i, b in enumerate(bodies)}

    # For each original body, take the most common reclassified grid type.
    grid_type_for_bidx = np.zeros(n_bodies, dtype=np.int64)
    for i in range(n_bodies):
        sub = grid[body_index == i]
        if sub.size:
            pos = sub[sub > 0]
            if pos.size:
                grid_type_for_bidx[i] = int(np.bincount(pos).argmax())

    # Parse gating graph and flows.
    source_bidx: Optional[int] = None
    source_q = 0.0
    source_v = 0.0
    source_centroid = origin_mm + 0.5 * np.asarray(shape, dtype=np.float64) * dx_mm
    children: Dict[int, List[Dict]] = {}
    entry_point: Dict[int, np.ndarray] = {}
    q_in: Dict[int, float] = {}
    ingate_branches: List[Dict] = []

    for node in gating_nodes:
        if "→" not in node.name or "→" not in node.body_type:
            continue
        up_name, down_name = [s.strip() for s in node.name.split("→")]
        up_type, down_type = [s.strip() for s in node.body_type.split("→")]
        point = np.asarray(node.centroid_mm, dtype=np.float64)
        if up_name == "Kaynak" or "SOURCE" in up_type:
            source_bidx = name_to_bidx.get(down_name)
            source_q = float(node.flow_rate_m3_s)
            source_v = float(
                node.max_velocity_m_s
                if node.max_velocity_m_s > 1e-12
                else node.velocity_m_s
            )
            source_centroid = point
            continue
        up_bidx = name_to_bidx.get(up_name)
        if up_bidx is None:
            continue
        is_part = down_name == "Parça" or down_type == "PART"
        down_bidx = None if is_part else name_to_bidx.get(down_name)
        child = {
            "bidx": down_bidx,
            "point": point,
            "v": float(
                node.max_velocity_m_s
                if node.max_velocity_m_s > 1e-12
                else node.velocity_m_s
            ),
            "q": float(node.flow_rate_m3_s),
        }
        children.setdefault(up_bidx, []).append(child)
        if is_part:
            ingate_branches.append({"parent_bidx": up_bidx, "point": point, "q": child["q"], "v": child["v"]})
        else:
            if down_bidx is not None:
                entry_point[down_bidx] = point
                q_in[down_bidx] = q_in.get(down_bidx, 0.0) + child["q"]

    if source_bidx is None or not (0 <= source_bidx < n_bodies) or not ingate_branches:
        idx = np.argwhere(cavity)
        if idx.size:
            centers = origin_mm + (idx + 0.5) * dx_mm
            z = centers @ up
            z_min = float(z.min())
            z_max = float(z.max())
            rng = max(z_max - z_min, 1e-6)
            fill[cavity] = ((z - z_min) / rng) * fill_time_s
        fill[~cavity] = 0.0
        fill[grid == int(BodyType.CORE)] = np.inf
        return fill

    q_in[source_bidx] = source_q
    entry_point[source_bidx] = source_centroid

    # Helper to make a body mask consistent with its grid type.
    def _body_mask(bidx: int) -> np.ndarray:
        mask = body_index == bidx
        if 0 <= bidx < n_bodies:
            mask &= grid == grid_type_for_bidx[bidx]
        return mask

    if source_mask is None or not source_mask.any():
        source_mask, _ = _select_inlet_cells(grid, cavity, g, "SPRUE_THROAT")
    if source_mask is None or not source_mask.any():
        # Fall back to the upstream-most cells of the source body.
        if source_bidx is not None and 0 <= source_bidx < n_bodies:
            sm = _body_mask(source_bidx)
            if sm.any():
                proj = _projection_along(grid.shape, origin_mm, dx_mm, g)
                best = tuple(np.argwhere(sm)[np.argmax(proj[sm])])
                source_mask = np.zeros_like(sm)
                source_mask[best] = True

    # Volume/continuity arrival time through the gating system.
    T_gating, t_enter_arr, t_exit_arr = _gating_volume_time(
        grid,
        body_index,
        bodies,
        origin_mm,
        dx_mm,
        source_bidx,
        source_q,
        source_centroid,
        children,
        q_in,
        entry_point,
    )

    gating_types = {
        int(BodyType.SPRUE),
        int(BodyType.SPRUE_THROAT),
        int(BodyType.POURING_BASIN),
        int(BodyType.RUNNER),
        int(BodyType.DISTRIBUTOR),
        int(BodyType.INGATE),
    }
    gating_all = np.isin(grid, list(gating_types)) & cavity
    finite_gating = gating_all & np.isfinite(T_gating)
    fill[finite_gating] = T_gating[finite_gating]

    # Keep the source body fully visible at t=0.
    if source_bidx is not None and 0 <= source_bidx < n_bodies:
        source_body_mask = _body_mask(source_bidx)
        if source_body_mask.any():
            fill[source_body_mask] = 0.0

    # Entry time of each gating body (used for ingate priming).
    t_entry: Dict[int, float] = {source_bidx: 0.0} if source_bidx is not None else {}
    for bidx in range(n_bodies):
        if bidx == source_bidx:
            continue
        if np.isfinite(t_enter_arr[bidx]):
            mask = _body_mask(bidx)
            if mask.any():
                t_entry[bidx] = float(t_enter_arr[bidx])

    # Compute the time each ingate is fully primed (body exit time from the
    # volume/continuity tree).  Sibling ingates fed by the same manifold will
    # therefore start together.
    for branch in ingate_branches:
        parent_bidx = branch.get("parent_bidx")
        if parent_bidx is not None and 0 <= parent_bidx < n_bodies:
            if np.isfinite(t_exit_arr[parent_bidx]):
                branch["t_open"] = float(t_exit_arr[parent_bidx])
            else:
                branch["t_open"] = t_entry.get(parent_bidx, 0.0)
        else:
            branch["t_open"] = 0.0

    # Part: gravity-aware 3-D fast-marching front from all primed ingates.
    if ingate_branches:
        gating_bidx_arr = np.fromiter(t_entry.keys(), dtype=np.int64)
        on_gating = np.isin(body_index, gating_bidx_arr)
        part_mask = (grid == int(BodyType.PART)) & ~on_gating & cavity
        if not part_mask.any():
            part_mask = (grid == int(BodyType.PART)) & cavity

        if part_mask.any():
            fill_part = _part_fill_volume_level(
                grid,
                part_mask,
                origin_mm,
                dx_mm,
                g,
                T_gating,
                ingate_branches,
                fill_time_s,
            )
            fill[part_mask] = fill_part[part_mask]

    # Risers/feeders: fill from the nearest already-filled non-riser metal.
    riser_bidx = [
        i
        for i in range(n_bodies)
        if grid_type_for_bidx[i] == int(BodyType.RISER) and i not in t_entry
    ]
    if riser_bidx:
        riser_mask = np.isin(
            body_index, np.fromiter(riser_bidx, dtype=np.int64, count=len(riser_bidx))
        ) & cavity
        if riser_mask.any():
            non_riser_metal = cavity & ~riser_mask & np.isfinite(fill)
            if non_riser_metal.any():
                dist, indices = ndimage.distance_transform_edt(
                    ~non_riser_metal, return_indices=True, return_distances=True
                )
                nearest_fill = fill[indices[0], indices[1], indices[2]]
                travel_m = dist[riser_mask] * dx_mm / 1000.0
                fill[riser_mask] = (
                    nearest_fill[riser_mask]
                    + travel_m / max(source_v, 1e-6)
                )
            else:
                fill[riser_mask] = fill_time_s

    fill[~cavity] = 0.0
    fill[grid == int(BodyType.CORE)] = np.inf
    inf_metal = np.isinf(fill) & cavity
    if inf_metal.any():
        fill[inf_metal] = fill_time_s

    if cavity.any():
        fmin = float(fill[cavity].min())
        fmax = float(fill[cavity].max())
        if (fmax - fmin) < 1e-3 * fill_time_s or fmin > 0.9 * fill_time_s:
            idx = np.argwhere(cavity)
            centers = origin_mm + (idx + 0.5) * dx_mm
            z = centers @ up
            z_min = float(z.min())
            z_max = float(z.max())
            rng = max(z_max - z_min, 1e-6)
            fill[cavity] = ((z - z_min) / rng) * fill_time_s

    return fill


def _section_downstream_flux(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    section_mask: np.ndarray,
    cavity: np.ndarray,
    g: np.ndarray,
    dx_m: float,
) -> Tuple[float, float]:
    """Compute net flux (m³/s) and face area (m²) from section to the rest of the cavity.

    Sums all six face orientations so curved or side-fed gating sections are handled.
    """
    section = section_mask & cavity
    if not section.any():
        return 0.0, 0.0
    rest = cavity & ~section

    flux = 0.0
    face_count = 0

    # x faces: u[1:-1] is flux from cell i-1 to i across face i
    left_sec = section[:-1] & rest[1:]
    right_sec = section[1:] & rest[:-1]
    ux = u[1:-1]
    flux += float(ux[left_sec].sum())
    flux -= float(ux[right_sec].sum())
    face_count += int(left_sec.sum() + right_sec.sum())

    # y faces: v[:, 1:-1]
    down_sec = section[:, :-1] & rest[:, 1:]
    up_sec = section[:, 1:] & rest[:, :-1]
    vy = v[:, 1:-1]
    flux += float(vy[down_sec].sum())
    flux -= float(vy[up_sec].sum())
    face_count += int(down_sec.sum() + up_sec.sum())

    # z faces: w[:, :, 1:-1]
    back_sec = section[:, :, :-1] & rest[:, :, 1:]
    front_sec = section[:, :, 1:] & rest[:, :, :-1]
    wz = w[:, :, 1:-1]
    flux += float(wz[back_sec].sum())
    flux -= float(wz[front_sec].sum())
    face_count += int(back_sec.sum() + front_sec.sum())

    area = face_count * dx_m * dx_m
    flux = flux * dx_m * dx_m
    return flux, area


def _node_velocities(
    grid: np.ndarray,
    cavity: np.ndarray,
    g: np.ndarray,
    dx_m: float,
    Q_m3_s: float,
    fine_grid: Optional[np.ndarray] = None,
    fine_cavity: Optional[np.ndarray] = None,
    fine_dx_m: Optional[float] = None,
    section_areas_m2: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Average section velocity = Q / representative cross-sectional area (continuity).

    If `section_areas_m2` is supplied it overrides the voxel face area for the
    corresponding section; otherwise the original (fine) voxel grid is used.
    """
    g_grid = fine_grid if fine_grid is not None else grid
    g_cavity = fine_cavity if fine_cavity is not None else cavity
    g_dx = fine_dx_m if fine_dx_m is not None else dx_m
    areas = section_areas_m2 or {}
    section_keys = [
        "SPRUE_THROAT",
        "SPRUE_BASE",
        "RUNNER",
        "DISTRIBUTOR",
        "CURUFLUK",
        "INGATE",
        "FILTER",
        "RISER",
    ]
    out: Dict[str, float] = {}
    for name in section_keys:
        measured = areas.get(name)
        if measured is not None and measured > 1e-18:
            out[name] = float(Q_m3_s / measured)
            continue
        face, _ = _section_face_cells(g_grid, g_cavity, name, g, allow_fallback=False)
        area_m2 = float(face.sum()) * g_dx * g_dx
        if area_m2 > 1e-18:
            out[name] = float(Q_m3_s / area_m2)
        else:
            out[name] = 0.0
    return out


def _ingate_contact_velocity(
    grid: np.ndarray,
    cavity: np.ndarray,
    dx_m: float,
    g: np.ndarray,
    Q_m3_s: float,
    fine_grid: Optional[np.ndarray] = None,
    fine_cavity: Optional[np.ndarray] = None,
    fine_dx_m: Optional[float] = None,
    section_areas_m2: Optional[Dict[str, float]] = None,
) -> float:
    """Velocity normal to the ingate-to-part interface (m/s).

    Computed from continuity Q / gate contact area.
    """
    areas = section_areas_m2 or {}
    measured = areas.get("INGATE")
    if measured is not None and measured > 1e-18:
        return float(Q_m3_s / measured)
    g_grid = fine_grid if fine_grid is not None else grid
    g_cavity = fine_cavity if fine_cavity is not None else cavity
    g_dx = fine_dx_m if fine_dx_m is not None else dx_m
    face, _ = _section_face_cells(g_grid, g_cavity, "INGATE", g, allow_fallback=False)
    area_m2 = float(face.sum()) * g_dx * g_dx
    return float(Q_m3_s / area_m2) if area_m2 > 1e-18 else 0.0


def _component_section_areas_m2(
    cells: np.ndarray,
    dx_mm: float,
    dx_m: float,
    axis: np.ndarray,
) -> Tuple[float, float, float]:
    """Return the inlet, outlet and throat areas of a voxel component.

    The component is sliced perpendicular to its principal (flow) axis.
    The first and last slices are the inlet/outlet and the minimum non-zero
    slice is the throat.  This is independent of gravity, so horizontal runners
    and angled sprues get the correct cross-sectional area.
    """
    if cells.size == 0:
        return (0.0, 0.0, 0.0)
    centers = (cells.astype(np.float64) + 0.5) * dx_mm
    a = np.asarray(axis, dtype=np.float64)
    if np.linalg.norm(a) < 1e-18:
        a = np.array([0.0, 0.0, -1.0])
    a = a / np.linalg.norm(a)
    proj = np.dot(centers, a)
    if proj.size == 0:
        return (0.0, 0.0, 0.0)
    bins = np.floor(proj / dx_mm).astype(np.int64)
    bins = bins - int(bins.min())
    counts = np.bincount(bins)
    if counts.size == 0:
        return (0.0, 0.0, 0.0)
    nonzero = counts[counts > 0]
    throat_count = float(nonzero.min()) if nonzero.size > 0 else 0.0
    inlet_count = float(counts[0])
    outlet_count = float(counts[-1])
    return (
        inlet_count * dx_m * dx_m,
        outlet_count * dx_m * dx_m,
        throat_count * dx_m * dx_m,
    )


def _aggregate_section_velocities(
    gating_nodes: List[GatingNode],
    Q_user: float,
    inlet_area_m2: float,
    gate_results: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], float]:
    """Derive section-aggregate velocities and the ingate contact velocity.

    The aggregate values are taken directly from the contact-node values so the
    report and the 3-D labels always agree.  When a local 3-D gate solve is
    available, the section velocity is taken from the solver's ``max_velocity``
    for the corresponding body instead of the coarse Q/A interface estimate.
    The source (SPRUE_THROAT) velocity falls back to Q / inlet_area when no
    upstream contact exists.
    """
    section_velocities: Dict[str, List[float]] = {
        "SPRUE_THROAT": [],
        "SPRUE_BASE": [],
        "RUNNER": [],
        "DISTRIBUTOR": [],
        "CURUFLUK": [],
        "FILTER": [],
        "INGATE": [],
        "RISER": [],
    }

    # Body-type name -> report/label section key.
    _SECTION_KEY: Dict[str, Optional[str]] = {
        "SPRUE": "SPRUE_BASE",
        "SPRUE_THROAT": "SPRUE_THROAT",
        "RUNNER": "RUNNER",
        "DISTRIBUTOR": "DISTRIBUTOR",
        "CURUFLUK": "CURUFLUK",
        "FILTER": "FILTER",
        "INGATE": "INGATE",
        "RISER": "RISER",
        "PART": None,
        "SOURCE": None,
    }

    def _section_for(up: str, down: str) -> Optional[str]:
        # Prefer the upstream section; e.g. SPRUE_THROAT→SPRUE belongs to
        # SPRUE_THROAT and SPRUE→INGATE belongs to SPRUE_BASE.  Downstream
        # is only used as a fallback (e.g. SOURCE→SPRUE_THROAT).
        for body in (up, down):
            sec = _SECTION_KEY.get(body, body)
            if sec in section_velocities:
                return sec
        return None

    # The 3-D gate solve refines pressure drop and local peaks; the section
    # velocity reported at each node is already the mean contact velocity
    # (Q / A) for that branch, set in solve_gate_flows.

    for n in gating_nodes:
        parts = n.body_type.split("→")
        if len(parts) != 2:
            continue
        up, down = parts
        sec = _section_for(up, down)
        if sec is None:
            continue
        v = n.max_velocity_m_s if n.max_velocity_m_s > 1e-12 else n.velocity_m_s
        section_velocities[sec].append(v)

    node_v: Dict[str, float] = {}
    for key, vals in section_velocities.items():
        if vals:
            node_v[key] = float(np.mean(vals))

    # No upstream contact for the source sprue: use Q / inlet area.
    if "SPRUE_THROAT" not in node_v and inlet_area_m2 > 1e-18:
        node_v["SPRUE_THROAT"] = float(Q_user / inlet_area_m2)

    v_ingate_contact = float(node_v.get("INGATE", 0.0))

    return node_v, v_ingate_contact


def _reclassify_comp_meta_from_graph(
    comp_meta: Dict[int, Tuple[BodyType, str]],
    comp_centroids: Dict[int, np.ndarray],
    contacts: List[Dict],
    part_id: int,
    g: np.ndarray,
    source_section: str,
    verbose: bool = False,
) -> Dict[int, Tuple[BodyType, str]]:
    """Correct voxel-merged/misclassified body types using the CAD mesh contact graph.

    The contact graph is already directed (up_id -> down_id) by surface normals.
    This function finds the topmost source that can reach the part and re-types
    every reachable component according to its position in the flow path.
    Side branches (e.g. risers) that cannot reach the part are left as RISER
    or EMPTY so they do not steal main flow.
    """
    g_u = np.asarray(g, dtype=np.float64)
    norm = float(np.linalg.norm(g_u))
    if norm > 1e-12:
        g_u = g_u / norm
    else:
        g_u = np.array([0.0, 0.0, -1.0])

    def _rank(cid: int) -> float:
        return float(-np.dot(comp_centroids.get(cid, np.zeros(3)), g_u))

    # Use the undirected contact graph for source selection; the local face-normal
    # orientation in _build_mesh_contacts may flip side/T-junction contacts.
    can_reach_part: Dict[int, bool] = _part_reachable_undirected(contacts, part_id)

    # Distance from the part on the undirected graph (the source should be the
    # farthest reachable component that is not itself a gate).
    from collections import defaultdict, deque

    undirected: Dict[int, List[int]] = defaultdict(list)
    to_part: Dict[int, bool] = {cid: False for cid in comp_meta}
    degree: Dict[int, int] = {cid: 0 for cid in comp_meta}
    for c in contacts:
        u = int(c["id1"])
        v = int(c["id2"])
        undirected[u].append(v)
        undirected[v].append(u)
        degree[u] += 1
        degree[v] += 1
        if u == part_id:
            to_part[v] = True
        if v == part_id:
            to_part[u] = True

    dist_from_part: Dict[int, int] = {part_id: 0}
    q: deque[int] = deque([part_id])
    while q:
        cur = q.popleft()
        for nb in undirected.get(cur, []):
            if nb not in dist_from_part:
                dist_from_part[nb] = dist_from_part[cur] + 1
                q.append(nb)

    non_part = [cid for cid in comp_meta if cid != part_id]
    candidates = [
        cid
        for cid in non_part
        if can_reach_part.get(cid, False)
        and not to_part.get(cid, False)
        and degree.get(cid, 0) > 0
    ]
    if not candidates:
        candidates = [
            cid
            for cid in non_part
            if can_reach_part.get(cid, False) and degree.get(cid, 0) > 0
        ]
    if not candidates:
        return comp_meta

    # Farthest from the part, then highest upstream (most negative dot with g_u).
    source = max(candidates, key=lambda cid: (dist_from_part.get(cid, 0), _rank(cid)))

    # Orient every contact away from the chosen source toward the part.
    _reorient_contacts_to_source(
        contacts, source, comp_meta, comp_centroids, g_u, part_id, verbose=verbose
    )

    # Build the directed graph from the now-correctly oriented contacts and
    # recompute reachability/classification as before.
    adj: Dict[int, List[int]] = {cid: [] for cid in comp_meta}
    incoming: Dict[int, List[int]] = {cid: [] for cid in comp_meta}
    to_part = {cid: False for cid in comp_meta}
    for c in contacts:
        up = int(c["up_id"])
        down = int(c["down_id"])
        if down == part_id:
            to_part[up] = True
            incoming[part_id].append(up)
            continue
        adj[up].append(down)
        incoming[down].append(up)

    can_reach_part = {part_id: True}
    queue = [part_id]
    while queue:
        cur = queue.pop(0)
        for prev in incoming.get(cur, []):
            if prev not in can_reach_part:
                can_reach_part[prev] = True
                queue.append(prev)

    indeg = {cid: 0 for cid in comp_meta}
    outdeg = {cid: 0 for cid in comp_meta}
    for c in contacts:
        up = int(c["up_id"])
        down = int(c["down_id"])
        if down == part_id:
            if can_reach_part.get(up, False):
                outdeg[up] += 1
            continue
        if can_reach_part.get(up, False) and can_reach_part.get(down, False):
            outdeg[up] += 1
            indeg[down] += 1

    # BFS from source along the directed, part-reachable graph.
    parent: Dict[int, int] = {source: -1}
    children: Dict[int, List[int]] = {source: []}
    visited = {source}
    queue = [source]
    while queue:
        cur = queue.pop(0)
        for nxt in adj.get(cur, []):
            if nxt in visited or not can_reach_part.get(nxt, False):
                continue
            visited.add(nxt)
            parent[nxt] = cur
            children.setdefault(cur, []).append(nxt)
            children.setdefault(nxt, [])
            queue.append(nxt)

    new_meta = dict(comp_meta)
    section_up = source_section.upper()

    for cid in list(comp_meta.keys()):
        if cid == part_id:
            continue
        bt, name = new_meta[cid]
        is_source = cid == source
        kids = children.get(cid, [])
        has_part = to_part.get(cid, False)

        if is_source:
            if "THROAT" in section_up or "BOĞAZ" in section_up:
                new_meta[cid] = (BodyType.SPRUE_THROAT, name)
            elif "BASIN" in section_up or "HAVUZ" in section_up:
                new_meta[cid] = (BodyType.POURING_BASIN, name)
            else:
                new_meta[cid] = (BodyType.SPRUE, name)
            continue

        if cid not in visited:
            # Not on the source-to-part path: keep a riser, reset everything else.
            if bt != BodyType.RISER:
                new_meta[cid] = (BodyType.EMPTY, name)
            continue

        if has_part:
            new_meta[cid] = (BodyType.INGATE, name)
            continue

        if len(kids) >= 2:
            new_meta[cid] = (BodyType.DISTRIBUTOR, name) if parent.get(cid) != source else (BodyType.SPRUE, name)
            continue

        if len(kids) == 1:
            new_meta[cid] = (BodyType.RUNNER, name)
            continue

        # Leaf with no downstream: keep a riser, otherwise empty.
        if bt == BodyType.RISER:
            continue
        new_meta[cid] = (BodyType.EMPTY, name)

    # Direct part-feeders that were not reached from the source are still gates.
    for cid in non_part:
        if cid not in visited and to_part.get(cid, False):
            bt, name = new_meta[cid]
            if bt in {BodyType.PART, BodyType.RISER, BodyType.EMPTY}:
                new_meta[cid] = (BodyType.INGATE, name)

    return new_meta


def _gating_node_velocities(
    grid: np.ndarray,
    origin_mm: np.ndarray,
    dx_mm: float,
    Q_user: float,
    source_area_m2: float,
    source_section_key: str,
    g: np.ndarray,
    bodies: Optional[List[Body]],
    section_areas_m2: Optional[Dict[str, float]] = None,
    velocity_m_s: Optional[np.ndarray] = None,
    dx_m: float = 0.0,
    u_m_s: Optional[np.ndarray] = None,
    v_m_s: Optional[np.ndarray] = None,
    w_m_s: Optional[np.ndarray] = None,
    face_fractions: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> List[GatingNode]:
    """Compute contact-node velocities from the Darcy velocity field if available.

    When staggered face velocities ``u_m_s``, ``v_m_s``, ``w_m_s`` (z, y, x
    components on the MAC grid) are supplied, the flux through each contact is
    integrated directly from the Darcy field using a projected main-flow-axis
    method.  This eliminates the staircasing area inflation and direction noise
    of raw voxel normals on curved/amorphous geometry, and gives the *actual* Q
    per branch so continuity Q = vA is satisfied.

    If no Darcy velocity is supplied, the network falls back to a hydraulic
    Q = vA split proportional to throat area.
    - Q_user is computed once at the source (sprue/pouring-basin) from the user
      velocity and the selected source area.
    - Voxels are used ONLY to discover which gating element touches which.
    - Each element's flow direction is derived from the oriented contact graph, not
      from gravity or from the longest geometric dimension.  This makes the
      throat cross-section of gates and angled runners correct.
    - The throat/contact area is taken from a trimesh plane-section on the CAD
      mesh perpendicular to the element's flow direction, then capped only by the
      user-defined source area and the total INGATE design area.
    - Q is propagated through the directed graph; at any branch the incoming Q is
      split proportionally to the outlet throat areas.
    """
    if bodies is None:
        bodies = []

    gating_types = [
        BodyType.SPRUE,
        BodyType.SPRUE_THROAT,
        BodyType.RUNNER,
        BodyType.DISTRIBUTOR,
        BodyType.CURUFLUK,
        BodyType.INGATE,
        BodyType.FILTER,
        BodyType.POURING_BASIN,
        BodyType.RISER,
    ]

    dx_m = float(dx_mm) / 1000.0
    area_face = dx_m * dx_m

    if face_fractions is None:
        nz, ny, nx = grid.shape
        f_A_z = np.ones((nz + 1, ny, nx), dtype=np.float64)
        f_A_y = np.ones((nz, ny + 1, nx), dtype=np.float64)
        f_A_x = np.ones((nz, ny, nx + 1), dtype=np.float64)
    else:
        f_A_z, f_A_y, f_A_x = face_fractions

    g_u = np.asarray(g, dtype=np.float64)
    if np.linalg.norm(g_u) > 1e-12:
        g_u = g_u / np.linalg.norm(g_u)
    else:
        g_u = np.array([0.0, 0.0, -1.0])

    # Build component IDs and match them to the provided Body meshes.
    part_id = 1
    comp_id = np.zeros(grid.shape, dtype=np.int32)
    comp_id[grid == BodyType.PART] = part_id
    comp_meta: Dict[int, Tuple[BodyType, str]] = {part_id: (BodyType.PART, "Parça")}
    comp_centroids: Dict[int, np.ndarray] = {part_id: np.zeros(3)}
    comp_cells: Dict[int, np.ndarray] = {}
    comp_body: Dict[int, Body] = {}

    bodies_by_type: Dict[BodyType, List[Body]] = {}
    part_body: Optional[Body] = None
    for b in bodies:
        bodies_by_type.setdefault(b.body_type, []).append(b)
        if b.body_type == BodyType.PART and part_body is None:
            part_body = b

    next_id = 2
    for gtype in gating_types:
        mask = grid == gtype
        if not mask.any():
            continue
        labeled, n = ndimage.label(mask)
        type_bodies = bodies_by_type.get(gtype, [])

        centroids_vox: Dict[int, np.ndarray] = {}
        for label_id in range(1, n + 1):
            idx = np.argwhere(labeled == label_id)
            centroids_vox[label_id] = idx.mean(axis=0) if idx.size else np.zeros(3)

        matched: Dict[int, Body] = {}
        used: set = set()
        for b in type_bodies:
            try:
                bc_mm = np.asarray(b.mesh.centroid, dtype=np.float64)
            except Exception:
                continue
            best_label = None
            best_dist = float("inf")
            for label_id, cen in centroids_vox.items():
                if label_id in used:
                    continue
                cen_mm = cen * dx_mm + origin_mm
                dist = float(np.linalg.norm(cen_mm - bc_mm))
                if dist < best_dist:
                    best_dist = dist
                    best_label = label_id
            if best_label is not None:
                matched[best_label] = b
                used.add(best_label)

        for label_id in range(1, n + 1):
            comp_mask = labeled == label_id
            comp_id[comp_mask] = next_id
            name = matched[label_id].name if label_id in matched else f"{gtype.name}_{label_id}"
            comp_meta[next_id] = (gtype, name)
            idx = np.argwhere(comp_mask)
            comp_centroids[next_id] = idx.mean(axis=0) * dx_mm + origin_mm if idx.size else np.zeros(3)
            comp_cells[next_id] = idx
            comp_body[next_id] = matched.get(label_id)
            next_id += 1

    # Make the part mesh reachable as well so gate->part contacts can be
    # sectioned on the gate (upstream) body when needed.
    comp_body[part_id] = part_body

    # Correct part centroid (it is used for the main flow direction of gate→part contacts).
    part_idx = np.argwhere(grid == BodyType.PART)
    if part_idx.size:
        comp_centroids[part_id] = part_idx.mean(axis=0) * dx_mm + origin_mm

    # Map every real Body to a component ID, even if the voxel grid merged or
    # missed it.  This makes the contact graph CAD-driven instead of voxel-driven.
    body_to_cid: Dict[int, int] = {}
    for cid, b in comp_body.items():
        if b is not None:
            body_to_cid[id(b)] = cid

    for b in bodies:
        if b is None or b.mesh is None or len(b.mesh.faces) == 0:
            continue
        if id(b) in body_to_cid:
            continue
        comp_body[next_id] = b
        comp_meta[next_id] = (b.body_type, b.name)
        comp_centroids[next_id] = np.asarray(b.center, dtype=np.float64)
        body_to_cid[id(b)] = next_id
        next_id += 1

    # Recompute the integer key space to include all real bodies.
    max_id = int(max(comp_meta.keys()))
    mult = max_id + 1
    n_keys = mult * mult
    area_b = np.zeros(n_keys, dtype=np.float64)
    cx_b = np.zeros(n_keys, dtype=np.float64)
    cy_b = np.zeros(n_keys, dtype=np.float64)
    cz_b = np.zeros(n_keys, dtype=np.float64)
    nx_b = np.zeros(n_keys, dtype=np.float64)
    ny_b = np.zeros(n_keys, dtype=np.float64)
    nz_b = np.zeros(n_keys, dtype=np.float64)

    # Discover contacts directly from the CAD meshes.  Voxel adjacency is only
    # used for the optional Darcy flux integral below, not for the node graph.
    contacts = _build_mesh_contacts(
        comp_body,
        comp_meta,
        comp_centroids,
        g=g,
        part_id=part_id,
        max_gap_mm=0.5,
        max_query=5000,
        verbose=False,
    )

    if not contacts:
        raise GatingVelocityError(
            "Temas yok: CAD mesh'lerinde birbirine değen döküm elemanı bulunamadı, "
            "düğüm hızları hesaplanamadı."
        )

    # The voxel grid may have merged gates into the part or mislabelled runners as
    # risers.  Re-type components from the CAD mesh contact graph so node
    # velocities and section reports are physically correct.
    source_section = (source_section_key or "SPRUE_THROAT").upper()
    comp_meta = _reclassify_comp_meta_from_graph(
        comp_meta,
        comp_centroids,
        contacts,
        part_id,
        g,
        source_section,
        verbose=False,
    )

    # Reachability is first determined on the undirected contact graph so that a
    # locally-flipped side contact (T-junction) does not hide reachable gates.
    can_reach_part: Dict[int, bool] = _part_reachable_undirected(contacts, part_id)

    # Source selection: use the BodyType selected by the user's velocity_section_key.
    # Restrict to components that can actually reach the part.
    section_to_types = {
        "SPRUE": {BodyType.SPRUE},
        "SPRUE_BASE": {BodyType.SPRUE},
        "SPRUE_THROAT": {BodyType.SPRUE_THROAT, BodyType.SPRUE},
        "POURING_BASIN": {BodyType.POURING_BASIN},
        "RUNNER": {BodyType.RUNNER},
        "DISTRIBUTOR": {BodyType.DISTRIBUTOR},
        "CURUFLUK": {BodyType.CURUFLUK},
        "FILTER": {BodyType.FILTER},
        "INGATE": {BodyType.INGATE},
    }
    allowed_source_types = section_to_types.get(
        source_section,
        {BodyType.POURING_BASIN, BodyType.SPRUE_THROAT, BodyType.SPRUE},
    )
    source_candidates = [
        cid for cid, (bt, _) in comp_meta.items()
        if bt in allowed_source_types and cid != part_id and can_reach_part.get(cid, False)
    ]
    if not source_candidates:
        source_candidates = [
            cid for cid, (bt, _) in comp_meta.items()
            if bt in {BodyType.POURING_BASIN, BodyType.SPRUE_THROAT, BodyType.SPRUE}
            and cid != part_id and can_reach_part.get(cid, False)
        ]

    gating_ids = [cid for cid in comp_meta if cid != part_id]
    if not gating_ids:
        raise GatingVelocityError(
            "Düğüm hızları çözülemedi: döküm sistemi elemanı (sprue/yolluk/meme) bulunamadı. "
            "Hız kesitleri parça/geometri nedeniyle hesaplanamadı."
        )

    def _upstream_rank(cid: int) -> float:
        return float(-np.dot(comp_centroids[cid], g_u))

    if source_candidates:
        source_id = max(source_candidates, key=_upstream_rank)
    else:
        reachable_gating = [cid for cid in gating_ids if can_reach_part.get(cid, False)]
        if not reachable_gating:
            raise GatingVelocityError(
                "Düğüm hızları çözülemedi: hiçbir döküm elemanı parçaya ulaşan yol üzerinde değil. "
                "Hız kesitleri parça/geometri nedeniyle hesaplanamadı."
            )
        source_id = max(reachable_gating, key=_upstream_rank)

    if source_id not in comp_meta or source_id == part_id:
        raise GatingVelocityError(
            "Düğüm hızları çözülemedi: kaynak (sprue/döküm ağzı) seçilemedi. "
            "Hız kesitleri parça/geometri nedeniyle hesaplanamadı."
        )
    if Q_user <= 1e-18:
        raise GatingVelocityError(
            "Düğüm hızları çözülemedi: giriş debisi tanımlanamadı. "
            "Hız kesitleri parça/geometri nedeniyle hesaplanamadı."
        )

    # Recompute reachability on the now directed graph.
    incoming_map: Dict[int, List[int]] = {cid: [] for cid in comp_meta}
    for c in contacts:
        up = int(c["up_id"])
        down = int(c["down_id"])
        if down == part_id:
            incoming_map[part_id].append(up)
        else:
            incoming_map[down].append(up)

    can_reach_part = {part_id: True}
    queue = [part_id]
    while queue:
        cur = queue.pop(0)
        for prev in incoming_map.get(cur, []):
            if prev not in can_reach_part:
                can_reach_part[prev] = True
                queue.append(prev)

    # Prune the directed contact graph to the part-reachable subgraph so side
    # branches (risers, isolated sprues) do not steal main flow.
    outgoing: Dict[int, List[Dict]] = {cid: [] for cid in comp_meta if cid != part_id}
    for c in contacts:
        up = int(c["up_id"])
        down = int(c["down_id"])
        if can_reach_part.get(up, False) and can_reach_part.get(down, False):
            outgoing[up].append(c)

    # BFS from source along the directed, part-reachable graph.
    parent: Dict[int, int] = {source_id: -1}
    visited = {source_id}
    order = [source_id]
    queue = [source_id]
    while queue:
        current = queue.pop(0)
        for c in outgoing.get(current, []):
            down = int(c["down_id"])
            if down == part_id:
                continue
            if down in visited:
                continue
            visited.add(down)
            parent[down] = current
            queue.append(down)
            order.append(down)

    # Entry / exit contacts for each gating component.
    body_entry: Dict[int, Dict] = {}
    for cid, p in parent.items():
        if p == -1:
            continue
        for oc in outgoing.get(p, []):
            if oc.get("down_id") == cid:
                body_entry[cid] = oc
                break
    body_exits: Dict[int, List[Dict]] = {
        cid: list(outgoing.get(cid, [])) for cid in comp_meta if cid != part_id
    }

    # If the source does not actually reach the part, there is no valid gating
    # path to calculate, even when contacts exist elsewhere.
    source_reaches_part = any(
        int(c["down_id"]) == part_id
        for cid in order
        for c in outgoing.get(cid, [])
    )
    if not source_reaches_part:
        raise GatingVelocityError(
            "Düğüm hızları hesaplanamadı: seçilen kaynak parçaya ulaşan yol "
            "üzerinde değil; CAD geometrisi kaynak ile parça arasında bağlantı "
            " içermiyor."
        )

    # Directions for the optional Darcy flux integral below.  The node graph
    # itself is now built from CAD mesh proximity, but the voxel grid is still
    # used for the 3-B Darcy field (fill time / porozite).
    unique_dirs = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Real 3B Darcy flux / projected-contact-area model.
    #
    # 1) First pass: the local flow direction for each unordered contact is
    #    the Darcy velocity vector at the shared faces, weighted by the
    #    normal flux magnitude (|v_n| * A_real).  This direction comes from
    #    the solved velocity field, not from OBB/gravity guesses.
    # 2) Second pass: each face area is projected onto the plane perpendicular
    #    to that local flow direction.  The projected area is the throat area
    #    that satisfies Q = v * A.  The flux is the signed Darcy volumetric
    #    flow from the upstream component to the downstream component.
    # ------------------------------------------------------------------
    flux_pair = np.zeros(n_keys, dtype=np.float64)
    area_pair = np.zeros(n_keys, dtype=np.float64)
    vel_sum_pair = np.zeros((3, n_keys), dtype=np.float64)
    weight_pair = np.zeros(n_keys, dtype=np.float64)
    up_by_key = np.zeros(n_keys, dtype=np.int32)
    down_by_key = np.zeros(n_keys, dtype=np.int32)
    contact_keys = np.zeros(n_keys, dtype=bool)

    for c in contacts:
        up = c.get("up_id")
        down = c.get("down_id")
        if up is None or down is None:
            continue
        k = int(min(up, down)) * mult + int(max(up, down))
        contact_keys[k] = True
        up_by_key[k] = int(up)
        down_by_key[k] = int(down)

    use_face = (
        u_m_s is not None
        and v_m_s is not None
        and w_m_s is not None
        and u_m_s.ndim == 3
        and v_m_s.ndim == 3
        and w_m_s.ndim == 3
    )
    use_cell = (
        velocity_m_s is not None
        and dx_m > 0.0
        and velocity_m_s.ndim == 4
    )

    if use_face or use_cell:
        area_face = dx_m * dx_m

        def _slice(axis: int, d: int) -> Tuple[slice, slice]:
            if d > 0:
                return slice(0, -d), slice(d, None)
            if d < 0:
                return slice(-d, None), slice(0, d)
            return slice(None), slice(None)

        def _iter_contact_faces():
            for di, dj, dk in unique_dirs:
                if abs(di) + abs(dj) + abs(dk) != 1:
                    continue
                sa0, sb0 = _slice(0, di)
                sa1, sb1 = _slice(1, dj)
                sa2, sb2 = _slice(2, dk)

                id_a = comp_id[sa0, sa1, sa2]
                id_b = comp_id[sb0, sb1, sb2]
                valid = (id_a != 0) & (id_b != 0) & (id_a != id_b)
                if not valid.any():
                    continue

                start_i = sa0.start if sa0.start is not None else 0
                start_j = sa1.start if sa1.start is not None else 0
                start_k = sa2.start if sa2.start is not None else 0

                i, j, k = np.where(valid)
                gi_i = (start_i + i).astype(np.int64)
                gj_j = (start_j + j).astype(np.int64)
                gk_k = (start_k + k).astype(np.int64)
                ni_idx = gi_i + di
                nj_idx = gj_j + dj
                nk_idx = gk_k + dk

                id_a_v = id_a[valid]
                id_b_v = id_b[valid]
                k_arr = np.minimum(id_a_v, id_b_v) * mult + np.maximum(id_a_v, id_b_v)
                in_contact = contact_keys[k_arr]
                if not in_contact.any():
                    continue

                k_arr = k_arr[in_contact]
                gi_i = gi_i[in_contact]
                gj_j = gj_j[in_contact]
                gk_k = gk_k[in_contact]
                ni_idx = ni_idx[in_contact]
                nj_idx = nj_idx[in_contact]
                nk_idx = nk_idx[in_contact]
                id_a_v = id_a_v[in_contact]

                if use_face:
                    if di == 1:
                        v0 = u_m_s[ni_idx, gj_j, gk_k]
                        v1 = v_m_s[ni_idx, gj_j, gk_k]
                        v2 = w_m_s[ni_idx, gj_j, gk_k]
                        f_A_face = f_A_z[ni_idx, gj_j, gk_k]
                    elif dj == 1:
                        v0 = u_m_s[gi_i, nj_idx, gk_k]
                        v1 = v_m_s[gi_i, nj_idx, gk_k]
                        v2 = w_m_s[gi_i, nj_idx, gk_k]
                        f_A_face = f_A_y[gi_i, nj_idx, gk_k]
                    else:  # dk == 1
                        v0 = u_m_s[gi_i, gj_j, nk_idx]
                        v1 = v_m_s[gi_i, gj_j, nk_idx]
                        v2 = w_m_s[gi_i, gj_j, nk_idx]
                        f_A_face = f_A_x[gi_i, gj_j, nk_idx]
                    v = np.stack([v0, v1, v2], axis=0).astype(np.float64, copy=False)
                else:
                    v_ref = velocity_m_s[:, gi_i, gj_j, gk_k]
                    v_nb = velocity_m_s[:, ni_idx, nj_idx, nk_idx]
                    v = (0.5 * (v_ref + v_nb)).astype(np.float64, copy=False)
                    f_A_face = np.ones_like(v[0])

                yield di, dj, dk, k_arr, id_a_v, v, f_A_face

        # First pass: find the local Darcy flow direction for each contact.
        for di, dj, dk, k_arr, id_a_v, v, f_A_face in _iter_contact_faces():
            n_vec = np.array([float(di), float(dj), float(dk)], dtype=np.float64)
            v_normal = np.einsum("i,ij->j", n_vec, v)
            A_face = area_face * f_A_face
            active = np.abs(v_normal) > 1e-6
            weight = np.where(active, np.abs(v_normal) * A_face, 0.0)
            vel_contribution = v * weight
            for idx in range(3):
                vel_sum_pair[idx] += np.bincount(
                    k_arr, weights=vel_contribution[idx], minlength=n_keys
                )
            weight_pair += np.bincount(k_arr, weights=weight, minlength=n_keys)

        # Per-contact mean flow direction from the Darcy field.
        V_flow_pair = np.zeros((3, n_keys), dtype=np.float64)
        nonzero = weight_pair > 1e-18
        if nonzero.any():
            V_flow_pair[:, nonzero] = vel_sum_pair[:, nonzero] / weight_pair[nonzero]
            norms = np.linalg.norm(V_flow_pair[:, nonzero], axis=0)
            norms[norms < 1e-18] = 1.0
            V_flow_pair[:, nonzero] /= norms

        # Second pass: projected throat area and signed flux.
        for di, dj, dk, k_arr, id_a_v, v, f_A_face in _iter_contact_faces():
            n_vec = np.array([float(di), float(dj), float(dk)], dtype=np.float64)
            v_normal = np.einsum("i,ij->j", n_vec, v)

            V = V_flow_pair[:, k_arr]
            V_norm = np.linalg.norm(V, axis=0)
            V_safe = V_norm.copy()
            V_safe[V_safe < 1e-18] = 1.0
            V_unit = V / V_safe
            cos = np.abs(np.einsum("i,ij->j", n_vec, V_unit))
            # If no Darcy direction was found for this contact, use the raw face area.
            cos = np.where(V_norm > 1e-18, cos, 1.0)

            active = np.abs(v_normal) > 1e-6
            A_proj_face = np.where(active, area_face * f_A_face * cos, 0.0)

            up_k = up_by_key[k_arr]
            sign = np.where(id_a_v == up_k, 1.0, -1.0).astype(np.float64)
            Q_face = np.where(active, sign * v_normal * area_face * f_A_face, 0.0)

            flux_pair += np.bincount(k_arr, weights=Q_face, minlength=n_keys)
            area_pair += np.bincount(k_arr, weights=A_proj_face, minlength=n_keys)

    # Assign flux and direct contact area (no trimesh, no fallback)
    for c in contacts:
        up = c.get("up_id")
        down = c.get("down_id")
        if up is None or down is None:
            continue
        k = int(min(up, down)) * mult + int(max(up, down))
        c["flux_m3_s"] = float(flux_pair[k])
        # area_b = raw (merdivenli), area_pair = projected A*cos (akışa dik)
        # Gerçek CAD boğazı akışa dik, o yüzden projected kullanıyoruz
        # Sen verdin: 886 vs 908 (projected), raw 1623 oluyor fazla
        raw_a = float(area_b[k])
        proj_a = float(area_pair[k])
        c["area_real_m2"] = proj_a if proj_a > 1e-18 else raw_a
        c["area_raw_m2"] = raw_a
        c["area_proj_m2"] = proj_a
        c["area_m2"] = c["area_real_m2"]

    # ------------------------------------------------------------------
    # THROAT AREA - gerçek geometrik temas yüzeyi alanı (mesh yüzey proximite)
    # ------------------------------------------------------------------
    for c in contacts:
        up = c.get("up_id")
        down = c.get("down_id")
        if up is None or down is None:
            continue

        body_up = comp_body.get(up)
        body_down = comp_body.get(down)

        # Direction for reporting/flow normal: centroid-to-centroid.
        d = comp_centroids[down] - comp_centroids[up]
        d_norm = float(np.linalg.norm(d))
        d_unit = d / d_norm if d_norm > 1e-12 else -g_u

        a_contact = 0.0
        a_up_mm2 = 0.0
        a_down_mm2 = 0.0
        if body_up is not None and body_down is not None:
            up_name = comp_meta.get(up, (None, str(up)))[1]
            down_name = comp_meta.get(down, (None, str(down)))[1]
            label = f"{up_name} -> {down_name}"
            a_contact, a_up_mm2, a_down_mm2, _, _ = _contact_surface_area_m2(
                body_up.mesh,
                body_down.mesh,
                np.asarray(c["centroid_mm"], dtype=np.float64),
                tol_mm=0.5,
                label=label,
                verbose=False,
                contact_normal=np.asarray(c["normal"], dtype=np.float64)
                if c.get("normal") is not None
                else None,
                contact_up_normal=np.asarray(c["up_normal"], dtype=np.float64)
                if c.get("up_normal") is not None
                else None,
                contact_down_normal=np.asarray(c["down_normal"], dtype=np.float64)
                if c.get("down_normal") is not None
                else None,
                contact_up_pt=np.asarray(c["up_pt_mm"], dtype=np.float64)
                if c.get("up_pt_mm") is not None
                else None,
                contact_down_pt=np.asarray(c["down_pt_mm"], dtype=np.float64)
                if c.get("down_pt_mm") is not None
                else None,
            )

        if a_contact <= 1e-18:
            # No geometric contact surface found: this edge cannot carry flow.
            c["area_m2"] = 0.0
            c["area_geo_m2"] = 0.0
            c["area_up_m2"] = 0.0
            c["area_down_m2"] = 0.0
            c["flow_normal"] = d_unit
            c["skip_node"] = True
            continue

        c["area_m2"] = a_contact
        c["area_geo_m2"] = a_contact
        c["area_up_m2"] = a_up_mm2 * 1e-6
        c["area_down_m2"] = a_down_mm2 * 1e-6
        c["flow_normal"] = d_unit
        c["skip_node"] = False

    # Propagate Q and compute velocities.
    Q_in: Dict[int, float] = {cid: 0.0 for cid in comp_meta if cid != part_id}
    Q_in[source_id] = float(Q_user)
    # (up_id, down_id, GatingNode) used for post-processing to unify throat areas
    node_entries: List[Tuple[int, int, GatingNode]] = []

    def _node_name(up_id: int, down_id: int) -> str:
        return f"{comp_meta[up_id][1]} → {comp_meta[down_id][1]}"

    def _node_body_type(up_id: int, down_id: int) -> str:
        return f"{comp_meta[up_id][0].name}→{comp_meta[down_id][0].name}"

    def _make_node(up_id: int, down_id: int, area_m2: float, Q: float, centroid: np.ndarray, flow_rate_m3_s: float = 0.0) -> GatingNode:
        v = float(Q / area_m2) if area_m2 > 1e-18 else 0.0
        return GatingNode(
            name=_node_name(up_id, down_id),
            body_type=_node_body_type(up_id, down_id),
            velocity_m_s=v,
            section_area_cm2=float(area_m2 * 1e4),
            centroid_mm=tuple(float(x) for x in centroid),
            flow_rate_m3_s=float(flow_rate_m3_s),
            max_velocity_m_s=v,
        )

    nodes: List[GatingNode] = []

    # Source inlet node (user velocity at the source throat).
    if source_area_m2 > 1e-18:
        src_type, src_name = comp_meta[source_id]
        source_centroid = comp_centroids[source_id]
        nodes.append(
            GatingNode(
                name=f"Kaynak → {src_name}",
                body_type=f"SOURCE→{source_section_key}",
                velocity_m_s=float(Q_user / source_area_m2),
                section_area_cm2=float(source_area_m2 * 1e4),
                centroid_mm=tuple(float(x) for x in source_centroid),
                flow_rate_m3_s=float(Q_user),
            )
        )

    for cid in order:
        Q = Q_in[cid]
        out_edges = [c for c in outgoing.get(cid, []) if not c.get("skip_node")]
        if comp_meta[cid][0] in {BodyType.RISER, BodyType.CURUFLUK}:
            continue
        if not out_edges:
            continue
        A_total = sum(c["area_m2"] for c in out_edges)
        if A_total <= 1e-18:
            raise GatingVelocityError(
                f"Düğüm hızları çözülemedi: {comp_meta[cid][1]} elemanının toplam çıkış kesit alanı sıfır. "
                "Hız kesitleri parça/geometri nedeniyle hesaplanamadı."
            )
        # Basit + akıllı hesap: gelen toplam debi, çıkış darboğaz
        # alanlarına göre orantılı bölünür.  Akış yolunda her kavşakta
        # v_common = Q / A_total ve Q_i = v_common * A_i; düğüm hızı
        # v_i = Q_i / A_i = v_common olur.  Darcy flux paylaşımına dönülmez.
        areas = np.array([float(c["area_m2"]) for c in out_edges], dtype=np.float64)
        A_total = float(areas.sum())
        if A_total <= 1e-18:
            raise GatingVelocityError(
                f"Düğüm hızları çözülemedi: {comp_meta[cid][1]} elemanının toplam çıkış kesit alanı sıfır. "
                "Hız kesitleri parça/geometri nedeniyle hesaplanamadı."
            )

        Q_branches = Q * (areas / A_total)

        for i, c in enumerate(out_edges):
            A = float(c["area_m2"])
            if A <= 1e-18:
                raise GatingVelocityError(
                    f"Düğüm hızları çözülemedi: {comp_meta[cid][1]} → "
                    f"{comp_meta[c['down_id']][1]} temas kesit alanı sıfır. "
                    "Hız kesitleri parça/geometri nedeniyle hesaplanamadı."
                )
            Q_branch = float(Q_branches[i])
            v_branch = float(Q_branch / A)
            c["Q_branch"] = Q_branch
            c["v_branch"] = v_branch
            down_id = c["down_id"]
            print(
                f"[GATING_NODE] {comp_meta[cid][1]} -> {comp_meta[down_id][1]}  "
                f"type={comp_meta[cid][0].name}→{comp_meta[down_id][0].name}  "
                f"area_cm2={A*1e4:.4f}  Q_L_s={Q_branch*1e3:.4f}  v_m_s={v_branch:.4f}",
                flush=True,
            )
            node = _make_node(cid, down_id, A, Q_branch, c["centroid_mm"], flow_rate_m3_s=Q_branch)
            nodes.append(node)
            node_entries.append((cid, down_id, node))
            if down_id != part_id:
                Q_in[down_id] += Q_branch

    if not nodes:
        raise GatingVelocityError(
            "Düğüm hızları çözülemedi: hesaplanan düğüm listesi boş. "
            "Hız kesitleri parça/geometri nedeniyle hesaplanamadı."
        )

    # Node velocities are already v = Q / A_contact from _make_node.
    # Using each contact's own area preserves distinct velocities through the
    # gating chain (source throat, sprue throat, ingate exit, etc.).

    # Sort so the report follows the BFS fill path (source first).
    up_name_to_cid = {name: cid for cid, (_, name) in comp_meta.items()}
    order_index = {cid: i for i, cid in enumerate(order)}

    def _node_order(node: GatingNode) -> int:
        if node.body_type.startswith("SOURCE"):
            return -1
        up_name = node.name.split(" → ")[0]
        cid = up_name_to_cid.get(up_name, -1)
        return order_index.get(cid, 9999)

    nodes.sort(key=_node_order)
    return nodes, comp_meta, comp_centroids, comp_id, part_id


def _effective_mold_from_bodies(
    mold: MoldMaterial,
    bodies: Optional[List[Body]],
    body_index: Optional[np.ndarray],
) -> MoldMaterial:
    """Blend the global mould with per-CORE body sand overrides.

    A casting assembly may contain several cores with different sands.  The
    body_index voxel map tells us how many voxels belong to each body; the
    returned MoldMaterial is the area-weighted average of the green-sand
    parameters (AFS, moisture, binder, compactability).  Other thermal
    properties keep the global mould values.
    """
    if bodies is None or body_index is None:
        return mold
    if not getattr(mold, "is_sand", True):
        return mold
    core_bodies = {
        b.index: b for b in bodies
        if getattr(b, "body_type", None) == BodyType.CORE
    }
    if not core_bodies:
        return mold

    max_idx = max(max(core_bodies.keys()), int(body_index.max()))
    counts = np.zeros(max_idx + 1, dtype=np.float64)
    for b in core_bodies.values():
        if b.index < counts.size:
            counts[b.index] = float(np.sum(body_index == b.index))

    total = float(counts.sum())
    if total < 1.0:
        return mold

    from dataclasses import replace

    afs = moisture = binder = compact = 0.0
    for b in core_bodies.values():
        w = counts[b.index] / total
        if b.mold_preset and b.mold_preset in MOLDS:
            base = MOLDS[b.mold_preset]
        else:
            base = mold
        afs += w * (b.mold_afs_grain_size or base.afs_grain_size)
        moisture += w * (b.mold_moisture_percent or base.moisture_percent)
        binder += w * (b.mold_binder_percent or base.binder_percent)
        compact += w * (b.mold_compactability_percent or base.compactability_percent)

    return replace(
        mold,
        afs_grain_size=afs,
        moisture_percent=moisture,
        binder_percent=binder,
        compactability_percent=compact,
    )


def _area_to_diameter_mm(area_m2: float) -> float:
    if area_m2 <= 0.0:
        return 0.0
    return 1000.0 * math.sqrt(4.0 * area_m2 / math.pi)


def _simple_hydraulic_filling_result(
    grid: np.ndarray,
    origin: np.ndarray,
    dx: float,
    bodies: Optional[List[Body]],
    design_velocity_m_s: float,
    design_section_key: str,
    design_area_m2: float,
    section_areas_m2: Optional[Dict[str, float]],
    g: Tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> FillingResult:
    """Fast Q = v * A hydraulic fill estimate without Darcy/LBM.

    Total flow rate is fixed from the user/design velocity and the selected
    section area.  When bodies are available we use the same CAD-contact
    graph and throat-area propagation as the full solver; otherwise we fall
    back to the aggregate section-area split.  3-D velocity / fill-time
    arrays are not produced; only discrete gating-node labels and scalar
    totals are returned, which is enough for the UI and the thermal solver.
    """
    section_areas_m2 = section_areas_m2 or {}
    Q_m3_s = design_velocity_m_s * design_area_m2 if design_velocity_m_s > 0.0 and design_area_m2 > 0.0 else 0.0

    # Total metal volume from the voxel grid (mm -> m).
    is_metal = grid != int(BodyType.EMPTY)
    dx_m = dx / 1000.0
    V_metal_m3 = float(np.count_nonzero(is_metal)) * (dx_m ** 3)
    fill_time_s = V_metal_m3 / Q_m3_s if Q_m3_s > 1e-12 else 0.0

    g_vec = _gravity_unit(g)
    gating_nodes: List[GatingNode] = []
    per_gate_v: Dict[str, float] = {}
    per_gate_area: Dict[str, float] = {}
    per_gate_q: Dict[str, float] = {}
    node_velocities: Dict[str, float] = {}
    ingate_contact_velocity_m_s = 0.0
    total_ingate_q = 0.0

    inlet_key = (design_section_key or "SPRUE_THROAT").upper()
    inlet_area_m2 = section_areas_m2.get(inlet_key, design_area_m2)

    if bodies:
        try:
            nodes, _, _, _, _ = _gating_node_velocities(
                grid=grid,
                origin_mm=origin,
                dx_mm=dx,
                Q_user=Q_m3_s,
                source_area_m2=design_area_m2,
                source_section_key=design_section_key,
                g=g_vec,
                bodies=bodies,
                section_areas_m2=None,
                velocity_m_s=None,
                dx_m=0.0,
                u_m_s=None,
                v_m_s=None,
                w_m_s=None,
                face_fractions=None,
            )
            gating_nodes = nodes
            node_velocities, ingate_contact_velocity_m_s = _aggregate_section_velocities(
                gating_nodes, Q_m3_s, design_area_m2
            )
            for n in gating_nodes:
                parts = n.body_type.split("→")
                if len(parts) != 2:
                    continue
                up, down = parts
                if down in ("PART", "Parça") and not up.startswith("SOURCE"):
                    gate_name = n.name.split(" → ")[0]
                    per_gate_v[gate_name] = (
                        n.max_velocity_m_s if n.max_velocity_m_s > 1e-12 else n.velocity_m_s
                    )
                    per_gate_area[gate_name] = n.section_area_cm2
                    per_gate_q[gate_name] = n.flow_rate_m3_s
            total_ingate_q = float(sum(per_gate_q.values())) if per_gate_q else 0.0
            reason = (
                f"Basit hidrolik dolum (CAD temas Q/A): Q={Q_m3_s*1e3:.3f} L/s, "
                f"kaynak={design_section_key}, V_metal={V_metal_m3*1e6:.1f} cm³, "
                f"t_fill={fill_time_s:.2f} s."
            )
            return FillingResult(
                gating_nodes=gating_nodes,
                node_velocities=node_velocities,
                ingate_contact_velocity_m_s=ingate_contact_velocity_m_s,
                Q_m3_s=Q_m3_s,
                inlet_area_m2=inlet_area_m2,
                fill_time_s=fill_time_s,
                per_gate_contact_velocity_m_s=per_gate_v,
                per_gate_contact_area_cm2=per_gate_area,
                per_gate_flow_rate_m3_s=per_gate_q,
                total_ingate_flow_m3_s=total_ingate_q,
                reason=reason,
            )
        except GatingVelocityError:
            pass

    # Fallback: aggregate section areas when no bodies/contact graph.
    gate_body_types = {
        BodyType.INGATE,
        BodyType.RUNNER,
        BodyType.SPRUE,
        BodyType.SPRUE_THROAT,
        BodyType.DISTRIBUTOR,
        BodyType.CURUFLUK,
        BodyType.FILTER,
        BodyType.POURING_BASIN,
        BodyType.COOLING_SPRUE,
    }
    total_ingate_area_m2 = 0.0

    if bodies:
        from collections import Counter
        type_counts = Counter()
        gating_bodies = [b for b in bodies if b.body_type in gate_body_types]
        for b in gating_bodies:
            type_counts[BodyType(b.body_type).name] += 1

        for b in gating_bodies:
            type_name = BodyType(b.body_type).name
            total_area_m2 = section_areas_m2.get(type_name, 0.0)
            n_type = max(type_counts[type_name], 1)
            area_m2 = total_area_m2 / n_type
            area_cm2 = area_m2 * 1e4

            if type_name == BodyType.INGATE.name:
                total_ingate_area_m2 += area_m2
                A_total_ingate = section_areas_m2.get(BodyType.INGATE.name, 0.0)
                if A_total_ingate > 1e-12:
                    q_i = Q_m3_s * (area_m2 / A_total_ingate)
                else:
                    q_i = Q_m3_s / max(n_type, 1)
                total_ingate_q += q_i
            else:
                q_i = Q_m3_s

            v_i = q_i / area_m2 if area_m2 > 1e-12 else 0.0
            centroid = tuple(float(x) for x in getattr(b, "center", (0.0, 0.0, 0.0)))
            gating_nodes.append(
                GatingNode(
                    name=f"source → {b.name}",
                    body_type=f"{type_name}→{type_name}",
                    velocity_m_s=v_i,
                    section_area_cm2=area_cm2,
                    centroid_mm=centroid,
                    flow_rate_m3_s=q_i,
                    max_velocity_m_s=v_i,
                )
            )
            node_velocities[type_name] = max(node_velocities.get(type_name, 0.0), v_i)
            if type_name == BodyType.INGATE.name:
                per_gate_v[b.name] = v_i
                per_gate_area[b.name] = area_cm2
                per_gate_q[b.name] = q_i

        if total_ingate_area_m2 > 1e-12:
            ingate_contact_velocity_m_s = Q_m3_s / total_ingate_area_m2

    reason = (
        f"Basit hidrolik dolum: Q={Q_m3_s*1e3:.3f} L/s, "
        f"kaynak={design_section_key}, V_metal={V_metal_m3*1e6:.1f} cm³, "
        f"t_fill={fill_time_s:.2f} s."
    )

    return FillingResult(
        gating_nodes=gating_nodes,
        node_velocities=node_velocities,
        ingate_contact_velocity_m_s=ingate_contact_velocity_m_s,
        Q_m3_s=Q_m3_s,
        inlet_area_m2=inlet_area_m2,
        fill_time_s=fill_time_s,
        per_gate_contact_velocity_m_s=per_gate_v,
        per_gate_contact_area_cm2=per_gate_area,
        per_gate_flow_rate_m3_s=per_gate_q,
        total_ingate_flow_m3_s=total_ingate_q,
        reason=reason,
    )


def solve_filling_flow(
    grid: np.ndarray,
    origin: np.ndarray,
    dx: float,
    casting_params,
    alloy,
    bodies=None,
    body_index: Optional[np.ndarray] = None,
    max_solver_cells: int = _FLOW_CFG.max_solver_cells,
    progress_callback=None,
    design_velocity_m_s: float = 0.0,
    design_section_key: str = "SPRUE_THROAT",
    design_area_m2: float = 0.0,
    section_areas_m2: Optional[Dict[str, float]] = None,
    mold=None,
    fast_hydraulic: bool = False,
) -> FillingResult:
    """Run the Darcy filling-flow solver and return section/node velocities.

    Parameters
    ----------
    grid : np.ndarray
        Body-type voxel grid from `build_voxel_grid`.
    origin : np.ndarray
        Grid origin in mm.
    dx : float
        Voxel size in mm.
    casting_params : CastingParameters
        User inputs; `ingate_velocity_m_s` and `velocity_section_key` drive Q.
    alloy : Alloy
        Used for density and, in later versions, viscosity.
    bodies : list[Body], optional
        Reserved for future per-body indexed simulations.
    max_solver_cells : int
        Downsample the grid so the pressure solve stays fast.
    progress_callback : callable(int), optional
        Optional progress reporting 0-100.
    design_velocity_m_s : float
        Fallback velocity (m/s) when the user leaves the input zero.
        Default comes from the gating design choke velocity.
    design_section_key : str
        Gating section that the fallback velocity refers to.
    design_area_m2 : float
        Reference area (m²) paired with `design_velocity_m_s`.  When
        present, Q = design_velocity_m_s * design_area_m2, which keeps the
        flow consistent with the gating design even if the CAD sprue is
        oversized/undersized.
    section_areas_m2 : dict[str, float]
        Optional measured cross-sectional areas (m²) for each gating section.
        If provided, node velocities use these areas instead of the voxel grid.
    fast_hydraulic : bool
        If True, bypass the 3-D Darcy/VOF solve and return a fast Q=vA
        estimate.  This is quick but does not produce 3-D velocity/fill-time
        arrays, so the flow animation will not work.

    Returns
    -------
    FillingResult
    """
    if progress_callback:
        progress_callback(2)

    g_vec = _gravity_unit(getattr(casting_params, "gravity_vector", (0.0, 0.0, -1.0)))

    # Fast hydraulic path: Q = v * A using the same CAD-contact Q/A
    # propagation as the Darcy path, but without the expensive solve.
    if fast_hydraulic and design_velocity_m_s > 0.0 and design_area_m2 > 0.0:
        return _simple_hydraulic_filling_result(
            grid=grid,
            origin=origin,
            dx=dx,
            bodies=bodies,
            design_velocity_m_s=design_velocity_m_s,
            design_section_key=design_section_key,
            design_area_m2=design_area_m2,
            section_areas_m2=section_areas_m2,
            g=g_vec,
        )

    # Blend global mould with per-CORE sand overrides before the flow solve.
    if mold is not None:
        mold = _effective_mold_from_bodies(mold, bodies, body_index)

    # The input analysis grid is kept as the reference frame for the returned
    # velocity / fill_time fields (so they match result.grid downstream).
    orig_grid = grid.copy()
    orig_origin = origin.copy()
    orig_dx = float(dx)

    # If body geometry is available, build a flow-dedicated grid fine enough to
    # capture gate cross-sections (≤ ~1.8 mm) while staying within the solver
    # cavity budget.  Otherwise fall back to the supplied analysis grid.
    ref_grid, ref_origin, ref_dx = _flow_refined_grid(
        bodies, casting_params, desired_dx_mm=_FLOW_CFG.desired_dx_mm, max_cells=max_solver_cells
    )
    if ref_grid is not None and ref_dx < dx * 0.95:
        grid, origin, dx = ref_grid, ref_origin, ref_dx

    # Downsample to keep the linear solve tractable.
    grid_c, origin_c, dx_c = _downsample_grid(grid, origin, dx, max_solver_cells)
    dx_m = dx_c / 1000.0

    g = _gravity_unit(getattr(casting_params, "gravity_vector", (0.0, 0.0, -1.0)))
    mu = max(float(getattr(alloy, "viscosity_pa_s", 0.005) or 0.005), 1e-9)
    cavity, solid = _cavity_and_solid_masks(grid_c)
    if not cavity.any():
        return FillingResult(reason="Mold cavity (non-empty voxels) not found.")

    # Keep the original (fine) grid for accurate cross-sectional areas; the
    # pressure solve uses the downsampled grid for speed.
    fine_grid = grid
    fine_cavity, _ = _cavity_and_solid_masks(fine_grid)
    fine_dx_m = float(dx) / 1000.0

    if progress_callback:
        progress_callback(10)

    # The user's velocity/reference section (e.g. INGATE) drives Q and node
    # velocities, but the *physical* pour inlet for the Darcy/animation front
    # must always be the top of the sprue/pouring basin.  Decouple them so the
    # filling simulation starts from the actual pour point, not from downstream
    # gates chosen only for the velocity input.
    section_key = getattr(casting_params, "velocity_section_key", "SPRUE")
    physical_source_key = None
    for try_key in ("SPRUE_THROAT", "POURING_BASIN", "SPRUE"):
        inlet_cells, inlet_name = _select_inlet_cells(grid_c, cavity, g, try_key)
        if inlet_cells.any():
            physical_source_key = try_key
            break
    if not inlet_cells.any():
        # Last resort: whatever section the user velocity refers to.
        inlet_cells, inlet_name = _select_inlet_cells(grid_c, cavity, g, section_key)
        physical_source_key = section_key
    vent_cells = _select_vent_cells(grid_c, cavity, g)

    if not inlet_cells.any():
        return FillingResult(reason="Inlet (sprue top) could not be detected in the voxel grid.")

    # Build Dirichlet mask / values: p=1 at inlet, p=0 at vents.
    dirichlet = inlet_cells | vent_cells
    dirichlet_value = np.where(inlet_cells, 1.0, 0.0)

    # Isolated cavities (disconnected risers etc.) need a Dirichlet cell too,
    # otherwise the Laplacian matrix becomes singular.
    dirichlet, dirichlet_value = _ensure_dirichlet_per_component(
        cavity, dirichlet, dirichlet_value, g, origin_c, dx_m
    )

    if progress_callback:
        progress_callback(20)

    # FAVOR fractional face areas from the CAD geometry.  These replace the raw
    # dx*dx area on curved/staircase surfaces so Q = v * A uses the real area.
    # At very high resolution the 4x zoom would explode memory (>60 GB for 120 M
    # cells), so fall back to binary face areas (sub=1) on large grids.
    is_metal_c = grid_c != BodyType.EMPTY
    sub_frac = 4 if is_metal_c.size < 5_000_000 else 1
    face_fractions = compute_face_fractions(is_metal_c, sub=sub_frac)
    f_A_z, f_A_y, f_A_x = face_fractions

    # Real source throat area (for reporting / validation only).
    source_real_area_m2 = _inlet_face_area_m2(inlet_cells, cavity, dx_m, face_fractions)

    # Uniform isotropic permeability: the Darcy pressure field is used only to
    # determine the flow direction/split, while the absolute velocity comes from
    # the user Q and the real voxel contact area (Q = v × A).  Face fractions
    # already carry the local cross-sectional area, so we do not add an extra
    # narrow-channel penalty that would make diagonal/small gates artificially
    # more resistive than the physics of the user-specified design implies.
    permeability_m2 = np.full(cavity.shape, (dx_m * 0.5) ** 2, dtype=np.float64)

    if progress_callback:
        progress_callback(25)

    # Pressure solve.  Viscosity enters as hydraulic conductivity K/mu.
    A, rhs, flat_idx, dirichlet_unknowns, dirichlet_value_flat = _build_laplace_matrix(
        cavity, dirichlet, dirichlet_value, permeability_m2, dx_m, face_fractions,
        viscosity_pa_s=mu,
    )
    if progress_callback:
        progress_callback(35)

    p_flat = _solve_pressure(A, rhs, dirichlet_unknowns, dirichlet_value_flat)
    p = np.zeros(cavity.shape, dtype=np.float64)
    p[cavity] = p_flat

    if progress_callback:
        progress_callback(55)

    # Face velocities (Darcy: v = -(K/μ) * dp/dx).  K is now spatially
    # variable, so narrow gates and wide runners feel their own hydraulic
    # resistance.  mu is in the matrix and in this gradient.
    u, v, w = _face_velocities(p, cavity, dx_m, permeability_m2, viscosity_pa_s=mu)

    # Determine user flow rate, preferring the FAVOR source area if it is available.
    fine_part_mask = (fine_grid == BodyType.PART) & fine_cavity
    part_volume_m3 = float(fine_part_mask.sum()) * (fine_dx_m ** 3)
    fill_time_input = float(getattr(casting_params, "t_fill_s", 0.0) or 0.0)
    user_velocity = float(getattr(casting_params, "ingate_velocity_m_s", 0.0) or 0.0)

    Q_user, area_m2, used_section = _compute_user_flow_rate(
        grid_c,
        cavity,
        dx_m,
        user_velocity,
        section_key,
        fill_time_input,
        part_volume_m3,
        g,
        design_velocity_m_s=design_velocity_m_s,
        design_section_key=design_section_key,
        design_area_m2=design_area_m2,
        fine_grid=fine_grid,
        fine_cavity=fine_cavity,
        fine_dx_m=fine_dx_m,
    )

    # Total flux leaving the inlet region in the raw pressure field.
    Q_raw = _inlet_flux_m3_s(u, v, w, inlet_cells, cavity, dx_m, face_fractions)
    if abs(Q_raw) < 1e-18:
        raise GatingVelocityError(
            "Girişten çıkan Darcy akı sıfır. Giriş bölgesi boşlukla bağlantılı değil "
            "veya geometri dejeneredir. Otomatik akı tahmini uygulanmıyor."
        )
    scale = Q_user / Q_raw
    # scale carries units of pressure (Pa) because the matrix was built with
    # K/mu and dimensionless Dirichlet p=1/0; it is the pressure drop needed
    # to drive Q_user through the Darcy medium.
    pressure_drop_pa = float(scale) if Q_raw != 0.0 else 0.0

    # Phase 2: particle-based sand-mold permeability and air-leakage estimate.
    sand_k_m2 = 0.0
    air_leak_m3_s = 0.0
    sand_phi = 0.0
    sand_d_mm = 0.0
    if (
        mold is not None
        and getattr(mold, "is_sand", True)
        and os.environ.get("JOSECAST_USE_CPP_SAND", "1").lower() in ("1", "true", "yes")
    ):
        try:
            from core.cpp_bridge import JOSECAST_CORE
            if JOSECAST_CORE is not None:
                wall = (~cavity) & ndimage.binary_dilation(cavity, iterations=1)
                sand_mask = wall.astype(np.uint8)
                p_air = max(pressure_drop_pa, 1000.0)
                k_field, sand_k_m2, air_leak_m3_s, sand_phi, sand_d_mm = (
                    JOSECAST_CORE.compute_sand_permeability(
                        sand_mask,
                        float(getattr(mold, "afs_grain_size", 50.0)),
                        float(getattr(mold, "moisture_percent", 4.0)),
                        float(getattr(mold, "binder_percent", 2.0)),
                        float(getattr(mold, "compactability_percent", 45.0)),
                        p_air,
                        1.81e-5,
                        dx_m,
                    )
                )
                # The C++ leak model treats the sand wall as one voxel thick;
                # scale to a realistic mold wall thickness (≥5 voxels or 50 mm).
                wall_thickness_m = max(5.0 * dx_m, 0.05)
                air_leak_m3_s *= dx_m / wall_thickness_m
        except Exception as exc:
            print(f"[SAND] Kum geçirgenliği hesaplanamadı: {exc}", flush=True)

    u *= scale
    v *= scale
    w *= scale

    # Cell-centered velocity components on the solver (coarse) grid.
    ux_c = 0.5 * (u[:-1] + u[1:])
    vy_c = 0.5 * (v[:, :-1] + v[:, 1:])
    wz_c = 0.5 * (w[:, :, :-1] + w[:, :, 1:])

    vmag = np.sqrt(ux_c * ux_c + vy_c * vy_c + wz_c * wz_c)
    vmag = np.nan_to_num(vmag, posinf=0.0, neginf=0.0)

    if progress_callback:
        progress_callback(75)

    # Total fill time estimate: part volume / user flow rate.
    fill_time_s = part_volume_m3 / Q_user if Q_user > 1e-18 else 0.0

    # Per-voxel fill time is computed from the gating graph after the node
    # velocities are known.  Placeholder is created here so the variable exists;
    # the actual computation follows ``_gating_node_velocities``.
    fill_time_fine = np.full(orig_grid.shape, 0.0, dtype=np.float64)

    if (
        vmag.shape == orig_grid.shape
        and abs(dx_c - orig_dx) < 1e-9
        and np.allclose(origin_c, orig_origin)
    ):
        vx_f, vy_f, vz_f = ux_c, vy_c, wz_c
    else:
        vx_f = _resample_to_grid(ux_c, origin_c, dx_c, orig_grid.shape, orig_origin, orig_dx)
        vy_f = _resample_to_grid(vy_c, origin_c, dx_c, orig_grid.shape, orig_origin, orig_dx)
        vz_f = _resample_to_grid(wz_c, origin_c, dx_c, orig_grid.shape, orig_origin, orig_dx)

    fine_metal = (orig_grid > 0) & (orig_grid != BodyType.CORE)
    velocity = np.stack(
        [
            np.where(fine_metal, vx_f, 0.0),
            np.where(fine_metal, vy_f, 0.0),
            np.where(fine_metal, vz_f, 0.0),
        ],
        axis=0,
    ).astype(np.float32)
    vmag_fine = np.linalg.norm(velocity, axis=0)

    # Air entrapment placeholders; filled from LBM/VOF or from the fallback detector.
    air_entrapment_fine = np.zeros_like(vmag_fine, dtype=np.float64)
    trapped_air_volume_m3 = 0.0
    air_entrapment_centroid_mm = np.array([], dtype=np.float64)
    orig_dx_m = orig_dx / 1000.0

    # Source throat area for the *physical* pour point (sprue/pouring basin),
    # independent of the user's velocity reference section.  Used for the source
    # node in the gating graph and for the synthetic source-node label.
    source_area_m2 = 0.0
    if section_areas_m2:
        source_area_m2 = float(section_areas_m2.get(physical_source_key, 0.0) or 0.0)
    if source_area_m2 <= 1e-18:
        face_src, _ = _section_face_cells(fine_grid, fine_cavity, physical_source_key, g, allow_fallback=False)
        if not face_src.any():
            face_src, _ = _section_face_cells(grid_c, cavity, physical_source_key, g, allow_fallback=False)
        src_dx_m = fine_dx_m if face_src.shape == fine_cavity.shape else dx_m
        source_area_m2 = float(face_src.sum()) * src_dx_m * src_dx_m
    if source_area_m2 <= 1e-18:
        source_area_m2 = float(area_m2)

    # ------------------------------------------------------------------
    # Optional 3-D free-surface Navier-Stokes / level-set solvers.
    # JOSECAST_USE_CPP_VOF=1   -> C++ fractional-step NS (OpenVDB-free dense grid)
    # JOSECAST_USE_TAICHI_VOF=1 -> Python/Taichi plug-flow VOF
    # ------------------------------------------------------------------
    use_cpp_vof = os.environ.get("JOSECAST_USE_CPP_VOF", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    use_taichi_vof = os.environ.get("JOSECAST_USE_TAICHI_VOF", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    use_cpp_lbm = os.environ.get("JOSECAST_USE_CPP_LBM", "1").lower() in (
        "1",
        "true",
        "yes",
    )
    # The C++ NS pressure projection is heavier per step; keep the grid smaller.
    vof_max_cells = int(os.environ.get("JOSECAST_CPP_VOF_MAX_CELLS", "120000"))
    if not use_cpp_vof:
        vof_max_cells = 500_000
    # LBM is lighter per step but still benefits from a coarse grid for first runs.
    lbm_max_cells = int(
        os.environ.get("JOSECAST_CPP_LBM_MAX_CELLS", str(_FLOW_CFG.lbm_max_cells))
    )

    vof_res = None
    inflow_v = 0.0
    if use_cpp_vof or use_taichi_vof or use_cpp_lbm:
        try:
            # Keep the VOF grid small enough for an interactive solve.
            lbm_vof_max_cells = lbm_max_cells if use_cpp_lbm else vof_max_cells
            vof_grid, vof_origin, vof_dx = _downsample_grid(
                grid_c, origin_c, dx_c, max_cells=lbm_vof_max_cells
            )
            vof_cavity = vof_grid != BodyType.EMPTY
            vof_inlet, _ = _select_inlet_cells(
                vof_grid, vof_cavity, g, physical_source_key
            )
            if not vof_inlet.any():
                # Last resort: use the same mask selected on the coarse grid.
                vof_inlet = (
                    _resample_to_grid(
                        inlet_cells.astype(np.float64),
                        origin_c,
                        dx_c,
                        vof_grid.shape,
                        vof_origin,
                        vof_dx,
                        fill_value=0.0,
                        order=0,
                    )
                    > 0.5
                )

            rho_vof = float(getattr(alloy, "rho_liquid_kg_m3", 7000) or 7000)
            mu_vof = float(mu)
            inflow_v = (
                float(user_velocity)
                if user_velocity > 1e-9
                else (
                    float(Q_user / source_area_m2)
                    if source_area_m2 > 1e-18
                    else 1.5
                )
            )
            # Scale the VOF inflow velocity so the volumetric flow rate Q_user is
            # preserved on the possibly-coarser VOF grid source cross-section.
            vof_g = np.asarray(g, dtype=np.float64)
            vof_axis = int(np.argmax(np.abs(vof_g)))
            axes = [0, 1, 2]
            axes.remove(vof_axis)
            vof_A_source = np.sum(vof_inlet, axis=tuple(axes))
            max_vof_source = int(vof_A_source.max())
            vof_dx_m = vof_dx / 1000.0
            if max_vof_source > 0 and Q_user > 1e-18:
                vof_source_area_m2 = max_vof_source * vof_dx_m * vof_dx_m
                vof_inflow_v = float(Q_user / max(vof_source_area_m2, 1e-18))
            else:
                vof_inflow_v = inflow_v
            # Initial signed-distance field: negative inside the inlet, positive
            # in the empty cavity with distance measured in voxel units.
            phi_vof = np.full(vof_grid.shape, 999.0, dtype=np.float64)
            with np.errstate(invalid="ignore"):
                dist_to_inlet = ndimage.distance_transform_edt(
                    vof_cavity & ~vof_inlet
                )
            phi_vof[vof_cavity] = dist_to_inlet[vof_cavity] - 1.0
            phi_vof[vof_inlet] = -1.0
            # Allow enough simulation time; VOF itself is fast so a generous
            # multiple of the gating fill time is fine.
            if fill_time_s > 0.0 and np.isfinite(fill_time_s):
                t_max_vof = max(2.0, fill_time_s * 50.0)
            else:
                est_volume_m3 = float(vof_cavity.sum()) * vof_dx_m ** 3
                t_max_vof = (
                    2.0
                    if Q_user <= 1e-18
                    else max(2.0, est_volume_m3 / Q_user * 4.0)
                )

            vof_outlet = _select_lbm_outlet_cells(vof_grid, vof_cavity, g, mold=mold)

            if use_cpp_lbm:
                from core.cpp_bridge import JOSECAST_CORE

                if JOSECAST_CORE is None:
                    raise RuntimeError("josecast_core C++ module is not available")
                vof_inflow_v = float(vof_inflow_v)
                vof_inflow_v = max(vof_inflow_v, 0.01)
                # C++ binding expects a plain Python list for the gravity vector.
                lbm_g = [float(x) for x in g]

                # The compiled C++ LBM is called with exactly 12 positional
                # arguments.  Older Windows .pyd builds expose 12 positional-only
                # arguments; newer builds have default optional target_velocity /
                # inlet_distance arrays, so 12 arguments is safe on both.
                print(
                    f"[LBM] C++ D3Q19 solve starting: grid={vof_grid.shape}, "
                    f"dx={vof_dx_m:.4f} m, inflow={vof_inflow_v:.3f} m/s, t_max={t_max_vof:.3f} s",
                    flush=True,
                )
                (
                    ft,
                    vmag,
                    vel,
                    phi,
                    trap,
                    trap_volume_m3,
                    success,
                    final_t,
                    filled_frac,
                    steps,
                ) = JOSECAST_CORE.solve_lbm_filling(
                    vof_grid.astype(np.uint8, copy=False),
                    vof_inlet.astype(np.uint8, copy=False),
                    vof_outlet.astype(np.uint8, copy=False),
                    float(vof_dx_m),
                    lbm_g,
                    float(rho_vof),
                    float(mu_vof / rho_vof),
                    vof_inflow_v,
                    float(t_max_vof),
                    int(os.environ.get("JOSECAST_CPP_LBM_MAX_STEPS", "12000")),
                    float(os.environ.get("JOSECAST_CPP_LBM_CFL", "0.3")),
                    float(os.environ.get("JOSECAST_CPP_LBM_SMAG", "0.18")),
                )
                vof_res = {
                    "fill_time": ft,
                    "velocity_magnitude": vmag,
                    "velocity": vel,
                    "phi": phi,
                    "air_entrapment": trap,
                    "trapped_volume_m3": float(trap_volume_m3),
                    "success": success,
                    "final_t": final_t,
                    "filled_fraction": filled_frac,
                    "steps": steps,
                }
                if not success or filled_frac < 0.9999:
                    import warnings

                    warnings.warn(
                        f"[LBM] solver did not fully fill the cavity: success={success}, "
                        f"filled_frac={filled_frac:.4f}, final_t={final_t:.4f} s, steps={steps}"
                    )
                    vof_res = None
            elif use_cpp_vof:
                from core.cpp_bridge import JOSECAST_CORE

                if JOSECAST_CORE is None:
                    raise RuntimeError("josecast_core C++ module is not available")
                vof_inflow_v = float(vof_inflow_v)
                vof_inflow_v = max(vof_inflow_v, 0.01)
                ft, vmag, vel, phi, success, final_t, filled_frac, steps = (
                    JOSECAST_CORE.solve_ns_vof(
                        vof_grid.astype(np.uint8, copy=False),
                        phi_vof,
                        vof_inlet.astype(np.uint8, copy=False),
                        float(vof_dx_m),
                        [float(x) for x in g],
                        float(rho_vof),
                        float(mu_vof / rho_vof),
                        vof_inflow_v,
                        float(t_max_vof),
                        int(os.environ.get("JOSECAST_CPP_VOF_MAX_STEPS", "1200")),
                        float(os.environ.get("JOSECAST_CPP_VOF_CFL", "2.0")),
                        int(os.environ.get("JOSECAST_CPP_VOF_PRESSURE_ITERS", "40")),
                        float(os.environ.get("JOSECAST_CPP_VOF_PRESSURE_TOL", "1e-5")),
                    )
                )
                vof_res = {
                    "fill_time": ft,
                    "velocity_magnitude": vmag,
                    "velocity": vel,
                    "phi": phi,
                    "success": success,
                    "final_t": final_t,
                    "filled_fraction": filled_frac,
                    "steps": steps,
                }
                if not success or filled_frac < 0.9999:
                    vof_res = None
            elif use_taichi_vof:
                from core import taichi_ns

                vof_res = taichi_ns.solve(
                    grid=vof_grid,
                    phi_init=phi_vof,
                    source_mask=vof_inlet.astype(np.uint8),
                    dx=vof_dx_m,
                    g=g,
                    rho=rho_vof,
                    nu=mu_vof / rho_vof,
                    inflow_velocity=vof_inflow_v,
                    t_max=t_max_vof,
                    max_steps=1200,
                    pressure_iters=40,
                    reinit_iters=2,
                    cfl=2.0,
                )
                if vof_res is None or vof_res.get("filled_fraction", 0.0) < 1e-6:
                    vof_res = None
        except Exception as exc:
            print(f"[VOF] Hata: {exc}", flush=True)
            vof_res = None

    if vof_res is not None:
        # Use the LBM/VOF fill-time field for the transient animation, but keep
        # the Darcy velocity field for per-point flow visualisation.
        fill_time_c = np.asarray(vof_res["fill_time"], dtype=np.float64)
        # Coarse cells that the LBM did not reach carry inf.  Trilinear
        # resampling of inf creates NaN on the fine grid, so replace them with a
        # finite sentinel before resampling and clamp later.
        fill_time_c = np.where(np.isfinite(fill_time_c), fill_time_c, 1e12)
        fill_time_fine = _resample_to_grid(
            fill_time_c,
            vof_origin,
            vof_dx,
            orig_grid.shape,
            orig_origin,
            orig_dx,
            fill_value=1e12,
            order=1,
        )
        fill_time_fine = np.where(fine_metal, fill_time_fine, 0.0)
        vof_final_t = float(vof_res.get("final_t", 0.0))
        vof_full = float(vof_res.get("filled_fraction", 0.0)) >= 0.9999

        # LBM front arrival times are kinematic (distance / velocity); for an
        # expanding cavity the last cell arrives much earlier than the
        # volumetric fill time (V_part / Q).  Scale the field so the final
        # arrival equals the volumetric estimate while keeping the front order.
        lbm_valid = fine_metal & (fill_time_fine > 0) & np.isfinite(fill_time_fine)
        if lbm_valid.any():
            lbm_max_t = float(np.nanmax(fill_time_fine[lbm_valid]))
            if fill_time_s > 1e-18 and lbm_max_t > 1e-18 and fill_time_s > lbm_max_t:
                lbm_scale = fill_time_s / lbm_max_t
                fill_time_fine = fill_time_fine * lbm_scale
                vof_final_t *= lbm_scale
        if vof_full and vof_final_t > 0.0:
            # Cells missed by the coarse VOF grid (thin features) are treated as
            # filled by the same total time; avoid letting the 1e12 sentinel leak
            # into the reported fill time.
            fill_time_fine = np.where(
                fine_metal & (fill_time_fine > vof_final_t),
                vof_final_t,
                fill_time_fine,
            )
        # Guard against any NaN/inf leaked by the interpolator around coarse
        # sentinel cells; these cells are still part of the filled metal domain.
        fill_time_fine = np.where(
            fine_metal,
            np.nan_to_num(fill_time_fine, nan=vof_final_t, posinf=vof_final_t, neginf=vof_final_t),
            0.0,
        )
        # The physical source section (sprue / pouring basin) is the metal
        # reservoir; mark every cell in it as filled at t=0 so the animation
        # shows the source vessel full from the very first frame and the front
        # advances from the source body into the runner/gates.
        if physical_source_key:
            source_type = getattr(BodyType, physical_source_key, None)
            if source_type is not None:
                source_mask = fine_metal & (orig_grid == source_type)
                if source_mask.any():
                    fill_time_fine[source_mask] = 0.0
        if fine_metal.any():
            valid = fine_metal & (fill_time_fine < 1e9)
            if valid.any():
                max_fill_t = float(np.nanmax(np.where(valid, fill_time_fine, np.nan)))
                if np.isfinite(max_fill_t) and max_fill_t > 0.0:
                    fill_time_s = max_fill_t

        # Air entrapment from the LBM/VOF trap field: resample the coarse binary
        # pocket mask to the fine grid and, for permeable molds (sand), let
        # near-surface air escape through the mold parting line.
        if vof_res.get("air_entrapment") is not None:
            trap_c = np.asarray(vof_res["air_entrapment"], dtype=np.float64)
            air_entrapment_fine = _resample_to_grid(
                trap_c,
                vof_origin,
                vof_dx,
                orig_grid.shape,
                orig_origin,
                orig_dx,
                fill_value=0.0,
                order=0,
            )
            air_entrapment_fine = np.clip(air_entrapment_fine, 0.0, 1.0)
            air_entrapment_fine = np.where(fine_metal, air_entrapment_fine, 0.0)

            # Permeability-aware correction: in sand molds some trapped air near
            # the surface can vent through the mold parting line; in ceramic or
            # metal molds the air stays trapped.
            permeability_proxy = float(getattr(mold, "permeability_proxy", 1.0))
            if fine_metal.any() and permeability_proxy > 1e-6:
                dist_to_surface_mm = ndimage.distance_transform_edt(
                    fine_metal, sampling=orig_dx
                )
                # vent_depth: sand ~22 mm, ceramic ~2 mm, metal ~0 mm.
                vent_depth_mm = 2.0 + 20.0 * np.clip(permeability_proxy, 0.0, 1.0)
                base_escape = 0.1 + 0.25 * np.clip(permeability_proxy, 0.0, 1.0)
                escape_factor = base_escape + (
                    np.clip(permeability_proxy, 0.0, 1.0) - base_escape
                ) * np.exp(-dist_to_surface_mm / max(vent_depth_mm, 1e-3))
                air_entrapment_fine = np.where(
                    fine_metal,
                    np.clip(air_entrapment_fine * (1.0 - escape_factor), 0.0, 1.0),
                    0.0,
                )

            trapped_mask = air_entrapment_fine > 0.3
            if trapped_mask.any():
                trapped_air_volume_m3 = float(np.sum(trapped_mask)) * (orig_dx_m ** 3)
                idx = np.argwhere(trapped_mask)
                centroid_vox = idx.mean(axis=0)
                air_entrapment_centroid_mm = orig_origin + centroid_vox * orig_dx

    # Post-process 3-D flow turbulence metrics (Re, turbulent intensity).
    if fine_metal.any():
        dt_m = ndimage.distance_transform_edt(fine_metal, sampling=orig_dx_m)
        D_h = 2.0 * dt_m
        rho_liq = float(getattr(alloy, "rho_liquid_kg_m3", 7000.0))
        reynolds_field = np.where(
            fine_metal & (D_h > 0.0), rho_liq * vmag_fine * D_h / mu, 0.0
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            turb_intensity = np.where(
                fine_metal & (reynolds_field > 1.0),
                0.16 * np.power(reynolds_field, -0.125),
                0.0,
            )
    else:
        reynolds_field = np.zeros_like(vmag_fine)
        turb_intensity = np.zeros_like(vmag_fine)

    # Contact-node velocities / areas for every gating-gating and gating-part interface.
    # Use the solver (coarse) grid, the staggered face velocities and the FAVOR
    # fractional face areas so the real contact area is used in Q = v * A.
    try:
        gating_nodes, comp_meta, comp_centroids, comp_id, part_id = _gating_node_velocities(
            grid_c,
            origin_c,
            dx_c,
            Q_user,
            source_area_m2,
            physical_source_key,
            g,
            bodies,
            section_areas_m2=section_areas_m2,
            u_m_s=u,
            v_m_s=v,
            w_m_s=w,
            dx_m=dx_c / 1000.0,
            face_fractions=face_fractions,
        )
    except GatingVelocityError as exc:
        # Hata varsa düğüm hızı hesabını atla; Darcy/porozite devam etsin.
        print(f"[GATING] {exc}", flush=True)
        gating_nodes = []
        comp_meta = {}
        comp_centroids = {}
        comp_id = np.zeros(grid_c.shape, dtype=np.int32)
        part_id = 1
        node_v = {}
        v_ingate_contact = 0.0

    # Local 3-D anisotropic Darcy–Forchheimer on each gating body.  Boundary
    # pressures are interpolated from the global voxel pressure field and the
    # flow rate is fixed by the global gating graph, giving smooth wall-aligned
    # gate velocities without disturbing the global fill solution.
    gate_flow_results: Dict[str, Dict[str, float]] = {}
    gate_results: Dict[str, Any] = {}
    enable_gate_mesh = bool(getattr(casting_params, "enable_gate_mesh", False))
    if gating_nodes and bodies and enable_gate_mesh:
        try:
            gate_results, gating_nodes = solve_gate_flows(
                bodies=bodies,
                grid=grid_c,
                origin_mm=origin_c,
                dx_mm=dx_c,
                p_dim=p,
                pressure_drop_pa=pressure_drop_pa,
                mu_pa_s=mu,
                rho_kg_m3=rho_liq,
                gating_nodes=gating_nodes,
                Q_user_m3_s=Q_user,
                gravity_vector=g,
            )
            for name, r in gate_results.items():
                gate_flow_results[name] = {
                    "section_velocity_m_s": float(r.section_velocity_m_s),
                    "peak_velocity_m_s": float(r.peak_velocity_m_s),
                    "outlet_flux_m3_s": float(r.outlet_flux_m3_s),
                    "pressure_drop_pa": float(r.pressure_drop_pa),
                    "forchheimer_pressure_drop_pa": float(r.forchheimer_pressure_drop_pa),
                    "total_pressure_drop_pa": float(r.total_pressure_drop_pa),
                    "max_reynolds": float(r.reynolds.max()),
                    "max_froude": float(r.froude.max()),
                    "air_entrainment_cells": int(r.air_entrainment.sum()),
                    "n_cells": int(len(r.pressures)),
                }
        except Exception as exc:
            print(f"[GATE_MESH] {exc}", flush=True)

    # ------------------------------------------------------------------
    # Volume-aware graph-based fill time: gating vessels fill sequentially,
    # then the cavity fills as a single rising metal level driven by the total
    # ingate flow Q and local cavity cross-section A(z).
    # ------------------------------------------------------------------
    if bodies is not None and body_index is not None and vof_res is None:
        fill_time_fine = _compute_fill_time_volume_layer(
            orig_grid,
            orig_origin,
            orig_dx,
            g,
            gating_nodes,
            bodies,
            body_index,
            fill_time_s,
        )
        fill_time_fine = np.where(orig_grid == BodyType.EMPTY, 0.0, fill_time_fine)
        # The reported fill time is the actual last-metal-arrival time, which
        # includes gating priming in addition to the cavity fill.
        fine_metal_for_time = (
            (orig_grid > 0)
            & (orig_grid != BodyType.CORE)
            & np.isfinite(fill_time_fine)
        )
        if fine_metal_for_time.any():
            max_fill_t = float(np.nanmax(fill_time_fine[fine_metal_for_time]))
            if np.isfinite(max_fill_t) and max_fill_t > fill_time_s:
                fill_time_s = max_fill_t

    # Sand/air backpressure: if the mold sand cannot leak displaced air as
    # fast as the metal is entering, the effective fill time increases because
    # trapped air resists the advancing front.
    if air_leak_m3_s > 1e-18 and Q_user > 1e-18:
        p_sand = max(pressure_drop_pa, 1000.0)
        k_sand = air_leak_m3_s / p_sand
        p_back = min(Q_user / max(k_sand, 1e-18), p_sand)
        p_metal = max(p_sand - p_back, 1e-12)
        backpressure_factor = min(max(p_sand / p_metal, 1.0), 5.0)
        fill_time_s *= backpressure_factor

    # Collect every node that feeds the part directly as a "gate" (meme).
    per_gate_v = {}
    per_gate_area = {}
    per_gate_q = {}
    for n in gating_nodes:
        parts = n.body_type.split("→")
        if len(parts) != 2:
            continue
        up, down = parts
        if down in ("PART", "Parça") and up != "SOURCE":
            gate_name = n.name.split(" → ")[0]
            per_gate_v[gate_name] = (
                n.max_velocity_m_s if n.max_velocity_m_s > 1e-12 else n.velocity_m_s
            )
            per_gate_area[gate_name] = n.section_area_cm2
            per_gate_q[gate_name] = n.flow_rate_m3_s
    total_ingate_flow_m3_s = float(sum(per_gate_q.values())) if per_gate_q else 0.0

    # Aggregate section velocities and ingate contact velocity directly from
    # the computed contact nodes so the report/viewer match the 3-D labels.
    if gating_nodes:
        node_v, v_ingate_contact = _aggregate_section_velocities(
            gating_nodes, Q_user, area_m2, gate_results=gate_results or None
        )
    else:
        node_v = {}
        v_ingate_contact = 0.0
    if per_gate_v:
        v_ingate_contact = float(np.mean(list(per_gate_v.values())))

    if progress_callback:
        progress_callback(95)

    filter_recommendation = _recommend_filter(gating_nodes, Q_user, alloy)

    if vof_res is not None:
        if use_cpp_lbm:
            reason = (
                f"C++ LBM D3Q19 dolum: giriş '{used_section}', "
                f"Q={Q_user*6e4:.2f} L/dak, kaynak hızı={inflow_v:.3f} m/s, "
                f"tahmini doldurma süresi={fill_time_s:.2f} s, "
                f"basınç düşümü={pressure_drop_pa:.1f} Pa."
            )
        else:
            reason = (
                f"Taichi VOF/Navier-Stokes dolum: giriş '{used_section}', "
                f"Q={Q_user*6e4:.2f} L/dak, kaynak hızı={inflow_v:.3f} m/s, "
                f"tahmini doldurma süresi={fill_time_s:.2f} s, "
                f"basınç düşümü={pressure_drop_pa:.1f} Pa."
            )
    else:
        reason = (
            f"Darcy akış çözümü: giriş '{used_section}', Q={Q_user*6e4:.2f} L/dak, "
            f"girdi hızı/alan={user_velocity:.3f} m/s / {area_m2*1e4:.2f} cm², "
            f"tahmini doldurma süresi={fill_time_s:.2f} s, "
            f"basınç düşümü={pressure_drop_pa:.1f} Pa, "
            f"kum geçirgenliği K={sand_k_m2:.2e} m², "
            f"hava kaçağı={air_leak_m3_s*6e4:.3f} L/dak."
        )

    return FillingResult(
        node_velocities=node_v,
        ingate_contact_velocity_m_s=v_ingate_contact,
        Q_m3_s=Q_user,
        inlet_area_m2=area_m2,
        fill_time_s=fill_time_s,
        velocity_magnitude=vmag_fine,
        velocity=velocity,
        fill_time=fill_time_fine,
        solver_grid=grid_c,
        solver_dx_mm=dx_c,
        pressure=p,
        reason=reason,
        per_gate_contact_velocity_m_s=per_gate_v,
        per_gate_contact_area_cm2=per_gate_area,
        per_gate_flow_rate_m3_s=per_gate_q,
        total_ingate_flow_m3_s=total_ingate_flow_m3_s,
        gating_nodes=gating_nodes,
        pressure_drop_pa=pressure_drop_pa,
        sand_permeability_m2=sand_k_m2,
        air_leak_rate_m3_s=air_leak_m3_s,
        sand_porosity=sand_phi,
        sand_grain_diameter_mm=sand_d_mm,
        reynolds=reynolds_field.astype(np.float32),
        turbulence_intensity=turb_intensity.astype(np.float32),
        filter_recommendation=filter_recommendation,
        gate_flow_results=gate_flow_results,
        air_entrapment=air_entrapment_fine,
        trapped_air_volume_m3=trapped_air_volume_m3,
        air_entrapment_centroid_mm=air_entrapment_centroid_mm,
    )
