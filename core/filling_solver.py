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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse import linalg as spla

from core.types import Body, BodyType, FillingResult, GatingNode


def _downsample_grid(
    grid: np.ndarray,
    origin: np.ndarray,
    dx: float,
    max_cells: int = 200_000,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Downsample a body-type grid with nearest-neighbour interpolation.

    The goal is to keep the pressure Poisson solve fast while still capturing
    the gating geometry.
    """
    nx, ny, nz = grid.shape
    cells = nx * ny * nz
    if cells <= max_cells:
        return grid.copy(), origin.copy(), dx

    # Choose an integer factor that brings the cell count below the limit.
    factor = int(np.ceil((cells / max_cells) ** (1.0 / 3.0)))
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


def _cavity_and_solid_masks(grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return cavity (mold cavity incl. gating+part) and solid masks."""
    cavity = grid != BodyType.EMPTY
    solid = ~cavity
    return cavity, solid


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
    labeled, n = ndimage.label(cavity, structure=np.ones((3, 3, 3), dtype=int))
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
    """Return a boolean mask of cells on the upstream or downstream boundary.

    side='up'  -> cells whose neighbour in the -g direction is not in the mask.
    side='down'-> cells whose neighbour in the +g direction is not in the mask.
    """
    boundary = np.zeros_like(mask, dtype=bool)
    axis = _roll_axis_from_g(g)
    if axis is None:
        # Gravity not aligned to a cardinal axis: fall back to projection max.
        return boundary

    sign = int(np.sign(g[axis]))
    # For axis=0: up-stream neighbour index = i - sign, down-stream = i + sign.
    # Use zero-padding to avoid periodic wrap-around errors.
    pad_width = [(1, 1) if a == axis else (0, 0) for a in range(3)]
    padded = np.pad(mask, pad_width, mode="constant", constant_values=False)

    # Slice padded so the result has the same shape as mask.
    if side == "up":
        # neighbour in -g direction: for original cell i, value is mask[i - sign]
        if sign == 1:
            neigh_slice = [slice(0, -2) if a == axis else slice(None) for a in range(3)]
        else:
            neigh_slice = [slice(2, None) if a == axis else slice(None) for a in range(3)]
    else:
        # neighbour in +g direction: for original cell i, value is mask[i + sign]
        if sign == 1:
            neigh_slice = [slice(2, None) if a == axis else slice(None) for a in range(3)]
        else:
            neigh_slice = [slice(0, -2) if a == axis else slice(None) for a in range(3)]

    neighbor = padded[tuple(neigh_slice)]
    boundary = mask & ~neighbor
    return boundary


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
        "SPRUE_THROAT": BodyType.SPRUE,
        "POURING_BASIN": BodyType.POURING_BASIN,
        "RUNNER": BodyType.RUNNER,
        "DISTRIBUTOR": BodyType.DISTRIBUTOR,
        "CURUFLUK": BodyType.CURUFLUK,
        "INGATE": BodyType.INGATE,
        "FILTER": BodyType.FILTER,
    }
    body_type = type_map.get(key_lower, BodyType.SPRUE)

    mask = (grid == body_type) & cavity
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
    """Select vent cells: top of PART / RISER along the -g (upstream) direction."""
    part_or_riser = np.isin(grid, [BodyType.PART, BodyType.RISER]) & cavity
    vent = _find_boundary_cells_along(part_or_riser, g, side="up") & cavity
    if not vent.any():
        # Use any cavity cell at the upstream (top) boundary.
        vent = _find_boundary_cells_along(cavity, g, side="up") & cavity
    return vent


def _build_laplace_matrix(
    cavity: np.ndarray,
    dirichlet: np.ndarray,
    dirichlet_value: np.ndarray,
) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Build a 7-point Laplacian matrix on the cavity with Dirichlet cells.

    Solid neighbours are treated as zero-flux (Neumann) boundaries.
    """
    flat_idx = np.full(cavity.shape, -1, dtype=np.int32)
    flat_idx[cavity] = np.arange(int(cavity.sum()))
    n_unknowns = int(cavity.sum())
    coords = np.argwhere(cavity)

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    rhs = np.zeros(n_unknowns, dtype=np.float64)

    neighbours = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ]

    for idx in range(n_unknowns):
        i, j, k = coords[idx]

        if dirichlet[i, j, k]:
            rows.append(idx)
            cols.append(idx)
            data.append(1.0)
            rhs[idx] = float(dirichlet_value[i, j, k])
            continue

        diag = 0
        for di, dj, dk in neighbours:
            ni, nj, nk = i + di, j + dj, k + dk
            if 0 <= ni < cavity.shape[0] and 0 <= nj < cavity.shape[1] and 0 <= nk < cavity.shape[2] and cavity[ni, nj, nk]:
                nidx = int(flat_idx[ni, nj, nk])
                diag += 1
                if dirichlet[ni, nj, nk]:
                    rhs[idx] -= float(dirichlet_value[ni, nj, nk])
                else:
                    rows.append(idx)
                    cols.append(nidx)
                    data.append(1.0)
            # else solid/outside: zero-flux (Neumann)

        if diag == 0:
            rows.append(idx)
            cols.append(idx)
            data.append(1.0)
        else:
            rows.append(idx)
            cols.append(idx)
            data.append(-float(diag))

    A = csr_matrix((data, (rows, cols)), shape=(n_unknowns, n_unknowns))
    return A, rhs, flat_idx


def _solve_pressure(
    A: csr_matrix,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve the sparse linear system; fall back to direct if iterative fails."""
    try:
        p, info = spla.cg(A, rhs, atol=0.0, rtol=1e-9, maxiter=500)
        if info == 0:
            return p
    except Exception:
        pass

    try:
        p = spla.spsolve(A, rhs)
    except Exception:
        # Extremely ill-conditioned; use least-squares fallback.
        p = np.zeros_like(rhs)
    return p


def _face_velocities(
    p: np.ndarray,
    cavity: np.ndarray,
    dx: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute staggered face velocities u = -dp/dx, v = -dp/dy, w = -dp/dz.

    Velocities are zero on faces adjacent to a solid cell.
    """
    u = np.zeros((p.shape[0] + 1, p.shape[1], p.shape[2]), dtype=np.float64)
    v = np.zeros((p.shape[0], p.shape[1] + 1, p.shape[2]), dtype=np.float64)
    w = np.zeros((p.shape[0], p.shape[1], p.shape[2] + 1), dtype=np.float64)

    # interior faces
    u[1:-1, :, :] = -(p[1:] - p[:-1]) / dx
    v[:, 1:-1, :] = -(p[:, 1:] - p[:, :-1]) / dx
    w[:, :, 1:-1] = -(p[:, :, 1:] - p[:, :, :-1]) / dx

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


def _inlet_flux_m3_s(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    source: np.ndarray,
    cavity: np.ndarray,
    dx: float,
) -> float:
    """Total flux (m³/s) leaving the source mask into the rest of the cavity."""
    area = dx * dx
    flux = 0.0

    # x-faces: u[1:-1] shape (nx-1, ny, nz) between cells (i-1) and (i)
    left_source = source[:-1] & ~source[1:] & cavity[1:]
    right_source = source[1:] & ~source[:-1] & cavity[:-1]
    ux = u[1:-1]
    flux += float(ux[left_source].sum()) * area
    flux -= float(ux[right_source].sum()) * area

    # y-faces
    down_source = source[:, :-1] & ~source[:, 1:] & cavity[:, 1:]
    up_source = source[:, 1:] & ~source[:, :-1] & cavity[:, :-1]
    vy = v[:, 1:-1]
    flux += float(vy[down_source].sum()) * area
    flux -= float(vy[up_source].sum()) * area

    # z-faces
    back_source = source[:, :, :-1] & ~source[:, :, 1:] & cavity[:, :, 1:]
    front_source = source[:, :, 1:] & ~source[:, :, :-1] & cavity[:, :, :-1]
    wz = w[:, :, 1:-1]
    flux += float(wz[back_source].sum()) * area
    flux -= float(wz[front_source].sum()) * area

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
        "SPRUE_THROAT": [BodyType.SPRUE],
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
    key = used_section.upper()
    if key in ("INGATE", "SPRUE_THROAT"):
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
    """Convert user velocity, fill time, or design velocity into a total flow rate Q."""
    a_grid = fine_grid if fine_grid is not None else grid
    a_cavity = fine_cavity if fine_cavity is not None else cavity
    a_dx = fine_dx_m if fine_dx_m is not None else dx_m

    used_section = (section_key or "SPRUE").upper()
    design_section = (design_section_key or used_section).upper()

    if velocity_m_s > 0.0:
        # When the user supplies a velocity, honour the paired reference area
        # (from the gating engine or from a manual SectionDialog pick) instead
        # of the raw voxel face area, so Q = v × A_reference.
        if design_area_m2 > 1e-18 and used_section == design_section:
            area_m2 = float(design_area_m2)
            face, _ = _section_face_cells(a_grid, a_cavity, section_key, g)
        else:
            face, used_section = _section_face_cells(a_grid, a_cavity, section_key, g)
            area_m2 = float(face.sum()) * (a_dx * a_dx)
        if area_m2 > 0.0:
            Q = velocity_m_s * area_m2
            return Q, area_m2, used_section
    elif fill_time_s > 0.0 and part_volume_m3 > 0.0:
        face, used_section = _section_face_cells(a_grid, a_cavity, section_key, g)
        area_m2 = float(face.sum()) * (a_dx * a_dx)
        Q = part_volume_m3 / fill_time_s
        return Q, area_m2, used_section
    elif design_velocity_m_s > 0.0:
        used_section = design_section
        face, _ = _section_face_cells(a_grid, a_cavity, design_section_key, g)
        area_m2 = float(face.sum()) * (a_dx * a_dx)
        # Use the design reference area (e.g. choke area from the gating engine)
        # instead of the raw voxel face area, so Q is consistent with the design.
        Q_area_m2 = design_area_m2 if design_area_m2 > 1e-18 else area_m2
        Q = design_velocity_m_s * Q_area_m2
        return Q, Q_area_m2, used_section

    # Last resort: a tiny flow to allow a solve; caller will report no user input.
    if "area_m2" not in locals():
        face, used_section = _section_face_cells(a_grid, a_cavity, section_key, g)
        area_m2 = float(face.sum()) * (a_dx * a_dx)
    if area_m2 > 0.0:
        Q = 0.01 * area_m2
    else:
        Q = 1.0
    return Q, area_m2, used_section


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

    while heap:
        t, i, j, k = heapq.heappop(heap)
        if visited[i, j, k]:
            continue
        visited[i, j, k] = True
        if t > fill[i, j, k] + 1e-12:
            continue
        for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            ni, nj, nk = i + di, j + dj, k + dk
            if not (0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]):
                continue
            if visited[ni, nj, nk] or not cavity[ni, nj, nk]:
                continue
            v_avg = 0.5 * (max(vmag[i, j, k], 1e-6) + max(vmag[ni, nj, nk], 1e-6))
            dt = dx_m / v_avg
            t_new = t + dt
            if t_new < fill[ni, nj, nk]:
                fill[ni, nj, nk] = t_new
                heapq.heappush(heap, (t_new, ni, nj, nk))

    fill[~cavity] = 0.0
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
) -> Tuple[Dict[str, float], float]:
    """Derive section-aggregate velocities and the ingate contact velocity.

    The aggregate values are taken directly from the contact-node values so the
    report and the 3-D labels always agree.  The source (SPRUE_THROAT) velocity
    falls back to Q / inlet_area when no upstream contact exists.
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

    for n in gating_nodes:
        parts = n.body_type.split("→")
        if len(parts) != 2:
            continue
        up, down = parts
        if up == "SOURCE" and down in section_velocities:
            section_velocities[down].append(n.velocity_m_s)
            continue
        if up == "SPRUE_THROAT":
            section_velocities["SPRUE_THROAT"].append(n.velocity_m_s)
        if up == "SPRUE" and down != "PART":
            section_velocities["SPRUE_BASE"].append(n.velocity_m_s)
        if up == "RUNNER":
            section_velocities["RUNNER"].append(n.velocity_m_s)
        if up == "DISTRIBUTOR":
            section_velocities["DISTRIBUTOR"].append(n.velocity_m_s)
        if up == "CURUFLUK":
            section_velocities["CURUFLUK"].append(n.velocity_m_s)
        if up == "FILTER":
            section_velocities["FILTER"].append(n.velocity_m_s)
        if up == "INGATE":
            section_velocities["INGATE"].append(n.velocity_m_s)
        if up == "RISER" or down == "RISER":
            section_velocities["RISER"].append(n.velocity_m_s)

    node_v: Dict[str, float] = {}
    for key, vals in section_velocities.items():
        if vals:
            node_v[key] = float(np.mean(vals))

    # No upstream contact for the source sprue: use Q / inlet area.
    if "SPRUE_THROAT" not in node_v and inlet_area_m2 > 1e-18:
        node_v["SPRUE_THROAT"] = float(Q_user / inlet_area_m2)

    ingate_vals = section_velocities["INGATE"]
    v_ingate_contact = float(np.mean(ingate_vals)) if ingate_vals else 0.0

    return node_v, v_ingate_contact


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
) -> List[GatingNode]:
    """Compute contact-node velocities from a pure hydraulic Q = vA network.

    The algorithm is completely independent of the Darcy solver.
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
    for b in bodies:
        bodies_by_type.setdefault(b.body_type, []).append(b)

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

    # Discover shared faces between gating components and the part.
    max_id = int(comp_id.max())
    mult = max_id + 1
    n_keys = mult * mult
    area_b = np.zeros(n_keys, dtype=np.float64)
    cx_b = np.zeros(n_keys, dtype=np.float64)
    cy_b = np.zeros(n_keys, dtype=np.float64)
    cz_b = np.zeros(n_keys, dtype=np.float64)

    nx_b = np.zeros(n_keys, dtype=np.float64)
    ny_b = np.zeros(n_keys, dtype=np.float64)
    nz_b = np.zeros(n_keys, dtype=np.float64)

    def _accumulate_contact_1d(
        id_a_1d: np.ndarray,
        id_b_1d: np.ndarray,
        x_1d: np.ndarray,
        y_1d: np.ndarray,
        z_1d: np.ndarray,
        s_x: float,
        s_y: float,
        s_z: float,
    ) -> None:
        nonlocal area_b, cx_b, cy_b, cz_b, nx_b, ny_b, nz_b
        if id_a_1d.size == 0:
            return
        up = np.minimum(id_a_1d, id_b_1d)
        down = np.maximum(id_a_1d, id_b_1d)
        key = up * mult + down
        area_b += np.bincount(key, minlength=n_keys) * area_face
        cx_b += np.bincount(key, weights=x_1d * area_face, minlength=n_keys)
        cy_b += np.bincount(key, weights=y_1d * area_face, minlength=n_keys)
        cz_b += np.bincount(key, weights=z_1d * area_face, minlength=n_keys)
        # The geometric face normal does not depend on id ordering; the sign is
        # irrelevant because the section plane is the same for +/-n.
        w_norm = np.ones(id_a_1d.shape[0], dtype=np.float64) * area_face
        nx_b += np.bincount(key, weights=s_x * w_norm, minlength=n_keys)
        ny_b += np.bincount(key, weights=s_y * w_norm, minlength=n_keys)
        nz_b += np.bincount(key, weights=s_z * w_norm, minlength=n_keys)

    # x-faces
    id_a = comp_id[:-1, :, :]
    id_b = comp_id[1:, :, :]
    valid = (id_a != 0) & (id_b != 0) & (id_a != id_b)
    if valid.any():
        i, j, k = np.where(valid)
        _accumulate_contact_1d(
            id_a[valid],
            id_b[valid],
            origin_mm[0] + (i + 1) * dx_mm,
            origin_mm[1] + (j + 0.5) * dx_mm,
            origin_mm[2] + (k + 0.5) * dx_mm,
            1.0, 0.0, 0.0,
        )

    # y-faces
    id_a = comp_id[:, :-1, :]
    id_b = comp_id[:, 1:, :]
    valid = (id_a != 0) & (id_b != 0) & (id_a != id_b)
    if valid.any():
        i, j, k = np.where(valid)
        _accumulate_contact_1d(
            id_a[valid],
            id_b[valid],
            origin_mm[0] + (i + 0.5) * dx_mm,
            origin_mm[1] + (j + 1) * dx_mm,
            origin_mm[2] + (k + 0.5) * dx_mm,
            0.0, 1.0, 0.0,
        )

    # z-faces
    id_a = comp_id[:, :, :-1]
    id_b = comp_id[:, :, 1:]
    valid = (id_a != 0) & (id_b != 0) & (id_a != id_b)
    if valid.any():
        i, j, k = np.where(valid)
        _accumulate_contact_1d(
            id_a[valid],
            id_b[valid],
            origin_mm[0] + (i + 0.5) * dx_mm,
            origin_mm[1] + (j + 0.5) * dx_mm,
            origin_mm[2] + (k + 1) * dx_mm,
            0.0, 0.0, 1.0,
        )

    contacts: List[Dict] = []
    for key in np.nonzero(area_b)[0]:
        id1 = int(key // mult)
        id2 = int(key % mult)
        if id1 == 0 or id2 == 0 or id1 == id2:
            continue
        btype1, name1 = comp_meta.get(id1, (BodyType.EMPTY, f"UNKNOWN_{id1}"))
        btype2, name2 = comp_meta.get(id2, (BodyType.EMPTY, f"UNKNOWN_{id2}"))
        area_m2 = float(area_b[key])
        if area_m2 <= 1e-18:
            continue
        centroid = np.array(
            [cx_b[key] / area_b[key], cy_b[key] / area_b[key], cz_b[key] / area_b[key]],
            dtype=np.float64,
        )
        nvec = np.array([abs(nx_b[key]), abs(ny_b[key]), abs(nz_b[key])], dtype=np.float64)
        n_norm = float(np.linalg.norm(nvec))
        if n_norm > 1e-18:
            contact_normal = nvec / n_norm
        else:
            contact_normal = -g_u
        contacts.append(
            {
                "id1": id1,
                "id2": id2,
                "type1": btype1,
                "type2": btype2,
                "name1": name1,
                "name2": name2,
                "voxel_area_m2": area_m2,
                "centroid_mm": centroid,
                "normal": contact_normal,
            }
        )

    if not contacts:
        return []

    # Source selection: prefer the component selected by the user.
    source_candidates = [
        cid for cid, (bt, _) in comp_meta.items()
        if bt in {BodyType.POURING_BASIN, BodyType.SPRUE_THROAT, BodyType.SPRUE} and cid != part_id
    ]

    def _upstream_rank(cid: int) -> float:
        return float(-np.dot(comp_centroids[cid], g_u))

    if source_candidates:
        source_id = max(source_candidates, key=_upstream_rank)
    else:
        source_id = max((cid for cid in comp_meta if cid != part_id), key=_upstream_rank)

    # Build undirected adjacency and orient it by BFS from the source.
    adj: Dict[int, List[Dict]] = {cid: [] for cid in comp_meta if cid != part_id}
    for c in contacts:
        adj.setdefault(c["id1"], []).append(c)
        adj.setdefault(c["id2"], []).append(c)

    parent: Dict[int, int] = {source_id: -1}
    outgoing: Dict[int, List[Dict]] = {cid: [] for cid in comp_meta if cid != part_id}
    queue = [source_id]
    visited = {source_id}
    order = [source_id]

    while queue:
        current = queue.pop(0)
        for c in adj.get(current, []):
            other = c["id1"] if c["id2"] == current else c["id2"]
            if other == part_id:
                outgoing[current].append(c)
                c["up_id"] = current
                c["down_id"] = part_id
                continue
            if other in visited:
                continue
            visited.add(other)
            parent[other] = current
            outgoing[current].append(c)
            c["up_id"] = current
            c["down_id"] = other
            queue.append(other)
            order.append(other)

    # Component throat areas will be filled after the contact-based mesh
    # sections are computed below.  Initialize with inf (part) and zero.
    comp_section_m2: Dict[int, Tuple[float, float, float]] = {
        part_id: (float("inf"), float("inf"), float("inf"))
    }
    for cid in comp_cells:
        if cid != part_id:
            comp_section_m2[cid] = (0.0, 0.0, 0.0)



    def _mesh_section_area_m2(mesh, plane_origin, plane_normal) -> float:
        """Return the throat area (m²) of a mesh near a contact plane.

        The plane is oriented perpendicular to the component's flow axis and we
        sweep a short distance around the contact to find the largest stable
        polygon area.  The largest avoids the tiny slivers produced when a plane
        just clips the body, while a small L prevents moving into an expanding
        tapered section.
        """
        if mesh is None or len(mesh.faces) == 0:
            return 0.0
        try:
            plane_origin = np.asarray(plane_origin, dtype=np.float64)
            n = np.asarray(plane_normal, dtype=np.float64)
            if np.linalg.norm(n) < 1e-18:
                n = -g_u
            n = n / np.linalg.norm(n)
            L = max(dx_mm * 0.5, 0.05)
            steps = np.concatenate([
                np.linspace(-L, -0.1 * L, 4),
                [0.0],
                np.linspace(0.1 * L, L, 4),
            ])
            best = 0.0
            for s in steps:
                section = mesh.section(plane_origin=plane_origin + s * n, plane_normal=n)
                if section is None:
                    continue
                path2d = section.to_2D()
                if isinstance(path2d, tuple):
                    path2d = path2d[0]
                area_mm2 = float(path2d.area)
                if area_mm2 > 1e-12:
                    best = max(best, area_mm2)
            return best * 1e-6
        except Exception:
            return 0.0

    # Compute the real throat/contact area for each contact from the CAD mesh.
    # Each contact area is the minimum of the two body cross-sections taken
    # perpendicular to the contact normal.  We also record the smallest such
    # cross-section for each component, which is its throat for design scaling.
    for c in contacts:
        id1, id2 = c["id1"], c["id2"]
        body1 = comp_body.get(id1)
        body2 = comp_body.get(id2)
        plane_origin = c["centroid_mm"]
        normal = c.get("normal", -g_u)

        a1 = _mesh_section_area_m2(body1.mesh if body1 else None, plane_origin, normal)
        a2 = _mesh_section_area_m2(body2.mesh if body2 else None, plane_origin, normal)
        if a1 > 1e-18 and a2 > 1e-18:
            a_mesh = float(min(a1, a2))
        elif a1 > 1e-18 or a2 > 1e-18:
            a_mesh = float(max(a1, a2))
        else:
            a_mesh = 0.0

        # Update component throat (minimum self-section at any contact).
        for cid, a_self in ((id1, a1), (id2, a2)):
            if cid == part_id:
                continue
            if a_self > 1e-18:
                old = comp_section_m2.get(cid, (0.0, 0.0, 0.0))
                new_throat = min(old[2], a_self) if old[2] > 1e-18 else a_self
                comp_section_m2[cid] = (new_throat, new_throat, new_throat)

        # Voxel shared-face area is a robust fallback when the mesh section fails.
        voxel_area_m2 = float(c.get("voxel_area_m2", 0.0))
        a_contact = max(a_mesh, voxel_area_m2)

        # The design caps are applied after comp_design_m2 is recomputed below,
        # so we just store the raw physical contact area for now.
        c["area_raw_m2"] = float(a_contact)

    # Design caps: source area, and total INGATE area split by throat.
    section_key_to_type = {
        "SPRUE_THROAT": BodyType.SPRUE,
        "SPRUE_BASE": BodyType.SPRUE,
        "RUNNER": BodyType.RUNNER,
        "DISTRIBUTOR": BodyType.DISTRIBUTOR,
        "CURUFLUK": BodyType.CURUFLUK,
        "FILTER": BodyType.FILTER,
        "POURING_BASIN": BodyType.POURING_BASIN,
        "RISER": BodyType.RISER,
        "INGATE": BodyType.INGATE,
    }
    type_design_total: Dict[BodyType, float] = {}
    if section_areas_m2:
        for key, val in section_areas_m2.items():
            btype = section_key_to_type.get(key.upper())
            if btype is not None and val > 1e-18:
                type_design_total[btype] = type_design_total.get(btype, 0.0) + float(val)

    raw_throat_per_type: Dict[BodyType, float] = {}
    for cid, (_, _, throat) in comp_section_m2.items():
        if cid == part_id:
            continue
        btype = comp_meta[cid][0]
        if throat > 1e-18:
            raw_throat_per_type[btype] = raw_throat_per_type.get(btype, 0.0) + float(throat)

    comp_design_m2: Dict[int, float] = {}
    ingate_design_total = type_design_total.get(BodyType.INGATE, 0.0)
    ingate_raw_total = raw_throat_per_type.get(BodyType.INGATE, 0.0)
    for cid in comp_meta:
        if cid == part_id:
            continue
        btype = comp_meta[cid][0]
        if cid == source_id:
            comp_design_m2[cid] = float(source_area_m2)
        elif btype == BodyType.INGATE and ingate_design_total > 1e-18 and ingate_raw_total > 1e-18:
            raw_throat = comp_section_m2[cid][2]
            comp_design_m2[cid] = ingate_design_total * (raw_throat / ingate_raw_total)
        else:
            comp_design_m2[cid] = float("inf")

    # Apply design caps to the raw contact areas.
    for c in contacts:
        id1, id2 = c["id1"], c["id2"]
        a_contact = c["area_raw_m2"]
        design1 = comp_design_m2.get(id1, float("inf"))
        design2 = comp_design_m2.get(id2, float("inf"))
        if a_contact > 1e-18:
            c["area_m2"] = float(min(a_contact, design1, design2))
        else:
            c["area_m2"] = float(min(design1, design2))

    # Propagate Q and compute velocities.
    Q_in: Dict[int, float] = {cid: 0.0 for cid in comp_meta if cid != part_id}
    Q_in[source_id] = float(Q_user)

    def _node_name(up_id: int, down_id: int) -> str:
        return f"{comp_meta[up_id][1]} → {comp_meta[down_id][1]}"

    def _node_body_type(up_id: int, down_id: int) -> str:
        return f"{comp_meta[up_id][0].name}→{comp_meta[down_id][0].name}"

    def _make_node(up_id: int, down_id: int, area_m2: float, Q: float, centroid: np.ndarray) -> GatingNode:
        v = float(Q / area_m2) if area_m2 > 1e-18 else 0.0
        return GatingNode(
            name=_node_name(up_id, down_id),
            body_type=_node_body_type(up_id, down_id),
            velocity_m_s=v,
            section_area_cm2=float(area_m2 * 1e4),
            centroid_mm=tuple(float(x) for x in centroid),
        )

    nodes: List[GatingNode] = []

    # Source inlet node (user velocity at the source throat).
    if source_area_m2 > 1e-18:
        src_type, src_name = comp_meta[source_id]
        source_centroid = comp_centroids[source_id] - 5.0 * g_u
        nodes.append(
            GatingNode(
                name=f"Kaynak → {src_name}",
                body_type=f"SOURCE→{source_section_key}",
                velocity_m_s=float(Q_user / source_area_m2),
                section_area_cm2=float(source_area_m2 * 1e4),
                centroid_mm=tuple(float(x) for x in source_centroid),
            )
        )

    for cid in order:
        Q = Q_in[cid]
        out_edges = outgoing.get(cid, [])
        if not out_edges:
            continue
        A_total = sum(c["area_m2"] for c in out_edges)
        if A_total <= 1e-18:
            continue
        for c in out_edges:
            A = c["area_m2"]
            Q_branch = Q * (A / A_total)
            c["Q_branch"] = Q_branch
            c["v_branch"] = Q_branch / A if A > 1e-18 else 0.0
            down_id = c["down_id"]
            if down_id != part_id:
                Q_in[down_id] += Q_branch
            nodes.append(_make_node(cid, down_id, A, Q_branch, c["centroid_mm"]))

    # Sort so the report follows the fill path (source first).
    rank_by_name = {comp_meta[cid][1]: _upstream_rank(cid) for cid in comp_meta if cid != part_id}
    nodes.sort(key=lambda n: (-rank_by_name.get(n.name.split(" → ")[0], 0.0), n.body_type))
    return nodes


def solve_filling_flow(
    grid: np.ndarray,
    origin: np.ndarray,
    dx: float,
    casting_params,
    alloy,
    bodies=None,
    max_solver_cells: int = 1_000_000,
    progress_callback=None,
    design_velocity_m_s: float = 0.0,
    design_section_key: str = "SPRUE_THROAT",
    design_area_m2: float = 0.0,
    section_areas_m2: Optional[Dict[str, float]] = None,
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

    Returns
    -------
    FillingResult
    """
    if progress_callback:
        progress_callback(2)

    # Downsample to keep the linear solve tractable.
    grid_c, origin_c, dx_c = _downsample_grid(grid, origin, dx, max_solver_cells)
    dx_m = dx_c / 1000.0

    g = _gravity_unit(getattr(casting_params, "gravity_vector", (0.0, 0.0, -1.0)))
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

    # Identify inlet (always the sprue/pouring-basin top) and vent (top of part/riser).
    section_key = getattr(casting_params, "velocity_section_key", "SPRUE")
    inlet_cells, inlet_name = _select_inlet_cells(grid_c, cavity, g, "SPRUE")
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

    # Pressure solve.
    A, rhs, flat_idx = _build_laplace_matrix(cavity, dirichlet, dirichlet_value)
    if progress_callback:
        progress_callback(35)

    p_flat = _solve_pressure(A, rhs)
    p = np.zeros(cavity.shape, dtype=np.float64)
    p[cavity] = p_flat

    if progress_callback:
        progress_callback(55)

    # Face velocities (Darcy: u = -K/μ * dp/dx).  K/μ is irrelevant because we
    # scale to the user-specified flow rate, so we set it to 1 here.
    u, v, w = _face_velocities(p, cavity, dx_m)

    # Determine user flow rate.
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
    Q_raw = _inlet_flux_m3_s(u, v, w, inlet_cells, cavity, dx_m)
    if abs(Q_raw) < 1e-18:
        # Degenerate geometry (e.g. disconnected inlet).  Fall back to area average.
        Q_raw = float(np.maximum(np.abs(u).sum(), 1e-18)) * (dx_m * dx_m)
    scale = Q_user / Q_raw

    u *= scale
    v *= scale
    w *= scale

    vmag = _cell_velocity_magnitude(u, v, w)
    vmag = np.nan_to_num(vmag, posinf=0.0, neginf=0.0)

    if progress_callback:
        progress_callback(75)

    # Total fill time estimate: part volume / user flow rate.
    fill_time_s = part_volume_m3 / Q_user if Q_user > 1e-18 else 0.0

    # Per-voxel front arrival time from the inlet.
    fill_time_c = _compute_fill_time(vmag, cavity, inlet_cells, dx_m)
    # Scale the raw fast-marching times so the last filled voxel equals the
    # robust volume/Q fill time; local velocities can otherwise give unrealistically
    # large times in stagnant pockets.
    finite_fill = fill_time_c[np.isfinite(fill_time_c) & cavity]
    if finite_fill.size > 0 and fill_time_s > 0.0:
        raw_max = float(finite_fill.max())
        if raw_max > 0.0:
            fill_time_c = np.where(
                cavity & np.isfinite(fill_time_c),
                fill_time_c * (fill_time_s / raw_max),
                fill_time_c,
            )
    if fill_time_c.shape == grid.shape:
        fill_time_fine = fill_time_c
    else:
        fill_time_fine = ndimage.zoom(
            fill_time_c,
            (
                grid.shape[0] / fill_time_c.shape[0],
                grid.shape[1] / fill_time_c.shape[1],
                grid.shape[2] / fill_time_c.shape[2],
            ),
            order=1,
        )
        fill_time_fine = np.where(grid > 0, fill_time_fine, 0.0)

    # Upsample velocity magnitude to the (fine) input grid for surface overlay
    # and per-gate velocity calculation.
    if vmag.shape == grid.shape:
        vmag_fine = vmag
    else:
        vmag_fine = ndimage.zoom(
            vmag,
            (
                grid.shape[0] / vmag.shape[0],
                grid.shape[1] / vmag.shape[1],
                grid.shape[2] / vmag.shape[2],
            ),
            order=1,
        )
        vmag_fine = np.where(grid > 0, vmag_fine, 0.0)

    # Contact-node velocities / areas for every gating-gating and gating-part interface.
    # Uses a pure hydraulic Q = vA network on the CAD mesh cross-sections; Darcy is
    # bypassed for node velocities (it is still computed for fill_time/velocity overlay).
    gating_nodes = _gating_node_velocities(
        grid,
        origin,
        dx,
        Q_user,
        area_m2,
        used_section,
        g,
        bodies,
        section_areas_m2=section_areas_m2,
    )
    # Collect every node that feeds the part directly as a "gate" (meme).
    per_gate_v = {}
    per_gate_area = {}
    for n in gating_nodes:
        parts = n.body_type.split("→")
        if len(parts) != 2:
            continue
        up, down = parts
        if down in ("PART", "Parça") and up != "SOURCE":
            gate_name = n.name.split(" → ")[0]
            per_gate_v[gate_name] = n.velocity_m_s
            per_gate_area[gate_name] = n.section_area_cm2

    # Aggregate section velocities and ingate contact velocity directly from
    # the computed contact nodes so the report/viewer match the 3-D labels.
    node_v, v_ingate_contact = _aggregate_section_velocities(gating_nodes, Q_user, area_m2)
    if per_gate_v:
        v_ingate_contact = float(np.mean(list(per_gate_v.values())))

    if progress_callback:
        progress_callback(95)

    reason = (
        f"Darcy akış çözümü: giriş '{used_section}', Q={Q_user*6e4:.2f} L/dak, "
        f"girdi hızı/alan={user_velocity:.3f} m/s / {area_m2*1e4:.2f} cm², "
        f"tahmini doldurma süresi={fill_time_s:.2f} s."
    )

    return FillingResult(
        node_velocities=node_v,
        ingate_contact_velocity_m_s=v_ingate_contact,
        Q_m3_s=Q_user,
        inlet_area_m2=area_m2,
        fill_time_s=fill_time_s,
        velocity_magnitude=vmag_fine,
        fill_time=fill_time_fine,
        solver_grid=grid_c,
        solver_dx_mm=dx_c,
        pressure=p,
        reason=reason,
        per_gate_contact_velocity_m_s=per_gate_v,
        per_gate_contact_area_cm2=per_gate_area,
        gating_nodes=gating_nodes,
    )
