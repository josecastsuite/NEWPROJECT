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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse import linalg as spla

from core.types import BodyType, FillingResult


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
    if key in ("INGATE", "SPRUE_BASE", "SPRUE_THROAT"):
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

    if velocity_m_s > 0.0:
        face, used_section = _section_face_cells(a_grid, a_cavity, section_key, g)
    elif fill_time_s > 0.0 and part_volume_m3 > 0.0:
        face, used_section = _section_face_cells(a_grid, a_cavity, section_key, g)
    elif design_velocity_m_s > 0.0:
        face, used_section = _section_face_cells(a_grid, a_cavity, design_section_key, g)
    else:
        face, used_section = _section_face_cells(a_grid, a_cavity, section_key, g)

    area_m2 = float(face.sum()) * (a_dx * a_dx)
    if velocity_m_s > 0.0 and area_m2 > 0.0:
        Q = velocity_m_s * area_m2
        return Q, area_m2, used_section

    if fill_time_s > 0.0 and part_volume_m3 > 0.0:
        Q = part_volume_m3 / fill_time_s
        return Q, area_m2, used_section

    if design_velocity_m_s > 0.0:
        # Use the design reference area (e.g. choke area from the gating engine)
        # instead of the raw voxel area, so Q is consistent with the design.
        Q_area_m2 = design_area_m2 if design_area_m2 > 1e-18 else area_m2
        Q = design_velocity_m_s * Q_area_m2
        return Q, Q_area_m2, used_section

    # Last resort: a tiny flow to allow a solve; caller will report no user input.
    if area_m2 > 0.0:
        Q = 0.01 * area_m2
    else:
        Q = 1.0
    return Q, area_m2, used_section


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
        "SPRUE",
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

    # Node / section velocities and ingate contact velocity.
    # Use continuity Q / cross-sectional area; the pressure field is kept for
    # visualization and future transient filling front work.
    node_v = _node_velocities(
        grid_c,
        cavity,
        g,
        dx_m,
        Q_user,
        fine_grid=fine_grid,
        fine_cavity=fine_cavity,
        fine_dx_m=fine_dx_m,
        section_areas_m2=section_areas_m2,
    )
    v_ingate_contact = _ingate_contact_velocity(
        grid_c,
        cavity,
        dx_m,
        g,
        Q_user,
        fine_grid=fine_grid,
        fine_cavity=fine_cavity,
        fine_dx_m=fine_dx_m,
        section_areas_m2=section_areas_m2,
    )

    # Total fill time estimate: part volume / user flow rate.
    fill_time_s = part_volume_m3 / Q_user if Q_user > 1e-18 else 0.0

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
        velocity_magnitude=vmag,
        solver_grid=grid_c,
        solver_dx_mm=dx_c,
        pressure=p,
        reason=reason,
    )
