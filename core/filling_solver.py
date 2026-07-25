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
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pyvista as pv
import trimesh
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse import linalg as spla

from core.types import Body, BodyType, FillingResult, GatingNode, GatingVelocityError
from core.voxelizer import build_voxel_grid, compute_face_fractions


def _downsample_grid(
    grid: np.ndarray,
    origin: np.ndarray,
    dx: float,
    max_cells: int = 6_000_000,
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


def _flow_refined_grid(
    bodies: List[Body],
    casting_params,
    desired_dx_mm: float = 1.75,
    max_cells: int = 6_000_000,
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
    # Total grid cells scale as target_dim^3; leave a 2x safety factor for
    # margin + empty space, and let _downsample_grid trim the cavity count.
    max_dim = int(np.floor((max_cells * 2.0) ** (1.0 / 3.0)))
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

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
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
            rows.extend(c.tolist())
            cols.extend(n.tolist())
            data.extend(wb.tolist())
            rows.extend(n.tolist())
            cols.extend(c.tolist())
            data.extend(wb.tolist())
            diag[c] += wb
            diag[n] += wb

        # cur non-Dirichlet, nb Dirichlet
        c_nd_nb_d = (~cdir) & ndir
        if c_nd_nb_d.any():
            c = cur_idx[c_nd_nb_d]
            wb = w[c_nd_nb_d]
            diag[c] += wb
            rhs[c] -= wb * nv[c_nd_nb_d]

        # cur Dirichlet, nb non-Dirichlet (the symmetric contribution from
        # the other side of the same face).
        c_d_nb_nd = cdir & (~ndir)
        if c_d_nb_nd.any():
            n = nb_idx[c_d_nb_nd]
            wb = w[c_d_nb_nd]
            diag[n] += wb
            rhs[n] -= wb * cv[c_d_nb_nd]

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
        rows.extend(unknown[dirichlet_unknowns].tolist())
        cols.extend(unknown[dirichlet_unknowns].tolist())
        data.extend(np.ones(dirichlet_unknowns.sum(), dtype=np.float64).tolist())
        rhs[dirichlet_unknowns] = dirichlet_value[cavity][dirichlet_unknowns]

    non_dirichlet = ~dirichlet_unknowns
    if non_dirichlet.any():
        rows.extend(unknown[non_dirichlet].tolist())
        cols.extend(unknown[non_dirichlet].tolist())
        data.extend((-diag[non_dirichlet]).tolist())

    A = csr_matrix((data, (rows, cols)), shape=(n_unknowns, n_unknowns))
    return A, rhs, flat_idx, dirichlet_unknowns, dirichlet_value[cavity]


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
    """Return the geometric throat cross-sectional area (m²) of a body.

    The plane passes through ``plane_origin`` and is perpendicular to the
    local flow direction ``plane_normal``.  This gives the real channel
    cross-section used in ``v = Q / A`` for a node, independent of the voxel
    grid resolution.
    """
    if mesh is None or len(mesh.faces) == 0:
        return None
    origin = np.asarray(plane_origin, dtype=np.float64)
    normal = np.asarray(plane_normal, dtype=np.float64)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-18)

    def _area_at(o: np.ndarray) -> Optional[float]:
        try:
            section = mesh.section(
                plane_origin=o,
                plane_normal=normal,
            )
            if section is None or getattr(section, "is_empty", True):
                return None
            p2d = section.to_2D()
            if isinstance(p2d, tuple):
                p2d = p2d[0]
            if p2d is None or not getattr(p2d, "polygons_closed", None):
                return None
            return float(sum(float(p.area) for p in p2d.polygons_closed))
        except Exception:
            return None

    # Try the requested origin and a small step on either side.  This handles
    # cases where the contact centroid sits exactly on a sharp edge and the
    # initial offset pointed outside the body.
    eps = 0.1
    for o in (origin, origin + normal * eps, origin - normal * eps):
        area_mm2 = _area_at(o)
        if area_mm2 is not None and area_mm2 > 1e-9:
            return area_mm2 * 1e-6
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
    # Projected Darcy flux per unordered component pair is computed after BFS,
    # once the true upstream/downstream direction of each contact is known.

    def _accumulate_contact_1d(
        id_a_1d: np.ndarray,
        id_b_1d: np.ndarray,
        x_1d: np.ndarray,
        y_1d: np.ndarray,
        z_1d: np.ndarray,
        s_x: float,
        s_y: float,
        s_z: float,
        f_A_1d: np.ndarray,
    ) -> None:
        nonlocal area_b, cx_b, cy_b, cz_b, nx_b, ny_b, nz_b
        if id_a_1d.size == 0:
            return
        up = np.minimum(id_a_1d, id_b_1d)
        down = np.maximum(id_a_1d, id_b_1d)
        key = up * mult + down
        w = area_face * f_A_1d
        area_b += np.bincount(key, weights=w, minlength=n_keys)
        cx_b += np.bincount(key, weights=x_1d * w, minlength=n_keys)
        cy_b += np.bincount(key, weights=y_1d * w, minlength=n_keys)
        cz_b += np.bincount(key, weights=z_1d * w, minlength=n_keys)
        # Signed normal weighted by the real fractional area.  The sign is fixed
        # by the +axis orientation and is oriented downstream later.
        nx_b += np.bincount(key, weights=s_x * w, minlength=n_keys)
        ny_b += np.bincount(key, weights=s_y * w, minlength=n_keys)
        nz_b += np.bincount(key, weights=s_z * w, minlength=n_keys)

    # Only orthogonal face neighbours carry hydraulic flow; diagonal/edge touches
    # do not create a real flow path and are ignored.
    directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    unique_dirs = directions

    for di, dj, dk in unique_dirs:
        # Slices for id_a (lower-index cell) and id_b (higher-index cell).
        sa0, sb0 = slice(0, -1), slice(1, None)
        sa1, sb1 = slice(0, -1), slice(1, None)
        sa2, sb2 = slice(0, -1), slice(1, None)
        if di == 1:
            pass
        elif dj == 1:
            sa0, sb0 = slice(None), slice(None)
            sa1, sb1 = slice(0, -1), slice(1, None)
            sa2, sb2 = slice(None), slice(None)
        else:  # dk == 1
            sa0, sb0 = slice(None), slice(None)
            sa1, sb1 = slice(None), slice(None)
            sa2, sb2 = slice(0, -1), slice(1, None)

        id_a = comp_id[sa0, sa1, sa2]
        id_b = comp_id[sb0, sb1, sb2]
        valid = (id_a != 0) & (id_b != 0) & (id_a != id_b)
        if not valid.any():
            continue

        s_x, s_y, s_z = float(di), float(dj), float(dk)

        i, j, k = np.where(valid)
        start_i = sa0.start if sa0.start is not None else 0
        start_j = sa1.start if sa1.start is not None else 0
        start_k = sa2.start if sa2.start is not None else 0
        gi_i = (start_i + i).astype(np.int64)
        gj_j = (start_j + j).astype(np.int64)
        gk_k = (start_k + k).astype(np.int64)

        # Face centre coordinates and the matching FAVOR face index.
        gi = gi_i.astype(np.float64) + 0.5 + di * 0.5
        gj = gj_j.astype(np.float64) + 0.5 + dj * 0.5
        gk = gk_k.astype(np.float64) + 0.5 + dk * 0.5

        if di == 1:
            f_A_face = f_A_z[gi_i + 1, gj_j, gk_k]
        elif dj == 1:
            f_A_face = f_A_y[gi_i, gj_j + 1, gk_k]
        else:
            f_A_face = f_A_x[gi_i, gj_j, gk_k + 1]

        _accumulate_contact_1d(
            id_a[valid],
            id_b[valid],
            origin_mm[0] + gi * dx_mm,
            origin_mm[1] + gj * dx_mm,
            origin_mm[2] + gk * dx_mm,
            s_x,
            s_y,
            s_z,
            f_A_face,
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
        nvec = np.array([nx_b[key], ny_b[key], nz_b[key]], dtype=np.float64)
        n_norm = float(np.linalg.norm(nvec))
        if n_norm > 1e-18:
            contact_normal = nvec / n_norm
        else:
            contact_normal = -g_u
        # Darcy flux and flow-axis orientation are recomputed below after BFS.
        flux_12 = 0.0
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
                "flux_m3_s": flux_12,
            }
        )

    if not contacts:
        raise GatingVelocityError(
            "Düğüm hızları çözülemedi: döküm sisteminde hiç temas yüzeyi bulunamadı. "
            "Hız kesitleri parça/geometri nedeniyle hesaplanamadı."
        )

    # Source selection: use the BodyType selected by the user's velocity_section_key.
    # Gravity is used only as a tie-breaker when multiple components of the
    # selected type exist.
    source_section = (source_section_key or "SPRUE_THROAT").upper()
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
        if bt in allowed_source_types and cid != part_id
    ]
    if not source_candidates:
        source_candidates = [
            cid for cid, (bt, _) in comp_meta.items()
            if bt in {BodyType.POURING_BASIN, BodyType.SPRUE_THROAT, BodyType.SPRUE}
            and cid != part_id
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
        source_id = max(gating_ids, key=_upstream_rank)

    if source_id not in comp_meta or source_id == part_id:
        raise GatingVelocityError(
            "Düğüm hızları çözülemedi: kaynak (sprue/döküm ağzı) seçilemedi. "
            "Hız kesitleri parça/geometri nedeniyle hesaplanamadı."
        )
    if source_area_m2 <= 1e-18 or Q_user <= 1e-18:
        raise GatingVelocityError(
            "Düğüm hızları çözülemedi: giriş debisi/hızı tanımlanamadı. "
            "Hız kesitleri parça/geometri nedeniyle hesaplanamadı."
        )

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

    unvisited = [
        cid for cid in gating_ids
        if cid not in visited and comp_meta[cid][0] != BodyType.RISER
    ]
    if unvisited:
        names = ", ".join(comp_meta[cid][1] for cid in unvisited)
        # Warn but continue: return partial gating nodes for the reachable
        # portion instead of crashing the whole analysis.
        import warnings
        warnings.warn(
            f"Düğüm hızları: şu elemanlar kaynaktan parçaya ulaşan zincire bağlı değil: {names}. "
            f"Yalnızca ulaşılabilir düğümler hesaplanacak.",
            RuntimeWarning,
        )

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

    # Assign the integrated Darcy flux and projected contact area to each contact.
    for c in contacts:
        up = c.get("up_id")
        down = c.get("down_id")
        if up is None or down is None:
            continue
        k = int(min(up, down)) * mult + int(max(up, down))
        c["flux_m3_s"] = float(flux_pair[k])
        c["area_real_m2"] = float(area_pair[k])
        # The final area is set below from the real CAD geometry; if that fails
        # the solver stops instead of silently falling back to an approximation.
        c["area_m2"] = float(area_pair[k])

    # ------------------------------------------------------------------
    # Replace the projected Darcy contact area with the real geometric throat
    # cross-section from the CAD mesh.  The plane is placed at the contact
    # centroid and oriented perpendicular to the local Darcy flow direction.
    # For a gate->part node the throat is the gate (upstream); for all other
    # gating->gating nodes it is the downstream channel.
    # No voxel/design fallback: if the geometry cannot be sectioned, we stop.
    # Contacts that carry no Darcy flux (dead branches) keep the Darcy area for
    # reporting and are skipped in the node list.
    # ------------------------------------------------------------------
    for c in contacts:
        up = c.get("up_id")
        down = c.get("down_id")
        if up is None or down is None:
            continue
        flux = float(c.get("flux_m3_s", 0.0))
        if abs(flux) < 1e-18:
            # Dead branch: velocity is zero, area is only for reporting.
            c["area_m2"] = float(max(c.get("area_real_m2", 0.0), 1e-12))
            c["skip_node"] = True
            continue

        k = int(min(up, down)) * mult + int(max(up, down))
        V = V_flow_pair[:, k]
        v_norm = float(np.linalg.norm(V))
        if v_norm > 1e-12:
            normal = V / v_norm
        else:
            # No Darcy direction: point from upstream centroid toward downstream.
            d = comp_centroids[down] - comp_centroids[up]
            d_norm = float(np.linalg.norm(d))
            if d_norm > 1e-12:
                normal = d / d_norm
            else:
                normal = -g_u

        if down == part_id:
            # Gate -> Part: the throat is the gate cross-section.
            body = comp_body.get(up)
        else:
            # All other transitions: throat is the downstream channel.
            body = comp_body.get(down)

        if body is None or body.mesh is None or len(body.mesh.faces) == 0:
            raise GatingVelocityError(
                f"Düğüm hızı için gövde mesh'i bulunamadı: "
                f"{comp_meta.get(up, (None, str(up)))[1]} -> "
                f"{comp_meta.get(down, (None, str(down)))[1]}."
            )

        centroid = np.asarray(c["centroid_mm"], dtype=np.float64)
        body_centroid = body.mesh.centroid
        to_body = body_centroid - centroid
        to_body_norm = float(np.linalg.norm(to_body))
        if to_body_norm > 1e-12:
            interior_dir = to_body / to_body_norm
        else:
            interior_dir = -normal

        # Try the obvious interior offset first.
        eps = max(0.05, dx_mm * 0.05)
        origin = centroid + interior_dir * eps
        a_geo = _body_throat_area_m2(body.mesh, origin, normal)

        # If the contact centroid sits on the far side of the shared surface,
        # the simple offset can miss the body.  Use PyVista to find the actual
        # entry point into the body interior and retry.
        if a_geo is None or a_geo <= 1e-18:
            mesh_pv = _mesh_surface_pv(body.mesh)
            if mesh_pv is not None:
                robust_origin = _origin_inside_body(
                    mesh_pv, centroid, interior_dir, body_centroid, eps
                )
                if robust_origin is not None:
                    a_geo = _body_throat_area_m2(body.mesh, robust_origin, normal)

        if a_geo is None or a_geo <= 1e-18:
            up_name = comp_meta.get(up, (None, str(up)))[1]
            down_name = comp_meta.get(down, (None, str(down)))[1]
            raise GatingVelocityError(
                f"Düğüm hızı için geometrik boğaz alanı hesaplanamadı: "
                f"{up_name} -> {down_name}. CAD geometrisi kapalı/watertight olmayabilir "
                f"veya temas noktası dejeneredir."
            )
        c["area_m2"] = float(a_geo)
        c["area_geo_m2"] = float(a_geo)
        c["skip_node"] = False

    # Propagate Q and compute velocities.
    Q_in: Dict[int, float] = {cid: 0.0 for cid in comp_meta if cid != part_id}
    Q_in[source_id] = float(Q_user)

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

    part_edges: List[Dict] = []

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
        # Split the incoming flow among the outgoing branches using the real
        # 3B Darcy flux integrated across each contact (flux_m3_s).  The flux
        # proportions come directly from the solved velocity field; the total is
        # then scaled to enforce continuity at the node.  Each branch uses its
        # own contact area, so gates with different cross-sections keep their
        # distinct velocity values.
        raw_fluxes = np.array(
            [abs(float(c.get("flux_m3_s", 0.0))) for c in out_edges],
            dtype=np.float64,
        )
        raw_total = float(raw_fluxes.sum())
        areas = np.array([float(c["area_m2"]) for c in out_edges], dtype=np.float64)
        A_total = float(areas.sum())

        if raw_total > 1e-18:
            Q_branches = raw_fluxes * (Q / raw_total)
        else:
            # No Darcy flux measured on active outlets: zero flow for all.
            Q_branches = np.zeros_like(raw_fluxes)

        # Each outlet keeps its own Darcy-derived flow share.  No automatic
        # equalisation across same-type outlets is applied; v = Q_i / A_i.
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
            if down_id == part_id:
                # Postpone INGATE->PART nodes until a second pass that equalises
                # the exit velocity of all ingates fed by the same distributor.
                part_edges.append({
                    "cid": cid,
                    "c": c,
                    "A": A,
                    "Q_branch": Q_branch,
                    "v_branch": v_branch,
                })
                continue
            Q_in[down_id] += Q_branch
            print(
                f"[GATING_NODE] {comp_meta[cid][1]} -> {comp_meta[down_id][1]}  "
                f"area_cm2={A*1e4:.4f}  Q_L_s={Q_branch*1e3:.4f}  v_m_s={v_branch:.4f}",
                flush=True,
            )
            node = _make_node(cid, down_id, A, Q_branch, c["centroid_mm"], flow_rate_m3_s=Q_branch)
            nodes.append(node)

    # INGATE -> PART nodes are emitted with their own Darcy-derived flow share.
    # No common exit velocity is forced; v = Q_i / A_i for each gate.
    for e in part_edges:
        cid = e["cid"]
        c = e["c"]
        A = e["A"]
        Q_branch = float(e["Q_branch"])
        v_branch = float(e["v_branch"])
        c["Q_branch"] = Q_branch
        c["v_branch"] = v_branch
        down_id = c["down_id"]
        print(
            f"[GATING_NODE] {comp_meta[cid][1]} -> {comp_meta[down_id][1]}  "
            f"area_cm2={A*1e4:.4f}  Q_L_s={Q_branch*1e3:.4f}  v_m_s={v_branch:.4f}",
            flush=True,
        )
        node = _make_node(cid, down_id, A, Q_branch, c["centroid_mm"], flow_rate_m3_s=Q_branch)
        nodes.append(node)

    if not nodes:
        raise GatingVelocityError(
            "Düğüm hızları çözülemedi: hesaplanan düğüm listesi boş. "
            "Hız kesitleri parça/geometri nedeniyle hesaplanamadı."
        )

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
    return nodes


def solve_filling_flow(
    grid: np.ndarray,
    origin: np.ndarray,
    dx: float,
    casting_params,
    alloy,
    bodies=None,
    max_solver_cells: int = 6_000_000,
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

    # The input analysis grid is kept as the reference frame for the returned
    # velocity / fill_time fields (so they match result.grid downstream).
    orig_grid = grid.copy()
    orig_origin = origin.copy()
    orig_dx = float(dx)

    # If body geometry is available, build a flow-dedicated grid fine enough to
    # capture gate cross-sections (≤ ~1.8 mm) while staying within the solver
    # cavity budget.  Otherwise fall back to the supplied analysis grid.
    ref_grid, ref_origin, ref_dx = _flow_refined_grid(
        bodies, casting_params, desired_dx_mm=1.75, max_cells=max_solver_cells
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

    # Identify inlet (the user-selected velocity section) and vent (top of part/riser).
    section_key = getattr(casting_params, "velocity_section_key", "SPRUE")
    inlet_cells, inlet_name = _select_inlet_cells(grid_c, cavity, g, section_key)
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
    is_metal_c = grid_c != BodyType.EMPTY
    face_fractions = compute_face_fractions(is_metal_c, sub=4)
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
    # Resample the fill-time and velocity fields back onto the original analysis
    # grid so that downstream modules (e.g. flow animator) see consistent shapes.
    # Nearest-neighbour is used for fill_time because it carries an inf sentinel
    # for unreached cells; linear interpolation is used for velocity.
    if (
        fill_time_c.shape == orig_grid.shape
        and abs(dx_c - orig_dx) < 1e-9
        and np.allclose(origin_c, orig_origin)
    ):
        fill_time_fine = fill_time_c
    else:
        fill_time_fine = _resample_to_grid(
            fill_time_c,
            origin_c,
            dx_c,
            orig_grid.shape,
            orig_origin,
            orig_dx,
            fill_value=np.inf,
            order=0,
        )
        fill_time_fine = np.where(orig_grid == BodyType.EMPTY, 0.0, fill_time_fine)

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

    fine_metal = orig_grid > 0
    velocity = np.stack(
        [
            np.where(fine_metal, vx_f, 0.0),
            np.where(fine_metal, vy_f, 0.0),
            np.where(fine_metal, vz_f, 0.0),
        ],
        axis=0,
    ).astype(np.float32)
    vmag_fine = np.linalg.norm(velocity, axis=0)

    # Contact-node velocities / areas for every gating-gating and gating-part interface.
    # Use the solver (coarse) grid, the staggered face velocities and the FAVOR
    # fractional face areas so the real contact area is used in Q = v * A.
    gating_nodes = _gating_node_velocities(
        grid_c,
        origin_c,
        dx_c,
        Q_user,
        area_m2,
        used_section,
        g,
        bodies,
        section_areas_m2=section_areas_m2,
        u_m_s=u,
        v_m_s=v,
        w_m_s=w,
        dx_m=dx_c / 1000.0,
        face_fractions=face_fractions,
    )
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
            per_gate_v[gate_name] = n.velocity_m_s
            per_gate_area[gate_name] = n.section_area_cm2
            per_gate_q[gate_name] = n.flow_rate_m3_s
    total_ingate_flow_m3_s = float(sum(per_gate_q.values())) if per_gate_q else 0.0

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
        f"tahmini doldurma süresi={fill_time_s:.2f} s, "
        f"basınç düşümü={pressure_drop_pa:.1f} Pa."
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
    )
