"""3-D transient enthalpy-based thermal solver for JoseCast.

Uses an implicit finite-volume discretisation of

    dH/dt = ∇ · (k ∇T) - ρ·cp · (v · ∇T)

with H = ρ·cp·T + ρ·L·(1-fs) in the metal and H = ρ·cp·T in the mould.
The latent-heat contribution is regularised as an apparent heat capacity
within the mushy zone, giving a stable, second-order-in-space solution.

The optional velocity field ``velocity_m_s`` is the Darcy filling velocity
(hücre-bazlı 3B hız alanı).  It is treated explicitly as a convective
(advective) term and is only active in metal voxels whose fill time has
already elapsed, so the thermal solver is coupled to the flow solution.
"""


from typing import Optional, Tuple

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import linalg as spla

from core.materials import Alloy, MoldMaterial
from core.types import BODY_METAL_TYPES


def _scheil_fs(
    T: np.ndarray,
    t_liquidus: float,
    t_solidus: float,
    partition_coeff: float,
) -> np.ndarray:
    """Scheil solid fraction [0..1].

    fs = 1 - ((T - T_solidus) / (T_liquidus - T_solidus))^(1 / (1 - k))
    """
    fs = np.zeros_like(T)
    mask = (T <= t_liquidus) & (T >= t_solidus)
    denom = max(t_liquidus - t_solidus, 1.0)
    # u is the solidified fraction of the temperature interval, 0 at liquidus, 1 at solidus
    u = np.clip((t_liquidus - T[mask]) / denom, 0.0, 1.0)
    v = 1.0 - u
    exponent = 1.0 / max(1.0 - partition_coeff, 1e-6)
    fs[mask] = 1.0 - np.power(v, exponent)
    fs[T < t_solidus] = 1.0
    return np.clip(fs, 0.0, 1.0)


def _dscheil_dT(
    T: np.ndarray,
    t_liquidus: float,
    t_solidus: float,
    partition_coeff: float,
) -> np.ndarray:
    """Derivative of Scheil solid fraction w.r.t. temperature [K^-1]."""
    d = np.zeros_like(T)
    mask = (T < t_liquidus) & (T > t_solidus)
    if not np.any(mask):
        return d
    denom = max(t_liquidus - t_solidus, 1.0)
    u = (t_liquidus - T[mask]) / denom
    u = np.clip(u, 1e-9, 1.0 - 1e-9)
    v = 1.0 - u
    k = max(partition_coeff, 1e-6)
    p = 1.0 / (1.0 - k)
    d[mask] = p * np.power(v, p - 1.0) / denom
    return np.clip(d, 0.0, 1e6)


def _cp_eff(
    T: np.ndarray,
    is_metal: np.ndarray,
    alloy: Alloy,
    mold: MoldMaterial,
) -> np.ndarray:
    """Apparent heat capacity [J/(kg·K)] including latent heat."""
    cp = np.where(is_metal, alloy.cp_j_kgk, mold.cp_j_kgk).astype(np.float64)
    if alloy.latent_heat_j_kg > 0:
        dT_mush = max(alloy.t_liquidus_c - alloy.t_solidus_c, 1.0)
        df = _dscheil_dT(T, alloy.t_liquidus_c, alloy.t_solidus_c, alloy.partition_coefficient)
        # Cap the latent contribution so the total latent over the mush equals L
        cp[is_metal] += alloy.latent_heat_j_kg * np.clip(df[is_metal], 0.0, 1.0 / dT_mush)
    return cp


def _downsample_grid(grid: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbour downsample of a material-id grid."""
    if factor <= 1:
        return grid
    nx, ny, nz = grid.shape
    shape_c = (max(1, nx // factor), max(1, ny // factor), max(1, nz // factor))
    return ndimage.zoom(grid, (shape_c[0] / nx, shape_c[1] / ny, shape_c[2] / nz), order=0)


def _upsample(field_c: np.ndarray, target_shape: Tuple[int, int, int], order: int = 1) -> np.ndarray:
    """Upsample a scalar field to the original grid shape.

    Fields that may contain ``inf`` sentinel values (e.g. t_liq, t_sol,
    fill_time) are upsampled with ``order=0`` (nearest neighbour) so the
    sentinel does not leak into neighbouring cells.
    """
    if field_c.shape == target_shape:
        return field_c
    return ndimage.zoom(
        field_c,
        (target_shape[0] / field_c.shape[0], target_shape[1] / field_c.shape[1], target_shape[2] / field_c.shape[2]),
        order=order,
    )


def _build_laplacian(k: np.ndarray, dx: float) -> sparse.csc_matrix:
    """Build the symmetric finite-volume matrix A for ∇·(k∇T) with harmonic k at faces.

    Grid shape is (nx, ny, nz).  Raveled index = x*(ny*nz) + y*nz + z, so the
    stride in x is ny*nz, in y is nz, and in z is 1.
    """
    nx, ny, nz = k.shape
    n = nx * ny * nz
    inv_dx2 = 1.0 / (dx * dx)

    # Harmonic mean face conductivities: kx along x, ky along y, kz along z.
    kx = 2.0 * k[1:, :, :] * k[:-1, :, :] / (k[1:, :, :] + k[:-1, :, :] + 1e-12)
    ky = 2.0 * k[:, 1:, :] * k[:, :-1, :] / (k[:, 1:, :] + k[:, :-1, :] + 1e-12)
    kz = 2.0 * k[:, :, 1:] * k[:, :, :-1] / (k[:, :, 1:] + k[:, :, :-1] + 1e-12)

    # D arrays hold the face coefficient for the upper neighbor;
    # the lower neighbor is obtained by the same array with a negative offset.
    Dx = np.zeros((nx, ny, nz), dtype=np.float64)
    Dx[:-1, :, :] = kx
    Dy = np.zeros((nx, ny, nz), dtype=np.float64)
    Dy[:, :-1, :] = ky
    Dz = np.zeros((nx, ny, nz), dtype=np.float64)
    Dz[:, :, :-1] = kz

    # Degree (sum of incident face conductivities) for the main diagonal
    deg_x = np.zeros((nx, ny, nz), dtype=np.float64)
    deg_x[:-1, :, :] += kx
    deg_x[1:, :, :] += kx
    deg_y = np.zeros((nx, ny, nz), dtype=np.float64)
    deg_y[:, :-1, :] += ky
    deg_y[:, 1:, :] += ky
    deg_z = np.zeros((nx, ny, nz), dtype=np.float64)
    deg_z[:, :, :-1] += kz
    deg_z[:, :, 1:] += kz

    stride_x = ny * nz
    stride_y = nz
    diag_x = Dx.ravel() * inv_dx2
    diag_y = Dy.ravel() * inv_dx2
    diag_z = Dz.ravel() * inv_dx2
    main = -(deg_x + deg_y + deg_z).ravel() * inv_dx2

    A = sparse.diags(
        [diag_x, diag_x, diag_y, diag_y, diag_z, diag_z, main],
        offsets=[-stride_x, stride_x, -stride_y, stride_y, -1, 1, 0],
        shape=(n, n),
        format="csc",
    )
    return A


def _upwind_advection(
    T: np.ndarray,
    velocity: np.ndarray,
    dx: float,
    fill_c: Optional[np.ndarray],
    sub_t: float,
    is_metal: np.ndarray,
) -> np.ndarray:
    """First-order upwind explicit advection of temperature.

    ``velocity`` is (3, nx, ny, nz) with [x, y, z] components.  For each axis the
    derivative is taken from the upwind side, which is stable for CFL <= 1.  The
    result is masked to metal cells that have already been filled at time
    ``sub_t``.
    """
    if fill_c is not None:
        active = (sub_t >= fill_c) & is_metal
    else:
        active = is_metal
    if not np.any(active):
        return np.zeros_like(T)

    # Pad with edge values so outflow boundaries use a one-sided difference.
    Tz = np.pad(T, ((1, 1), (0, 0), (0, 0)), mode="edge")
    Ty = np.pad(T, ((0, 0), (1, 1), (0, 0)), mode="edge")
    Tx = np.pad(T, ((0, 0), (0, 0), (1, 1)), mode="edge")

    # z-direction
    dTdz = np.where(
        velocity[0] >= 0,
        (Tz[1:-1, :, :] - Tz[:-2, :, :]) / dx,
        (Tz[2:, :, :] - Tz[1:-1, :, :]) / dx,
    )
    # y-direction
    dTdy = np.where(
        velocity[1] >= 0,
        (Ty[:, 1:-1, :] - Ty[:, :-2, :]) / dx,
        (Ty[:, 2:, :] - Ty[:, 1:-1, :]) / dx,
    )
    # x-direction
    dTdx = np.where(
        velocity[2] >= 0,
        (Tx[:, :, 1:-1] - Tx[:, :, :-2]) / dx,
        (Tx[:, :, 2:] - Tx[:, :, 1:-1]) / dx,
    )

    adv = velocity[0] * dTdz + velocity[1] * dTdy + velocity[2] * dTdx
    return np.where(active, adv, 0.0)


def solve_3d_thermal(
    grid: np.ndarray,
    alloy: Alloy,
    mold: MoldMaterial,
    dx: float,
    max_time_s: float = 600.0,
    downsample: int = 2,
    progress_callback: Optional[callable] = None,
    fill_time_s: Optional[np.ndarray] = None,
    velocity_m_s: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Implicit 3-D enthalpy thermal solver with optional Darcy-velocity advection.

    If ``fill_time_s`` is supplied it is interpreted as the per-voxel metal
    arrival time (s).  Solidification/liquidus times are shifted by this amount,
    so late-filled regions start cooling later.

    If ``velocity_m_s`` is supplied as a (3, nx, ny, nz) array with [x, y, z]
    components, an explicit convective term ``-ρ·cp·(v·∇T)`` is added to the
    enthalpy balance.  The term is masked to metal cells whose fill time has
    already elapsed.

    Returns fine-grid arrays:
    T_final, fs_final, t_liquidus, t_solidus, G_at_ts, R_at_ts, niyama
    """
    fine_shape = grid.shape
    if downsample > 1:
        grid_c = _downsample_grid(grid, downsample)
        dx_c_mm = dx * downsample
        if fill_time_s is not None:
            fill_c = ndimage.zoom(
                fill_time_s,
                (
                    grid_c.shape[0] / fill_time_s.shape[0],
                    grid_c.shape[1] / fill_time_s.shape[1],
                    grid_c.shape[2] / fill_time_s.shape[2],
                ),
                order=0,
            )
        else:
            fill_c = None
        if velocity_m_s is not None and velocity_m_s.ndim == 4:
            ratios = (
                grid_c.shape[0] / velocity_m_s.shape[1],
                grid_c.shape[1] / velocity_m_s.shape[2],
                grid_c.shape[2] / velocity_m_s.shape[3],
            )
            velocity_c = np.stack(
                [ndimage.zoom(velocity_m_s[i], ratios, order=1) for i in range(3)],
                axis=0,
            )
        else:
            velocity_c = None
    else:
        grid_c = grid
        dx_c_mm = dx
        fill_c = fill_time_s
        velocity_c = velocity_m_s

    # Extend the simulation so even the last-filled metal has time to solidify.
    if fill_c is not None:
        max_fill = float(np.nanmax(fill_c[np.isfinite(fill_c)])) if np.isfinite(fill_c).any() else 0.0
        max_time_s = max(max_time_s, max_fill + max_time_s)

    dx_m = dx_c_mm / 1000.0  # SI metres
    nx, ny, nz = grid_c.shape
    n = nx * ny * nz

    # Casting metal: all liquid-metal body types (PART, RISER, INGATE, RUNNER,
    # SPRUE, SPRUE_THROAT, POURING_BASIN). Chill/filter are excluded.
    casting_metal_ids = [int(t) for t in BODY_METAL_TYPES]
    is_metal_c = np.isin(grid_c, casting_metal_ids)
    chill_mask_3d = grid_c == 11  # COOLING_SPRUE

    rho = np.where(is_metal_c, alloy.rho_kg_m3, mold.rho_kg_m3).astype(np.float64).ravel()
    k = np.where(is_metal_c, alloy.k_w_mk, mold.k_w_mk).astype(np.float64)
    T = np.where(is_metal_c, alloy.t_pour_c, mold.t0_c).astype(np.float64)
    T0 = float(mold.t0_c)

    # Treat a cooling sprue as a steel/cast-iron chill insert at the mould temperature.
    # It extracts heat like a high-conductivity metal but never melts.
    if np.any(chill_mask_3d):
        rho_chill = 7850.0
        k_chill = 45.0
        cp_chill = 460.0
        rho[chill_mask_3d.ravel()] = rho_chill
        k[chill_mask_3d] = k_chill
        T[chill_mask_3d] = T0

    # Boundary mask: fixed-temperature outer shell
    boundary = np.zeros((nx, ny, nz), dtype=bool)
    boundary[0, :, :] = True
    boundary[-1, :, :] = True
    boundary[:, 0, :] = True
    boundary[:, -1, :] = True
    boundary[:, :, 0] = True
    boundary[:, :, -1] = True
    boundary_idx = np.flatnonzero(boundary.ravel())

    # Build constant-in-time diffusion operator on the raveled grid
    A = _build_laplacian(k, dx_m)
    k = k.ravel()

    # Dirichlet T=T0 on the outer shell: solve the reduced SPD system for the
    # interior cells only.  Boundary columns move to the RHS, so the interior
    # matrix stays symmetric and CG converges; the old row-replacement trick
    # made the matrix non-symmetric and caused very expensive spsolve fallbacks.
    A = A.tocsr()
    interior_mask = ~boundary
    interior_idx = np.flatnonzero(interior_mask.ravel())
    A_ii = A[interior_idx][:, interior_idx]
    A_ib = A[interior_idx][:, boundary_idx]
    A_ib_sum = np.asarray(A_ib.sum(axis=1)).ravel().astype(np.float64)
    del A, A_ib

    # Time stepping
    n_steps_target = max(20, min(200, int(max_time_s / 3.0)))
    dt_diff = max_time_s / n_steps_target
    t = 0.0
    step = 0

    # Advective CFL limit; only used while metal is still being filled and a
    # velocity field is supplied.  After filling the diffusion time-step is
    # restored because the explicit advection term is no longer physically
    # relevant for a stationary solidifying metal.
    if velocity_c is not None:
        vmax = float(np.max(np.linalg.norm(velocity_c, axis=0))) if np.any(velocity_c) else 0.0
        dt_adv = 0.3 * dx_m / max(vmax, 1e-6)
    else:
        vmax = 0.0
        dt_adv = np.inf
    fill_end = max_fill if fill_c is not None and np.isfinite(max_fill) else 0.0

    t_liq = np.full((nx, ny, nz), np.inf, dtype=np.float64)
    t_sol = np.full((nx, ny, nz), np.inf, dtype=np.float64)
    G_at_ts = np.zeros((nx, ny, nz), dtype=np.float64)
    R_at_ts = np.zeros((nx, ny, nz), dtype=np.float64)

    Tl = alloy.t_liquidus_c
    Ts = alloy.t_solidus_c
    report_interval = max(1, n_steps_target // 10)

    # Operator-split advection: up to this many explicit advection sub-cycles
    # are performed inside each implicit diffusion step so the matrix solve is
    # not repeated for the tiny advective CFL time step.
    max_adv_subcycles = 200

    while t < max_time_s:
        # Pick a stable time step.  While filling, the explicit upwind
        # advection sub-cycles must satisfy CFL <= 0.3; the diffusion step is
        # limited to at most ``max_adv_subcycles`` of those sub-cycles.  After
        # filling the normal diffusion time step is used.  The final step lands
        # exactly on max_time_s.
        if velocity_c is not None and t < fill_end:
            dt = min(dt_diff, max_adv_subcycles * dt_adv)
        else:
            dt = dt_diff
        if t + dt > max_time_s:
            dt = max_time_s - t
        if dt <= 0:
            break

        T_old = T.copy()
        cp_eff = _cp_eff(T_old, is_metal_c, alloy, mold)
        # Cooling sprue cp stays as a solid metal (no latent heat).
        if np.any(chill_mask_3d):
            cp_eff[chill_mask_3d] = cp_chill
        C = rho * cp_eff.ravel()

        # Explicit Darcy-velocity advection via first-order operator splitting.
        # The temperature is advected with small CFL-limited sub-steps first,
        # then that pre-advected field becomes the initial condition for the
        # implicit diffusion solve over the same interval.
        T_adv = T_old.copy()
        if velocity_c is not None:
            n_sub = max(1, int(np.ceil(dt / dt_adv))) if np.isfinite(dt_adv) else 1
            sub_dt = dt / n_sub
            for k in range(n_sub):
                sub_t = t + (k + 0.5) * sub_dt
                adv = _upwind_advection(T_adv, velocity_c, dx_m, fill_c, sub_t, is_metal_c)
                T_adv = T_adv - sub_dt * adv
            # Clip to physical bounds after explicit advection.
            T_adv = np.clip(T_adv, T0, alloy.t_pour_c)

        T_adv_r = T_adv.ravel()
        C_i = C[interior_idx]
        b_i = C_i * T_adv_r[interior_idx] + dt * T0 * A_ib_sum

        M_ii = -dt * A_ii + sparse.diags(C_i, format="csc")

        # CG with a diagonal (Jacobi) preconditioner for the symmetric positive
        # definite reduced system.  Keeping the matrix SPD avoids the expensive
        # direct solves that occurred with the old non-symmetric row replacement.
        precond = sparse.diags(1.0 / (M_ii.diagonal() + 1e-12), format="csc")
        T_i, info = spla.cg(M_ii, b_i, rtol=1e-7, atol=0.0, maxiter=300, M=precond)
        if info != 0:
            # Last-resort direct solve on the much smaller interior system only.
            T_i = spla.spsolve(M_ii, b_i)

        T_new = np.full(n, T0, dtype=np.float64)
        T_new[interior_idx] = T_i
        T_new = T_new.reshape((nx, ny, nz))
        # Guard against NaN/Inf from the linear solver before clipping/gradient.
        T_new = np.nan_to_num(T_new, nan=T0, posinf=alloy.t_pour_c, neginf=T0)
        T_new = np.clip(T_new, T0, alloy.t_pour_c)

        # Record solidification times and local G/R
        if is_metal_c.any():
            cross_liq = (T_old >= Tl) & (T_new < Tl) & is_metal_c
            finite_liq = np.isfinite(t_liq)
            mask_liq = cross_liq & (~finite_liq)
            t_liq[mask_liq] = t + dt * (Tl - T_old[mask_liq]) / (T_new[mask_liq] - T_old[mask_liq] + 1e-12)

            cross_sol = (T_old >= Ts) & (T_new < Ts) & is_metal_c
            finite_sol = np.isfinite(t_sol)
            mask_sol = cross_sol & (~finite_sol)
            if np.any(mask_sol):
                gz, gy, gx = np.gradient(T_new, dx_m)
                G_cross = np.sqrt(gx * gx + gy * gy + gz * gz) / 1000.0  # K/mm
                # Enthalpy-based cooling rate: dH = cp*dT + L*dfs; R = |dH/dt| / cp_eff
                # so latent heat release during solid-fraction change is explicit.
                fs_old = _scheil_fs(T_old, Tl, Ts, alloy.partition_coefficient)
                fs_new = _scheil_fs(T_new, Tl, Ts, alloy.partition_coefficient)
                dH = alloy.cp_j_kgk * (T_new - T_old) + alloy.latent_heat_j_kg * (fs_new - fs_old)
                R_cross = np.abs(dH[mask_sol]) / (dt * np.maximum(cp_eff[mask_sol], 1e-9))
                t_sol[mask_sol] = t + dt * (Ts - T_old[mask_sol]) / (T_new[mask_sol] - T_old[mask_sol] + 1e-12)
                G_at_ts[mask_sol] = G_cross[mask_sol]
                R_at_ts[mask_sol] = R_cross

        T = T_new
        t += dt
        step += 1

        if progress_callback and step % report_interval == 0:
            progress_callback(int(20 + 40 * min(t / max_time_s, 1.0)))

        # Early stop once all metal has solidified
        if is_metal_c.any() and not np.isinf(t_sol[is_metal_c]).any():
            break

    # Niyama on coarse grid
    with np.errstate(divide="ignore", invalid="ignore"):
        niyama_c = G_at_ts / np.sqrt(np.maximum(R_at_ts, 1e-12))
    niyama_c = np.nan_to_num(niyama_c, nan=0.0, posinf=0.0, neginf=0.0)
    niyama_c = np.where(is_metal_c, niyama_c, 0.0)

    # Final fields
    T_c = T.reshape((nx, ny, nz))
    fs_c = _scheil_fs(T_c, Tl, Ts, alloy.partition_coefficient)

    # Upsample to fine grid
    T_fine = _upsample(T_c, fine_shape, order=1)
    fs_fine = _upsample(fs_c, fine_shape, order=1)
    # t_liq/t_sol may contain inf for cells that never crossed the relevant
    # temperature; use order=0 (nearest-neighbour) upsampling so inf does not
    # leak into neighbouring cells.
    t_liq_fine = _upsample(t_liq, fine_shape, order=0)
    t_sol_fine = _upsample(t_sol, fine_shape, order=0)
    G_fine = _upsample(G_at_ts, fine_shape, order=1)
    R_fine = _upsample(R_at_ts, fine_shape, order=1)
    niyama_fine = _upsample(niyama_c, fine_shape, order=1)

    is_metal_fine = np.isin(grid, casting_metal_ids)

    # Shift liquidus/solidus times by the local metal arrival time.
    if fill_c is not None:
        fill_time_fine = _upsample(fill_c, fine_shape, order=0)
        fill_time_fine = np.where(is_metal_fine, fill_time_fine, 0.0)
        with np.errstate(invalid="ignore"):
            t_liq_fine = np.where(
                is_metal_fine & np.isfinite(t_liq_fine) & np.isfinite(fill_time_fine),
                t_liq_fine + fill_time_fine,
                t_liq_fine,
            )
            t_sol_fine = np.where(
                is_metal_fine & np.isfinite(t_sol_fine) & np.isfinite(fill_time_fine),
                t_sol_fine + fill_time_fine,
                t_sol_fine,
            )

    for arr in (niyama_fine, G_fine, R_fine, t_liq_fine, t_sol_fine, fs_fine):
        arr[:] = np.where(is_metal_fine, arr, 0.0)

    return T_fine, fs_fine, t_liq_fine, t_sol_fine, G_fine, R_fine, niyama_fine
