"""Transient D3Q7 scalar LBM air-entrapment solver.

One-way coupled to the C++ D3Q19 LBM: the Python callback receives
``v`` (m/s), ``F`` (gas volume fraction = 1 - phi_metal), ``nu_t``
(m^2/s) and the D3Q7 solver advances ``alpha_g * rho_g`` for the gas
phase in lockstep with the metal flow.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numba
import numpy as np

# D3Q7 lattice: rest + 6 axis directions (lattice units).
Q7 = 7
C7 = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ],
    dtype=np.int64,
)
OPPO7 = np.array([0, 2, 1, 4, 3, 6, 5], dtype=np.int64)
W7 = np.array([1.0 / 4.0] + [1.0 / 8.0] * 6, dtype=np.float64)

# Physical constants / tunable parameters.
R_AIR = 287.0  # J/(kg K)
P_ATM = 101_325.0  # Pa
MU_AIR = 1.81e-5  # Pa s
SC_T = 0.7
C_ENTRAP = 1.0  # source magnitude, calibrated in validation
K_CRIT = 0.1  # 1/m


@numba.njit(cache=True)
def _in_bounds(i: int, j: int, k: int, nx: int, ny: int, nz: int) -> bool:
    return 0 <= i < nx and 0 <= j < ny and 0 <= k < nz


@numba.njit(parallel=True, cache=True)
def _compute_mold_boundary_numba(
    solid: np.ndarray, boundary: np.ndarray, nx: int, ny: int, nz: int
) -> None:
    """Flag every cavity cell that touches solid or the domain boundary."""
    for idx in numba.prange(nx * ny * nz):
        i = idx // (ny * nz)
        rem = idx % (ny * nz)
        j = rem // nz
        k = rem % nz

        if solid[i, j, k]:
            boundary[i, j, k] = False
            continue

        is_boundary = (
            i == 0
            or i == nx - 1
            or j == 0
            or j == ny - 1
            or k == 0
            or k == nz - 1
        )
        if not is_boundary:
            for q in range(1, Q7):
                si = i - C7[q, 0]
                sj = j - C7[q, 1]
                sk = k - C7[q, 2]
                if _in_bounds(si, sj, sk, nx, ny, nz) and solid[si, sj, sk]:
                    is_boundary = True
                    break
        boundary[i, j, k] = is_boundary


@numba.njit(parallel=True, cache=True)
def _compute_source_mask(
    F: np.ndarray,
    v: np.ndarray,
    solid: np.ndarray,
    dx: float,
    source: np.ndarray,
    rho_atm: float,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Compute entrapment mass source from free-surface curvature and div(v).

    The source is active where the free-surface marker F has a negative
    divergence (gas pocket being compressed) and the local mean curvature
    of F is large.  The result has units kg/(m^3 s).
    """
    for idx in numba.prange(nx * ny * nz):
        i = idx // (ny * nz)
        rem = idx % (ny * nz)
        j = rem // nz
        k = rem % nz

        if solid[i, j, k]:
            source[i, j, k] = 0.0
            continue

        # ---- div(v) (physical, 1/s) using central differences ----
        dudx = 0.0
        if 0 < i < nx - 1:
            dudx = (v[0, i + 1, j, k] - v[0, i - 1, j, k]) / (2.0 * dx)
        dvdy = 0.0
        if 0 < j < ny - 1:
            dvdy = (v[1, i, j + 1, k] - v[1, i, j - 1, k]) / (2.0 * dx)
        dwdz = 0.0
        if 0 < k < nz - 1:
            dwdz = (v[2, i, j, k + 1] - v[2, i, j, k - 1]) / (2.0 * dx)
        div_v = dudx + dvdy + dwdz

        # ---- mean curvature kappa = div(grad F / |grad F|) ----
        fx = 0.0
        fy = 0.0
        fz = 0.0
        if 0 < i < nx - 1:
            fx = (F[i + 1, j, k] - F[i - 1, j, k]) / (2.0 * dx)
        if 0 < j < ny - 1:
            fy = (F[i, j + 1, k] - F[i, j - 1, k]) / (2.0 * dx)
        if 0 < k < nz - 1:
            fz = (F[i, j, k + 1] - F[i, j, k - 1]) / (2.0 * dx)

        grad_mag = math.sqrt(fx * fx + fy * fy + fz * fz)
        kappa = 0.0
        if grad_mag > 1e-12:
            # Face-centered normals at +/- dx/2, then div(n).
            # n_x faces at i +/- 1/2:
            nx_p = 0.0
            nx_m = 0.0
            if i < nx - 1:
                gpx = F[i + 1, j, k] - F[i, j, k]
                gpy = F[i, j + 1, k] - F[i, j, k] if j < ny - 1 else 0.0
                gpz = F[i, j, k + 1] - F[i, j, k] if k < nz - 1 else 0.0
                gpmag = math.sqrt(gpx * gpx + gpy * gpy + gpz * gpz)
                if gpmag > 1e-12:
                    nx_p = gpx / gpmag
            if i > 0:
                gmx = F[i, j, k] - F[i - 1, j, k]
                gmy = F[i, j, k] - F[i, j - 1, k] if j > 0 else 0.0
                gmz = F[i, j, k] - F[i, j, k - 1] if k > 0 else 0.0
                gmmag = math.sqrt(gmx * gmx + gmy * gmy + gmz * gmz)
                if gmmag > 1e-12:
                    nx_m = gmx / gmmag

            # y and z face normals
            ny_p = 0.0
            ny_m = 0.0
            if j < ny - 1:
                gpx = F[i + 1, j, k] - F[i, j, k] if i < nx - 1 else 0.0
                gpy = F[i, j + 1, k] - F[i, j, k]
                gpz = F[i, j, k + 1] - F[i, j, k] if k < nz - 1 else 0.0
                gpmag = math.sqrt(gpx * gpx + gpy * gpy + gpz * gpz)
                if gpmag > 1e-12:
                    ny_p = gpy / gpmag
            if j > 0:
                gmx = F[i, j, k] - F[i - 1, j, k] if i > 0 else 0.0
                gmy = F[i, j, k] - F[i, j - 1, k]
                gmz = F[i, j, k] - F[i, j, k - 1] if k > 0 else 0.0
                gmmag = math.sqrt(gmx * gmx + gmy * gmy + gmz * gmz)
                if gmmag > 1e-12:
                    ny_m = gmy / gmmag

            nz_p = 0.0
            nz_m = 0.0
            if k < nz - 1:
                gpx = F[i + 1, j, k] - F[i, j, k] if i < nx - 1 else 0.0
                gpy = F[i, j + 1, k] - F[i, j, k] if j < ny - 1 else 0.0
                gpz = F[i, j, k + 1] - F[i, j, k]
                gpmag = math.sqrt(gpx * gpx + gpy * gpy + gpz * gpz)
                if gpmag > 1e-12:
                    nz_p = gpz / gpmag
            if k > 0:
                gmx = F[i, j, k] - F[i - 1, j, k] if i > 0 else 0.0
                gmy = F[i, j, k] - F[i, j - 1, k] if j > 0 else 0.0
                gmz = F[i, j, k] - F[i, j, k - 1]
                gmmag = math.sqrt(gmx * gmx + gmy * gmy + gmz * gmz)
                if gmmag > 1e-12:
                    nz_m = gmz / gmmag

            kappa = (nx_p - nx_m + ny_p - ny_m + nz_p - nz_m) / dx

        # Entrapment source: negative F divergence + high curvature
        # and velocity convergence.  The result is multiplied by rho_atm to
        # obtain a mass source rate [kg/(m^3 s)].
        div_F = dudx  # placeholder: use a separate central difference below
        # recompute div(F) using F
        dFdx = 0.0
        if 0 < i < nx - 1:
            dFdx = (F[i + 1, j, k] - F[i - 1, j, k]) / (2.0 * dx)
        dFdy = 0.0
        if 0 < j < ny - 1:
            dFdy = (F[i, j + 1, k] - F[i, j - 1, k]) / (2.0 * dx)
        dFdz = 0.0
        if 0 < k < nz - 1:
            dFdz = (F[i, j, k + 1] - F[i, j, k - 1]) / (2.0 * dx)
        div_F = dFdx + dFdy + dFdz

        # Trigger entrapment when the free surface is compressed (negative F
        # gradient per voxel) and the local mean curvature exceeds the critical
        # value.  The source strength uses the local velocity divergence so that
        # gas generation is strongest where the liquid front is collapsing.
        if div_F * dx < -0.5 and kappa > K_CRIT and div_v < -0.5:
            source[i, j, k] = C_ENTRAP * abs(div_v) * (1.0 - F[i, j, k]) * rho_atm
        else:
            source[i, j, k] = 0.0


@numba.njit(parallel=False, cache=False)
def _d3q7_step(
    g_old: np.ndarray,
    g_new: np.ndarray,
    alpha_rho: np.ndarray,
    alpha_rho_new: np.ndarray,
    u_lb: np.ndarray,  # (3, nx, ny, nz) lattice velocities
    F: np.ndarray,
    solid: np.ndarray,
    tau: np.ndarray,
    source: np.ndarray,
    m_dot_escape: np.ndarray,
    dt: float,
    cs2_inv: float,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """One D3Q7 collision-streaming step with source/sink terms."""
    for idx in numba.prange(nx * ny * nz):
        i = idx // (ny * nz)
        rem = idx % (ny * nz)
        j = rem // nz
        k = rem % nz

        if solid[i, j, k]:
            for q in range(Q7):
                g_new[q, i, j, k] = 0.0
            alpha_rho_new[i, j, k] = 0.0
            continue

        new_ar = 0.0
        for q in range(Q7):
            ex = C7[q, 0]
            ey = C7[q, 1]
            ez = C7[q, 2]

            # Pull from upstream (source) lattice site.
            si = i - ex
            sj = j - ey
            sk = k - ez

            if _in_bounds(si, sj, sk, nx, ny, nz) and not solid[si, sj, sk]:
                c_src = alpha_rho[si, sj, sk]
                u = u_lb[0, si, sj, sk]
                v = u_lb[1, si, sj, sk]
                w = u_lb[2, si, sj, sk]
                eu = cs2_inv * (ex * u + ey * v + ez * w)
                g_eq = W7[q] * c_src * (1.0 + eu)
                t = tau[si, sj, sk]
                g_streamed = g_old[q, si, sj, sk] - (g_old[q, si, sj, sk] - g_eq) / t
            else:
                # Halfway bounce-back of the post-collision distribution at the
                # wall-adjacent fluid node.  The incoming population is the
                # opposite direction at (i,j,k) after collision.
                opp = OPPO7[q]
                ex_opp = C7[opp, 0]
                ey_opp = C7[opp, 1]
                ez_opp = C7[opp, 2]
                c_src = alpha_rho[i, j, k]
                u = u_lb[0, i, j, k]
                v = u_lb[1, i, j, k]
                w = u_lb[2, i, j, k]
                eu = cs2_inv * (ex_opp * u + ey_opp * v + ez_opp * w)
                g_eq = W7[opp] * c_src * (1.0 + eu)
                t = tau[i, j, k]
                g_streamed = g_old[opp, i, j, k] - (g_old[opp, i, j, k] - g_eq) / t

            # Add mass source and Darcy sink, distributed over directions.
            net = source[i, j, k] - m_dot_escape[i, j, k]
            g_new[q, i, j, k] = g_streamed + dt * W7[q] * net

            if g_new[q, i, j, k] < 0.0:
                g_new[q, i, j, k] = 0.0

            new_ar += g_new[q, i, j, k]

        alpha_rho_new[i, j, k] = new_ar


@numba.njit(parallel=True, cache=False)
def _compute_darcy_sink(
    alpha_rho: np.ndarray,
    F: np.ndarray,
    rho_g: np.ndarray,
    P_gas: np.ndarray,
    is_boundary: np.ndarray,
    K_mold: np.ndarray,
    b_klink: np.ndarray,
    P_dynamic: np.ndarray,
    m_dot: np.ndarray,
    T_melt: float,
    dx: float,
    L_wall: float,
    dt: float,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Compute Darcy-Forchheimer mass loss through mold walls.

    The local permeability is corrected with the Klinkenberg slip factor
    ``K_app = K_inf * (1 + b_klink / P_gas)``.  At low gas pressures this
    raises the effective permeability; at high pressures it tends to K_inf.

    The driving pressure difference is the larger of the gas overpressure and
    a local metal dynamic pressure, so that vented sand moulds can lose gas
    even before the pocket has had time to build up a large overpressure.
    """
    for idx in numba.prange(nx * ny * nz):
        i = idx // (ny * nz)
        rem = idx % (ny * nz)
        j = rem // nz
        k = rem % nz

        if not is_boundary[i, j, k] or F[i, j, k] < 1e-6:
            m_dot[i, j, k] = 0.0
            continue

        rg = alpha_rho[i, j, k] / max(F[i, j, k], 1e-9)
        rho_g[i, j, k] = rg
        P_gas[i, j, k] = rg * R_AIR * T_melt

        dP = P_gas[i, j, k] - P_ATM
        pdyn = P_dynamic[i, j, k]
        if dP < pdyn:
            dP = pdyn
        if dP <= 0.0:
            m_dot[i, j, k] = 0.0
            continue

        K = K_mold[i, j, k]
        b = b_klink[i, j, k]
        # Klinkenberg correction: apparent permeability increases as P drops.
        K_app = K * (1.0 + b / max(P_gas[i, j, k], 1e-9))
        # m_dot_escape = K_app * A / (mu * L * V) * rho_g * dP; A/V ~ 1/dx.
        # Clamp per-cell sink so it cannot remove more gas than is locally
        # available in one time step (preserves non-negative distributions).
        md = (K_app / (MU_AIR * L_wall * dx)) * rg * dP
        max_sink = alpha_rho[i, j, k] / dt
        if md > max_sink:
            md = max_sink
        m_dot[i, j, k] = md


class AirEntrapmentSolver_D3Q7:
    """D3Q7 scalar LBM solver for trapped/escaping air in mold filling.

    The solver is meant to be used as the ``callback`` argument of
    ``josecast_core.solve_lbm_filling_callback``.  It receives the LBM
    velocity field (m/s), gas volume fraction ``F = 1 - phi_metal`` and
    eddy viscosity ``nu_t`` (m^2/s) every callback step and advances
    ``alpha_g * rho_g`` for the same number of LBM substeps.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        dx: float,
        T_melt: float = 993.0,
        L_wall: float = 1e-3,
        rho_atm: Optional[float] = None,
        tau_min: float = 1.0,
        rho_metal: float = 7000.0,
    ):
        self.nx, self.ny, self.nz = grid_shape
        self.dx = float(dx)
        self.dt = 0.0
        self.T_melt = float(T_melt)
        self.L_wall = float(L_wall)
        # Atmospheric density at melt temperature from ideal gas law.
        self.rho_atm = float(rho_atm if rho_atm is not None else P_ATM / (R_AIR * T_melt))
        self.tau_min = float(tau_min)
        self.cs2_lb = 1.0 / 4.0  # D3Q7 lattice speed of sound squared (lattice units)
        self.rho_metal = float(rho_metal)

        self.alpha_rho = np.zeros(grid_shape, dtype=np.float64)
        self.alpha_rho_new = np.zeros(grid_shape, dtype=np.float64)
        self.rho_g = np.full(grid_shape, self.rho_atm, dtype=np.float64)
        self.P_gas = np.full(grid_shape, P_ATM, dtype=np.float64)
        self.g_old = np.zeros((Q7, *grid_shape), dtype=np.float64)
        self.g_new = np.zeros((Q7, *grid_shape), dtype=np.float64)

        self.solid: Optional[np.ndarray] = None
        self.K_mold: Optional[np.ndarray] = None
        self.b_klink: Optional[np.ndarray] = None
        self.is_boundary: Optional[np.ndarray] = None

        self.last_step = -1
        self.total_mass_initial = 0.0

    def set_mold_properties(
        self,
        solid: Optional[np.ndarray] = None,
        is_mold_boundary: Optional[np.ndarray] = None,
        K_mold: Optional[np.ndarray] = None,
        b_klink: Optional[np.ndarray] = None,
    ) -> None:
        if solid is not None:
            self.solid = solid.astype(np.bool_)
        if is_mold_boundary is not None:
            self.is_boundary = is_mold_boundary.astype(np.bool_)
        if K_mold is not None:
            self.K_mold = K_mold.astype(np.float64)
        if b_klink is not None:
            self.b_klink = b_klink.astype(np.float64)

    def initialize_from_F(self, F: np.ndarray) -> None:
        """Set initial gas mass from the LBM gas volume fraction."""
        F = np.asarray(F, dtype=np.float64)
        alpha = np.clip(F, 0.0, 1.0)
        self.alpha_rho = alpha * self.rho_atm
        self.alpha_rho_new = self.alpha_rho.copy()
        for q in range(Q7):
            self.g_old[q] = W7[q] * self.alpha_rho
            self.g_new[q] = 0.0
        # Avoid NaN warnings in np.where by only dividing where gas exists.
        self.rho_g = np.full_like(self.alpha_rho, self.rho_atm)
        np.divide(self.alpha_rho, alpha, out=self.rho_g, where=alpha > 1e-9)
        self.P_gas = self.rho_g * R_AIR * self.T_melt
        self.total_mass_initial = float(self.alpha_rho.sum())

    def _infer_solid(self, F: np.ndarray, v: np.ndarray) -> np.ndarray:
        vmag = np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        return (F <= 1e-9) & (vmag <= 1e-18)

    def _compute_mold_boundary(self, solid: np.ndarray) -> np.ndarray:
        boundary = np.zeros_like(solid, dtype=np.bool_)
        _compute_mold_boundary_numba(solid, boundary, self.nx, self.ny, self.nz)
        return boundary

    def on_lbm_step(
        self,
        step: int,
        dt: float,
        dx: float,
        cs2: float,
        vx: np.ndarray,
        vy: np.ndarray,
        vz: np.ndarray,
        F: np.ndarray,
        nu_t: np.ndarray,
    ) -> None:
        """Callback invoked by the C++ D3Q19 LBM every N steps."""
        F = np.asarray(F, dtype=np.float64)
        vx = np.asarray(vx, dtype=np.float64)
        vy = np.asarray(vy, dtype=np.float64)
        vz = np.asarray(vz, dtype=np.float64)
        nu_t = np.asarray(nu_t, dtype=np.float64)

        if self.solid is None:
            self.solid = self._infer_solid(F, np.stack([vx, vy, vz], axis=0))
            if self.is_boundary is None:
                self.is_boundary = self._compute_mold_boundary(self.solid)
            if self.K_mold is None:
                self.K_mold = np.full_like(F, 1e-11, dtype=np.float64)
            if self.b_klink is None:
                self.b_klink = np.full_like(F, 0.1 * P_ATM, dtype=np.float64)

        # Physical and lattice velocity fields (used for dynamic pressure below).
        v = np.stack([vx, vy, vz], axis=0)
        v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
        # 0.5 * rho_metal * v^2 is the metal dynamic pressure that pushes gas
        # through the mould wall.  A floor (0.5 % atm) represents the minimum
        # metal-head pressure available to drive gas through a permeable wall.
        P_dynamic = 0.5 * self.rho_metal * v2
        # A modest floor (0.5 % atm) represents the minimum metal-head pressure
        # available to push gas through a permeable mould wall.
        P_floor = 0.005 * P_ATM
        P_dynamic = np.where(P_dynamic > P_floor, P_dynamic, P_floor)

        if self.dt == 0.0 or step == 0:
            self.dt = float(dt)
            self.dx = float(dx)
            self.initialize_from_F(F)
            self.last_step = int(step)
            return

        n_substeps = max(1, int(step) - self.last_step)
        self.last_step = int(step)

        # Lattice velocity for the D3Q7 advection step.
        u_lb = v * (self.dt / self.dx)

        # D3Q7 diffusion relaxation time.  cs2_lb = 1/4 for the weights above,
        # so tau = 0.5 + D_t * dt / (cs2_lb * dx^2) = 0.5 + 4 * D_t * dt / dx^2.
        D_t = nu_t / SC_T
        tau = 0.5 + D_t * self.dt / (self.cs2_lb * self.dx * self.dx)
        tau = np.clip(tau, self.tau_min, 2.0)

        source = np.zeros_like(F)
        _compute_source_mask(F, v, self.solid, self.dx, source, self.rho_atm, self.nx, self.ny, self.nz)

        m_dot = np.zeros_like(F)

        for _ in range(n_substeps):
            _compute_darcy_sink(
                self.alpha_rho,
                F,
                self.rho_g,
                self.P_gas,
                self.is_boundary,
                self.K_mold,
                self.b_klink,
                P_dynamic,
                m_dot,
                self.T_melt,
                self.dx,
                self.L_wall,
                self.dt,
                self.nx,
                self.ny,
                self.nz,
            )

            _d3q7_step(
                self.g_old,
                self.g_new,
                self.alpha_rho,
                self.alpha_rho_new,
                u_lb,
                F,
                self.solid,
                tau,
                source,
                m_dot,
                self.dt,
                1.0 / self.cs2_lb,
                self.nx,
                self.ny,
                self.nz,
            )
            self.g_old, self.g_new = self.g_new, self.g_old
            self.alpha_rho, self.alpha_rho_new = self.alpha_rho_new, self.alpha_rho

        alpha = np.clip(F, 1e-9, 1.0)
        # rho_g = (alpha*rho_g) / alpha; guard empty cells to avoid divide-by-zero warnings.
        # Clamp to rho_atm: gas cannot become rarer than the surrounding
        # atmosphere; if mass is removed the pocket volume (alpha_g) shrinks
        # instead of the density dropping below atmospheric.
        self.rho_g = np.full_like(self.alpha_rho, self.rho_atm)
        np.divide(self.alpha_rho, alpha, out=self.rho_g, where=F > 1e-9)
        self.rho_g = np.maximum(self.rho_g, self.rho_atm)
        self.P_gas = self.rho_g * R_AIR * self.T_melt

    def risk_field(self) -> np.ndarray:
        """Gas volume fraction as air-entrapment risk."""
        # Use the larger of the computed gas density and the atmospheric density;
        # this makes escaped gas in vented sand moulds show a reduced alpha_g.
        rho_eff = np.maximum(self.rho_g, self.rho_atm)
        return np.clip(self.alpha_rho / rho_eff, 0.0, 1.0)
