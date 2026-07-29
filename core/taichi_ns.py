"""Taichi-based 3-D incompressible free-surface filling solver.

The program stays in Python; Taichi JIT-compiles the hot loops to native code
(CPU/GPU).  This is a stable-fluids / projection-method solver with a level-set
token phi (<0 metal, >=0 air).  Walls (mold/core) are excluded from the solve.
Source cells are held filled and driven by a prescribed inlet velocity.
"""

import warnings
from typing import Optional

import numpy as np
try:
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
except Exception:  # pragma: no cover
    sp = None
    spsolve = None

try:
    import taichi as ti
except Exception as exc:  # pragma: no cover
    warnings.warn(f"Taichi import failed: {exc}")
    ti = None  # type: ignore


def _init_taichi_once() -> None:
    if ti is None:
        return
    if getattr(_init_taichi_once, "done", False):
        return
    try:
        ti.init(arch=ti.gpu)
    except Exception:
        ti.init(arch=ti.cpu)
    _init_taichi_once.done = True  # type: ignore


if ti is not None:
    _init_taichi_once()


@ti.func
def _clamp(v, lo, hi):
    if v < lo:
        v = lo
    if v > hi:
        v = hi
    return v


@ti.func
def _sample(qf, u, v, w, nx, ny, nz):
    """Trilinear sample with clamping to the nearest boundary cell."""
    iu = int(u)
    iv = int(v)
    iw = int(w)
    iu = _clamp(iu, 0, int(nx) - 1)
    iv = _clamp(iv, 0, int(ny) - 1)
    iw = _clamp(iw, 0, int(nz) - 1)
    return qf[iu, iv, iw]


@ti.func
def _lerp(a, b, t):
    return a + t * (b - a)


@ti.func
def _bilerp(qf, p, nx, ny, nz):
    """Trilinear interpolation of a scalar field at fractional grid coords."""
    u = p[0]
    v = p[1]
    w = p[2]
    iu = int(ti.floor(u))
    iv = int(ti.floor(v))
    iw = int(ti.floor(w))
    fu = u - iu
    fv = v - iv
    fw = w - iw
    iu = _clamp(iu, 0, int(nx) - 1)
    iv = _clamp(iv, 0, int(ny) - 1)
    iw = _clamp(iw, 0, int(nz) - 1)
    iu1 = _clamp(iu + 1, 0, int(nx) - 1)
    iv1 = _clamp(iv + 1, 0, int(ny) - 1)
    iw1 = _clamp(iw + 1, 0, int(nz) - 1)

    a = qf[iu, iv, iw]
    b = qf[iu1, iv, iw]
    c = qf[iu, iv1, iw]
    d = qf[iu1, iv1, iw]
    e = qf[iu, iv, iw1]
    f = qf[iu1, iv, iw1]
    g = qf[iu, iv1, iw1]
    h = qf[iu1, iv1, iw1]

    c00 = _lerp(a, e, fw)
    c10 = _lerp(b, f, fw)
    c01 = _lerp(c, g, fw)
    c11 = _lerp(d, h, fw)
    c0 = _lerp(c00, c01, fv)
    c1 = _lerp(c10, c11, fv)
    return _lerp(c0, c1, fu)


@ti.func
def _bilerp_vec(vf, p, nx, ny, nz):
    u = p[0]
    v = p[1]
    w = p[2]
    iu = int(ti.floor(u))
    iv = int(ti.floor(v))
    iw = int(ti.floor(w))
    fu = u - iu
    fv = v - iv
    fw = w - iw
    iu = _clamp(iu, 0, int(nx) - 1)
    iv = _clamp(iv, 0, int(ny) - 1)
    iw = _clamp(iw, 0, int(nz) - 1)
    iu1 = _clamp(iu + 1, 0, int(nx) - 1)
    iv1 = _clamp(iv + 1, 0, int(ny) - 1)
    iw1 = _clamp(iw + 1, 0, int(nz) - 1)

    a = vf[iu, iv, iw]
    b = vf[iu1, iv, iw]
    c = vf[iu, iv1, iw]
    d = vf[iu1, iv1, iw]
    e = vf[iu, iv, iw1]
    f = vf[iu1, iv, iw1]
    g = vf[iu, iv1, iw1]
    h = vf[iu1, iv1, iw1]

    c00 = a * (1 - fw) + e * fw
    c10 = b * (1 - fw) + f * fw
    c01 = c * (1 - fw) + g * fw
    c11 = d * (1 - fw) + h * fw
    c0 = c00 * (1 - fv) + c01 * fv
    c1 = c10 * (1 - fv) + c11 * fv
    return c0 * (1 - fu) + c1 * fu


@ti.func
def _backtrace(vf, p, dt, nx, ny, nz):
    """RK3 trace-back for semi-Lagrangian advection."""
    v1 = _bilerp_vec(vf, p, nx, ny, nz)
    p1 = p - 0.5 * dt * v1
    v2 = _bilerp_vec(vf, p1, nx, ny, nz)
    p2 = p - 0.75 * dt * v2
    v3 = _bilerp_vec(vf, p2, nx, ny, nz)
    return p - dt * ((2.0 / 9.0) * v1 + (1.0 / 3.0) * v2 + (4.0 / 9.0) * v3)


@ti.kernel
def _set_source_fields(is_source: ti.template(), is_wall: ti.template()):
    for I in ti.grouped(is_source):
        if is_source[I]:
            is_wall[I] = 0


@ti.kernel
def _set_source_vel(vel: ti.template(), phi: ti.template(),
                    is_source: ti.template(), is_wall: ti.template(),
                    ux: ti.f32, uy: ti.f32, uz: ti.f32,
                    gx: ti.f32, gy: ti.f32, gz: ti.f32,
                    nx: ti.i32, ny: ti.i32, nz: ti.i32):
    for i, j, k in vel:
        if is_source[i, j, k]:
            vel[i, j, k] = ti.Vector([ux, uy, uz])
            phi[i, j, k] = -1.0
            is_wall[i, j, k] = 0
            continue
        # Seed the immediate downstream cell(s) with the inflow velocity so the
        # level-set can be advected out of the inlet.  Only the cell whose
        # upstream neighbour is a source inherits the velocity.
        di = int(ti.cast(gx, ti.i32))
        dj = int(ti.cast(gy, ti.i32))
        dk = int(ti.cast(gz, ti.i32))
        si = i - di
        sj = j - dj
        sk = k - dk
        if (
            si >= 0 and si < nx and sj >= 0 and sj < ny and sk >= 0 and sk < nz
            and not is_wall[i, j, k]
            and is_source[si, sj, sk]
        ):
            vel[i, j, k] = ti.Vector([ux, uy, uz])


@ti.kernel
def _add_body_force(vel: ti.template(), is_wall: ti.template(),
                    is_source: ti.template(), phi: ti.template(), dt: ti.f32,
                    gx: ti.f32, gy: ti.f32, gz: ti.f32, g_mag: ti.f32):
    for I in ti.grouped(vel):
        if is_wall[I] or is_source[I] or phi[I] >= 0.0:
            continue
        vel[I] += dt * g_mag * ti.Vector([gx, gy, gz])


@ti.kernel
def _advect_scalar(vel: ti.template(), qf: ti.template(), new_qf: ti.template(),
                   dt: ti.f32, nx: ti.i32, ny: ti.i32, nz: ti.i32,
                   is_wall: ti.template(), is_source: ti.template()):
    for i, j, k in qf:
        if is_wall[i, j, k]:
            new_qf[i, j, k] = 999.0
            continue
        p = ti.Vector([i + 0.5, j + 0.5, k + 0.5])
        p = _backtrace(vel, p, dt, nx, ny, nz)
        val = _bilerp(qf, p, nx, ny, nz)
        if is_source[i, j, k]:
            val = -1.0
        new_qf[i, j, k] = val


@ti.kernel
def _advect_vel(vel: ti.template(), new_vel: ti.template(),
                dt: ti.f32, nx: ti.i32, ny: ti.i32, nz: ti.i32,
                is_wall: ti.template(), is_source: ti.template()):
    for i, j, k in vel:
        if is_wall[i, j, k] or is_source[i, j, k]:
            new_vel[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            continue
        p = ti.Vector([i + 0.5, j + 0.5, k + 0.5])
        p = _backtrace(vel, p, dt, nx, ny, nz)
        new_vel[i, j, k] = _bilerp_vec(vel, p, nx, ny, nz)


@ti.kernel
def _viscosity_step(vel: ti.template(), new_vel: ti.template(),
                    is_wall: ti.template(), is_source: ti.template(),
                    dt: ti.f32, dx: ti.f32, nu: ti.f32):
    coeff = nu * dt / (dx * dx)
    for i, j, k in vel:
        if is_wall[i, j, k] or is_source[i, j, k]:
            new_vel[i, j, k] = vel[i, j, k]
            continue
        v_sum = ti.Vector([0.0, 0.0, 0.0])
        n = 0
        # 6-neighbour Laplacian
        if not is_wall[i + 1, j, k]:
            v_sum += vel[i + 1, j, k]
            n += 1
        if not is_wall[i - 1, j, k]:
            v_sum += vel[i - 1, j, k]
            n += 1
        if not is_wall[i, j + 1, k]:
            v_sum += vel[i, j + 1, k]
            n += 1
        if not is_wall[i, j - 1, k]:
            v_sum += vel[i, j - 1, k]
            n += 1
        if not is_wall[i, j, k + 1]:
            v_sum += vel[i, j, k + 1]
            n += 1
        if not is_wall[i, j, k - 1]:
            v_sum += vel[i, j, k - 1]
            n += 1
        if n > 0:
            new_vel[i, j, k] = vel[i, j, k] + coeff * (v_sum - n * vel[i, j, k])
        else:
            new_vel[i, j, k] = vel[i, j, k]


@ti.kernel
def _divergence(vel: ti.template(), div: ti.template(),
                phi: ti.template(), is_wall: ti.template(),
                nx: ti.i32, ny: ti.i32, nz: ti.i32):
    for i, j, k in div:
        if is_wall[i, j, k] or phi[i, j, k] >= 0.0:
            div[i, j, k] = 0.0
            continue
        nx_i = int(nx)
        ny_i = int(ny)
        nz_i = int(nz)
        vc = vel[i, j, k]
        iL = _clamp(i - 1, 0, nx_i - 1)
        iR = _clamp(i + 1, 0, nx_i - 1)
        jB = _clamp(j - 1, 0, ny_i - 1)
        jT = _clamp(j + 1, 0, ny_i - 1)
        kD = _clamp(k - 1, 0, nz_i - 1)
        kU = _clamp(k + 1, 0, nz_i - 1)

        vl = vel[iL, j, k]
        vr = vel[iR, j, k]
        vb = vel[i, jB, k]
        vt = vel[i, jT, k]
        vzf = vel[i, j, kU]
        vzb = vel[i, j, kD]

        # Mirror at domain boundaries so no wall-flux.
        if i == 0:
            vl.x = -vc.x
        if i == nx_i - 1:
            vr.x = -vc.x
        if j == 0:
            vb.y = -vc.y
        if j == ny_i - 1:
            vt.y = -vc.y
        if k == 0:
            vzb.z = -vc.z
        if k == nz_i - 1:
            vzf.z = -vc.z

        div[i, j, k] = 0.5 * ((vr.x - vl.x) + (vt.y - vb.y) + (vzf.z - vzb.z))


@ti.kernel
def _pressure_jacobi(pf: ti.template(), new_pf: ti.template(),
                       div: ti.template(), phi: ti.template(),
                       is_wall: ti.template(),
                       nx: ti.i32, ny: ti.i32, nz: ti.i32):
    for i, j, k in pf:
        if is_wall[i, j, k] or phi[i, j, k] >= 0.0:
            new_pf[i, j, k] = 0.0
            continue
        nx_i = int(nx)
        ny_i = int(ny)
        nz_i = int(nz)
        iL = _clamp(i - 1, 0, nx_i - 1)
        iR = _clamp(i + 1, 0, nx_i - 1)
        jB = _clamp(j - 1, 0, ny_i - 1)
        jT = _clamp(j + 1, 0, ny_i - 1)
        kD = _clamp(k - 1, 0, nz_i - 1)
        kU = _clamp(k + 1, 0, nz_i - 1)

        pl = pf[iL, j, k]
        pr = pf[iR, j, k]
        pb = pf[i, jB, k]
        pt = pf[i, jT, k]
        pzf = pf[i, j, kU]
        pzb = pf[i, j, kD]

        # Mirror pressure at domain walls; air already clamped to 0 above.
        if i == 0:
            pl = pf[i, j, k]
        if i == nx_i - 1:
            pr = pf[i, j, k]
        if j == 0:
            pb = pf[i, j, k]
        if j == ny_i - 1:
            pt = pf[i, j, k]
        if k == 0:
            pzb = pf[i, j, k]
        if k == nz_i - 1:
            pzf = pf[i, j, k]

        new_pf[i, j, k] = (pl + pr + pb + pt + pzf + pzb - div[i, j, k]) * (1.0 / 6.0)


@ti.kernel
def _subtract_gradient(vel: ti.template(), pf: ti.template(),
                       phi: ti.template(), is_wall: ti.template(), is_source: ti.template(),
                       nx: ti.i32, ny: ti.i32, nz: ti.i32):
    for i, j, k in vel:
        if is_wall[i, j, k] or is_source[i, j, k]:
            continue
        nx_i = int(nx)
        ny_i = int(ny)
        nz_i = int(nz)
        iL = _clamp(i - 1, 0, nx_i - 1)
        iR = _clamp(i + 1, 0, nx_i - 1)
        jB = _clamp(j - 1, 0, ny_i - 1)
        jT = _clamp(j + 1, 0, ny_i - 1)
        kD = _clamp(k - 1, 0, nz_i - 1)
        kU = _clamp(k + 1, 0, nz_i - 1)
        pl = pf[iL, j, k]
        pr = pf[iR, j, k]
        pb = pf[i, jB, k]
        pt = pf[i, jT, k]
        pzf = pf[i, j, kU]
        pzb = pf[i, j, kD]
        grad = ti.Vector([pr - pl, pt - pb, pzf - pzb]) * 0.5
        vel[i, j, k] -= grad


@ti.kernel
def _redistance_phi(phi: ti.template(), new_phi: ti.template(),
                    is_wall: ti.template(), dx: ti.f32,
                    nx: ti.i32, ny: ti.i32, nz: ti.i32):
    """Few PDE redistance sweeps on a narrow band around the interface."""
    for i, j, k in phi:
        if is_wall[i, j, k]:
            new_phi[i, j, k] = phi[i, j, k]
            continue
        if ti.abs(phi[i, j, k]) > 3.0 * dx:
            new_phi[i, j, k] = phi[i, j, k]
            continue
        nx_i = int(nx)
        ny_i = int(ny)
        nz_i = int(nz)
        s0 = phi[i, j, k] / ti.sqrt(phi[i, j, k] * phi[i, j, k] + dx * dx)
        a = (phi[_clamp(i + 1, 0, nx_i - 1), j, k] - phi[_clamp(i - 1, 0, nx_i - 1), j, k]) / (2.0 * dx)
        b = (phi[i, _clamp(j + 1, 0, ny_i - 1), k] - phi[i, _clamp(j - 1, 0, ny_i - 1), k]) / (2.0 * dx)
        c = (phi[i, j, _clamp(k + 1, 0, nz_i - 1)] - phi[i, j, _clamp(k - 1, 0, nz_i - 1)]) / (2.0 * dx)
        grad = ti.sqrt(a * a + b * b + c * c + 1e-12)
        new_phi[i, j, k] = phi[i, j, k] - 0.5 * dx * s0 * (grad - 1.0)


@ti.kernel
def _record_fill(phi: ti.template(), fill_time: ti.template(), t: ti.f32):
    for I in ti.grouped(phi):
        if phi[I] < 0.0 and fill_time[I] > 1e10:
            fill_time[I] = t


@ti.kernel
def _compute_vmag(vel: ti.template(), vmag: ti.template(), is_wall: ti.template()):
    for I in ti.grouped(vel):
        if is_wall[I]:
            vmag[I] = 0.0
        else:
            v = vel[I]
            vmag[I] = ti.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


@ti.kernel
def _max_vel_field(vel: ti.template(), is_wall: ti.template()) -> ti.f32:
    mx = 0.0
    for I in ti.grouped(vel):
        if not is_wall[I]:
            v = vel[I]
            s = ti.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
            if s > mx:
                mx = s
    return mx


@ti.kernel
def _filled_fraction(phi: ti.template(), is_wall: ti.template()) -> ti.f32:
    filled = 0.0
    total = 0.0
    for I in ti.grouped(phi):
        if not is_wall[I]:
            total += 1.0
            if phi[I] < 0.0:
                filled += 1.0
    return filled / (total + 1e-12)


@ti.kernel
def _extrapolate_velocity(vel: ti.template(), new_vel: ti.template(), phi: ti.template(),
                          is_wall: ti.template(), is_source: ti.template(),
                          nx: ti.i32, ny: ti.i32, nz: ti.i32):
    """Extend the liquid velocity into a one-cell air band for semi-Lagrangian advection."""
    for i, j, k in vel:
        if is_wall[i, j, k] or is_source[i, j, k] or phi[i, j, k] < 0.0:
            new_vel[i, j, k] = vel[i, j, k]
            continue
        v_sum = ti.Vector([0.0, 0.0, 0.0])
        n = 0
        nx_i = int(nx); ny_i = int(ny); nz_i = int(nz)
        if i + 1 < nx_i and phi[i + 1, j, k] < 0.0 and not is_wall[i + 1, j, k]:
            v_sum += vel[i + 1, j, k]; n += 1
        if i - 1 >= 0 and phi[i - 1, j, k] < 0.0 and not is_wall[i - 1, j, k]:
            v_sum += vel[i - 1, j, k]; n += 1
        if j + 1 < ny_i and phi[i, j + 1, k] < 0.0 and not is_wall[i, j + 1, k]:
            v_sum += vel[i, j + 1, k]; n += 1
        if j - 1 >= 0 and phi[i, j - 1, k] < 0.0 and not is_wall[i, j - 1, k]:
            v_sum += vel[i, j - 1, k]; n += 1
        if k + 1 < nz_i and phi[i, j, k + 1] < 0.0 and not is_wall[i, j, k + 1]:
            v_sum += vel[i, j, k + 1]; n += 1
        if k - 1 >= 0 and phi[i, j, k - 1] < 0.0 and not is_wall[i, j, k - 1]:
            v_sum += vel[i, j, k - 1]; n += 1
        if n > 0:
            new_vel[i, j, k] = v_sum * (1.0 / n)
        else:
            new_vel[i, j, k] = vel[i, j, k]


@ti.kernel
def _zero_vel_air(vel: ti.template(), phi: ti.template(), is_wall: ti.template()):
    for I in ti.grouped(vel):
        if is_wall[I] or phi[I] >= 0.0:
            vel[I] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def _zero_fields(phi: ti.template(), fill_time: ti.template(), vel: ti.template()):
    for I in ti.grouped(phi):
        phi[I] = 0.0
        fill_time[I] = 1e12
        vel[I] = ti.Vector([0.0, 0.0, 0.0])


def _pressure_projection_jacobi(
    vel,
    pressure,
    new_pressure,
    div,
    phi,
    is_wall,
    is_source,
    nx,
    ny,
    nz,
    iters: int,
):
    """Solve ∇·u = 0 in the metal region with damped Jacobi iterations.

    A pure Taichi pressure projection avoids rebuilding a sparse matrix at each
    time step and is much faster on moderate grids.  The free surface (phi>=0)
    and walls are treated as p=0 Dirichlet points, so the velocity correction
    only acts inside the liquid.
    """
    pressure.fill(0.0)
    new_pressure.fill(0.0)
    for _ in range(iters):
        _pressure_jacobi(pressure, new_pressure, div, phi, is_wall, nx, ny, nz)
        pressure, new_pressure = new_pressure, pressure
    _subtract_gradient(vel, pressure, phi, is_wall, is_source, nx, ny, nz)


def solve(
    grid: np.ndarray,
    phi_init: np.ndarray,
    source_mask: np.ndarray,
    dx: float,
    g: np.ndarray,
    rho: float = 7850.0,
    nu: float = 1e-5,
    inflow_velocity: float = 1.5,
    t_max: float = 2.0,
    max_steps: int = 2000,
    pressure_iters: int = 80,
    reinit_iters: int = 3,
    cfl: float = 0.5,
) -> Optional[dict]:
    """Run the Taichi free-surface NS solver.

    Parameters
    ----------
    grid: uint8 (nx,ny,nz)
        BodyType enum; EMPTY (0) and CORE (9) are treated as walls.
    phi_init: float64 (nx,ny,nz)
        Signed distance field: <0 = already metal, >=0 = air.
    source_mask: uint8 (nx,ny,nz)
        Cells that are held filled and driven by ``inflow_velocity``.
    dx, g, rho, nu, inflow_velocity: physical parameters.
    t_max: stop when simulation time reaches this.
    max_steps, pressure_iters, reinit_iters: numerical controls.

    Returns
    -------
    dict with ``fill_time`` and ``velocity`` arrays, or None if Taichi unavailable.
    """
    if ti is None:
        return None

    if grid.ndim != 3:
        raise ValueError("grid must be 3-D")

    nx, ny, nz = grid.shape
    g = np.asarray(g, dtype=np.float64)
    gnorm = float(np.linalg.norm(g)) + 1e-12
    gx, gy, gz = float(g[0] / gnorm), float(g[1] / gnorm), float(g[2] / gnorm)

    # Work in grid-cell units: velocity in cells/second, viscosity in cells^2/s,
    # gravity in cells/s^2.  This makes the finite-difference divergence and
    # gradient operators dimensionally consistent without carrying dx through
    # every kernel.
    if dx <= 0.0:
        dx = 1.0
    dx_inv = 1.0 / dx
    v_in = float(inflow_velocity) * dx_inv
    g_mag = 9.81 * dx_inv
    nu_grid = float(nu) * dx_inv * dx_inv
    dx_grid = 1.0

    is_wall_np = ((grid == 0) | (grid == 9)).astype(np.int32)
    is_source_np = source_mask.astype(np.int32)
    is_wall_np[is_source_np.astype(bool)] = 0

    phi_np = phi_init.astype(np.float32)
    phi_np[is_source_np.astype(bool)] = -1.0

    shape = (nx, ny, nz)
    vel = ti.Vector.field(3, ti.f32, shape=shape)
    new_vel = ti.Vector.field(3, ti.f32, shape=shape)
    pressure = ti.field(ti.f32, shape=shape)
    new_pressure = ti.field(ti.f32, shape=shape)
    phi = ti.field(ti.f32, shape=shape)
    new_phi = ti.field(ti.f32, shape=shape)
    fill_time = ti.field(ti.f32, shape=shape)
    div = ti.field(ti.f32, shape=shape)
    is_wall = ti.field(ti.i32, shape=shape)
    is_source = ti.field(ti.i32, shape=shape)
    vmag = ti.field(ti.f32, shape=shape)

    phi.from_numpy(phi_np)
    is_wall.from_numpy(is_wall_np)
    is_source.from_numpy(is_source_np)
    fill_time.fill(1.0e12)

    ux = v_in * gx
    uy = v_in * gy
    uz = v_in * gz

    t = 0.0
    step = 0
    success = False
    filled = 0.0

    for step in range(max_steps):
        if t >= t_max:
            success = True
            break

        vmax = _max_vel_field(vel, is_wall)
        dt = cfl / (max(vmax, v_in) + 1e-6)
        if nu_grid > 0.0:
            dt_visc = 0.5 / (nu_grid * 6.0)
            if dt_visc < dt:
                dt = dt_visc
        remaining = t_max - t
        if dt > remaining:
            dt = remaining
        if dt <= 0.0:
            break

        _set_source_vel(vel, phi, is_source, is_wall, ux, uy, uz, gx, gy, gz, nx, ny, nz)
        _add_body_force(vel, is_wall, is_source, phi, dt, gx, gy, gz, g_mag)
        _advect_vel(vel, new_vel, dt, nx, ny, nz, is_wall, is_source)
        vel, new_vel = new_vel, vel
        _set_source_vel(vel, phi, is_source, is_wall, ux, uy, uz, gx, gy, gz, nx, ny, nz)
        _viscosity_step(vel, new_vel, is_wall, is_source, dt, dx_grid, nu_grid)
        vel, new_vel = new_vel, vel
        _divergence(vel, div, phi, is_wall, nx, ny, nz)
        _pressure_projection_jacobi(
            vel, pressure, new_pressure, div, phi, is_wall, is_source, nx, ny, nz, pressure_iters
        )
        _set_source_vel(vel, phi, is_source, is_wall, ux, uy, uz, gx, gy, gz, nx, ny, nz)
        # Extrapolate liquid velocity into the surrounding air band so the level-set
        # can be advected by a continuous velocity field near the free surface.
        for _ in range(3):
            _extrapolate_velocity(vel, new_vel, phi, is_wall, is_source, nx, ny, nz)
            vel, new_vel = new_vel, vel
        _set_source_vel(vel, phi, is_source, is_wall, ux, uy, uz, gx, gy, gz, nx, ny, nz)
        _advect_scalar(vel, phi, new_phi, dt, nx, ny, nz, is_wall, is_source)
        phi, new_phi = new_phi, phi
        _set_source_vel(vel, phi, is_source, is_wall, ux, uy, uz, gx, gy, gz, nx, ny, nz)
        for _ in range(reinit_iters):
            _redistance_phi(phi, new_phi, is_wall, dx_grid, nx, ny, nz)
            phi, new_phi = new_phi, phi
        _set_source_vel(vel, phi, is_source, is_wall, ux, uy, uz, gx, gy, gz, nx, ny, nz)
        _zero_vel_air(vel, phi, is_wall)
        _record_fill(phi, fill_time, t + dt)

        t += dt
        filled = _filled_fraction(phi, is_wall)
        if filled >= 0.9999:
            success = True
            break

    if t >= t_max:
        success = True

    _compute_vmag(vel, vmag, is_wall)

    # Convert velocities back to m/s.
    fill_out = fill_time.to_numpy().astype(np.float64)
    vmag_out = (vmag.to_numpy() * dx).astype(np.float64)
    vel_np = (vel.to_numpy() * dx).astype(np.float64)  # (nx, ny, nz, 3)
    velocity_out = np.moveaxis(vel_np, -1, 0)  # (3, nx, ny, nz)
    phi_out = phi.to_numpy().astype(np.float64)

    return {
        "fill_time": fill_out,
        "velocity_magnitude": vmag_out,
        "velocity": velocity_out,
        "phi": phi_out,
        "success": success,
        "final_t": float(t),
        "filled_fraction": float(filled),
        "steps": step,
    }
