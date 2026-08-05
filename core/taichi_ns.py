"""Taichi-based 3-D free-surface VOF/continuity filling solver.

The program stays in Python; the heavy per-cell output kernels JIT-compile to
native CPU/GPU code.  The solver tracks the metal-air interface with a sharp
level-set ``phi`` (negative = metal, positive = air) and computes a divergence-
free plug-flow velocity field whose speed follows the local cavity cross-section:
``Q = v · A = constant``.

For gravity casting (high Reynolds, short fill time) this continuity-correct
plug-flow is the physically dominant behaviour.  The solver now handles both
source-at-the-top and side-gated geometries by treating the cavity as a pair of
branches (upstream and downstream of the inlet along ``g``) and splitting the
total flow so that every connected branch fills at the same total time.
"""

import warnings
from typing import Optional

import numpy as np

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


@ti.kernel
def _build_output_fields(
    phi: ti.template(),
    fill_time: ti.template(),
    vel: ti.template(),
    vmag: ti.template(),
    speed: ti.template(),
    layer_t: ti.template(),
    branch_dir: ti.template(),
    is_wall: ti.template(),
    is_source: ti.template(),
    axis: ti.i32,
    gx: ti.f32,
    gy: ti.f32,
    gz: ti.f32,
    v_in_grid: ti.f32,
    dx: ti.f32,
    band: ti.f32,
    t_max: ti.f32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Assemble per-cell phi, fill_time, velocity and vmag from 1-D plug-flow data."""
    for i, j, k in phi:
        if is_source[i, j, k]:
            vel[i, j, k] = ti.Vector([v_in_grid * gx, v_in_grid * gy, v_in_grid * gz])
            vmag[i, j, k] = v_in_grid * dx
            fill_time[i, j, k] = 0.0
            phi[i, j, k] = -band
            continue
        if is_wall[i, j, k]:
            vel[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            vmag[i, j, k] = 0.0
            fill_time[i, j, k] = 1e12
            phi[i, j, k] = band
            continue

        layer = ti.cast(0, ti.i32)
        if axis == 0:
            layer = i
        elif axis == 1:
            layer = j
        else:
            layer = k

        spd = speed[layer]
        dir_x = gx
        dir_y = gy
        dir_z = gz
        if branch_dir[layer] < 0.0:
            dir_x = -gx
            dir_y = -gy
            dir_z = -gz
        vel[i, j, k] = ti.Vector([spd * dir_x, spd * dir_y, spd * dir_z])
        vmag[i, j, k] = spd * dx
        t = layer_t[layer]
        fill_time[i, j, k] = t
        if t <= t_max:
            phi[i, j, k] = -band
        else:
            phi[i, j, k] = band


@ti.kernel
def _filled_fraction_from_time(fill_time: ti.template(), is_wall: ti.template(), t_max: ti.f32) -> ti.f32:
    filled = 0.0
    total = 0.0
    for I in ti.grouped(fill_time):
        if not is_wall[I]:
            total += 1.0
            if fill_time[I] <= t_max:
                filled += 1.0
    return filled / (total + 1e-12)


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
    reinit_iters: int = 0,
    cfl: float = 0.5,
) -> Optional[dict]:
    """Run the Taichi VOF/continuity free-surface filling solver.

    The velocity field is a divergence-free plug flow whose speed at each
    cross-section perpendicular to ``g`` is ``v(z) = Q / A(z)``.  ``fill_time``
    is obtained by integrating ``dt = A(z) dz / Q`` from the inlet; ``phi`` is a
    sharp indicator of the metal region at ``t_max``.

    Parameters
    ----------
    grid: uint8 (nx, ny, nz)
        BodyType enum; EMPTY (0) and CORE (9) are treated as walls.
    phi_init: float64 (nx, ny, nz)
        Kept for API compatibility; reinitialised internally.
    source_mask: uint8 (nx, ny, nz)
        Inlet cells held filled and driven by ``inflow_velocity``.
    dx, g, rho, nu, inflow_velocity: physical parameters (rho/nu kept for API).
    t_max: simulation horizon in seconds.
    max_steps, pressure_iters, reinit_iters, cfl: numerical controls kept for
        API compatibility; not used by the analytical plug-flow solver.

    Returns
    -------
    dict with ``fill_time``, ``velocity``, ``velocity_magnitude`` and metadata,
    or None if Taichi is unavailable.
    """
    if ti is None:
        return None

    if grid.ndim != 3:
        raise ValueError("grid must be 3-D")

    nx, ny, nz = grid.shape
    g = np.asarray(g, dtype=np.float64)
    gnorm = float(np.linalg.norm(g)) + 1e-12
    gx, gy, gz = float(g[0] / gnorm), float(g[1] / gnorm), float(g[2] / gnorm)

    if dx <= 0.0:
        dx = 1.0

    is_wall_np = ((grid == 0) | (grid == 9)).astype(np.int32)
    is_source_np = source_mask.astype(np.int32)
    is_wall_np[is_source_np.astype(bool)] = 0

    # Dominant gravity axis and layer bookkeeping.
    abs_g = np.abs(g)
    axis = int(np.argmax(abs_g))
    n_axis = [nx, ny, nz][axis]

    axes = [0, 1, 2]
    axes.remove(axis)
    ax0, ax1 = axes

    # Cross-sectional cell counts perpendicular to the gravity axis.
    A_cavity = np.sum(is_wall_np == 0, axis=(ax0, ax1)).astype(np.float64)
    A_source = np.sum(is_source_np == 1, axis=(ax0, ax1)).astype(np.float64)
    max_source = int(A_source.max())
    if max_source <= 0:
        return None

    source_layers = np.where(A_source > 0)[0]
    sign = int(np.sign(g[axis]))
    if sign < 0:
        source_bottom = int(source_layers.min())
    else:
        source_bottom = int(source_layers.max())

    # Inlet speed in grid units and total flow rate.
    v_in_grid = float(inflow_velocity) / dx
    Q_grid = v_in_grid * float(max_source)  # cells^3 / s at the inlet
    if Q_grid <= 1e-18:
        return None

    # Branch volumes (grid cells) downstream and upstream of the inlet.
    if sign < 0:
        lower_layers = list(range(source_bottom - 1, -1, -1))
        upper_layers = list(range(source_bottom + 1, n_axis))
    else:
        lower_layers = list(range(source_bottom + 1, n_axis))
        upper_layers = list(range(source_bottom - 1, -1, -1))

    V_lower = float(sum(A_cavity[layer] for layer in lower_layers))
    V_upper = float(sum(A_cavity[layer] for layer in upper_layers))
    V_total = V_lower + V_upper

    def _branch_Q(branch_V):
        if V_total <= 1e-18:
            return Q_grid
        return Q_grid * (branch_V / V_total)

    Q_lower = _branch_Q(V_lower)
    Q_upper = _branch_Q(V_upper)

    # Per-layer speed and fill time.  The sign of branch_dir encodes the
    # movement direction: +1 along g (downstream), -1 opposite g (upstream).
    speed_np = np.zeros(n_axis, dtype=np.float32)
    branch_dir = np.zeros(n_axis, dtype=np.float32)
    layer_t = np.full(n_axis, 1e12, dtype=np.float32)
    layer_t[source_layers] = 0.0

    def _fill_branch(layers, Q, direction):
        if Q <= 1e-18 or not layers:
            return
        cum = 0.0
        for layer in layers:
            A = A_cavity[layer]
            if A > 0.0:
                dt = float(A) / Q
                cum += 0.5 * dt
                layer_t[layer] = cum
                speed_np[layer] = float(Q / A)
                branch_dir[layer] = float(direction)
                cum += 0.5 * dt

    _fill_branch(lower_layers, Q_lower, 1.0)
    _fill_branch(upper_layers, Q_upper, -1.0)

    band = 3.0

    shape = (nx, ny, nz)
    phi = ti.field(ti.f32, shape=shape)
    fill_time = ti.field(ti.f32, shape=shape)
    vel = ti.Vector.field(3, ti.f32, shape=shape)
    vmag = ti.field(ti.f32, shape=shape)
    is_wall = ti.field(ti.i32, shape=shape)
    is_source = ti.field(ti.i32, shape=shape)
    speed_1d = ti.field(ti.f32, shape=n_axis)
    layer_t_1d = ti.field(ti.f32, shape=n_axis)
    branch_dir_1d = ti.field(ti.f32, shape=n_axis)

    is_wall.from_numpy(is_wall_np)
    is_source.from_numpy(is_source_np)
    speed_1d.from_numpy(speed_np)
    layer_t_1d.from_numpy(layer_t)
    branch_dir_1d.from_numpy(branch_dir)

    _build_output_fields(
        phi, fill_time, vel, vmag, speed_1d, layer_t_1d, branch_dir_1d,
        is_wall, is_source, int(axis), gx, gy, gz,
        v_in_grid, dx, band, float(t_max), nx, ny, nz,
    )

    filled = _filled_fraction_from_time(fill_time, is_wall, float(t_max))
    finite_times = layer_t[(A_cavity > 0) & (layer_t < 1e10)]
    max_fill_t = float(np.max(finite_times)) if finite_times.size > 0 else 0.0
    final_t = min(float(t_max), max_fill_t)
    success = filled >= 0.9999 or final_t >= max_fill_t

    phi_out = phi.to_numpy().astype(np.float64)
    ft_out = fill_time.to_numpy().astype(np.float64)
    vmag_out = vmag.to_numpy().astype(np.float64)
    vel_np = vel.to_numpy() * dx
    velocity_out = np.moveaxis(vel_np.astype(np.float64), -1, 0)

    return {
        "fill_time": ft_out,
        "velocity_magnitude": vmag_out,
        "velocity": velocity_out,
        "phi": phi_out,
        "success": success,
        "final_t": float(final_t),
        "filled_fraction": float(filled),
        "steps": 1,
    }
