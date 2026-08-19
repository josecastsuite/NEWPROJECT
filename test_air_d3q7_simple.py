"""Synthetic D3Q7 advection + mass conservation test.

A spherical air bubble is advected at 1 m/s in a closed box with all
mold boundaries disabled (K_mold -> 0).  The bubble center-of-mass must
translate by exactly ``u_lb * n_steps = (dt/dx) * v * n_steps`` voxels,
and total ``alpha_g * rho_g`` mass must be conserved to near machine
precision.
"""
import numpy as np
from core.air_entrapment_d3q7 import AirEntrapmentSolver_D3Q7

nx, ny, nz = 40, 20, 20
dx = 1e-3
dt = 1e-4
shape = (nx, ny, nz)

# Localized bubble, not a full cavity, so the center of mass can move.
F = np.zeros(shape, dtype=np.float64)
cx, cy, cz = 8, ny // 2, nz // 2
r = 4
ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
mask = (ii - cx) ** 2 + (jj - cy) ** 2 + (kk - cz) ** 2 <= r**2
F[mask] = 1.0

v = np.zeros((3, *shape), dtype=np.float64)
v[0, :, :, :] = 1.0  # 1 m/s in x

nu_t = np.zeros(shape, dtype=np.float64)

solver = AirEntrapmentSolver_D3Q7(shape, dx, T_melt=993.0, L_wall=1e-3)
# Disable Darcy escape so this is a pure advection/diffusion test.
solver.is_boundary = np.zeros(shape, dtype=np.bool_)

solver.on_lbm_step(0, dt, dx, 0.0, v[0], v[1], v[2], F, nu_t)

m0 = float(solver.alpha_rho.sum())
print(f"initial mass = {m0:.6e}")

n_steps = 100
for step in [10, 20, 30, 40, 50, n_steps]:
    solver.on_lbm_step(step, dt, dx, 0.0, v[0], v[1], v[2], F, nu_t)
    m = float(solver.alpha_rho.sum())
    print(f"step {step:3d}: mass={m:.6e} rel_err={(m - m0) / m0:.3e}")

com = np.array(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"))
cog = np.sum(solver.alpha_rho * com, axis=(1, 2, 3)) / np.sum(solver.alpha_rho)
expected_cogx = cx + v[0, 0, 0, 0] * (dt / dx) * n_steps
print(f"center of mass = {cog}")
print(f"expected cog x = {expected_cogx:.2f}")

mass_ok = abs((solver.alpha_rho.sum() - m0) / m0) < 1e-9
advection_ok = abs(cog[0] - expected_cogx) < 0.5  # within half a voxel
print("PASS" if mass_ok and advection_ok else "FAIL")
