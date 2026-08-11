import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from core.enthalpy_lut import build_H_T_fs_LUT, interp_H_LUT
from core.materials import get_alloy, get_mold
from core.sdf_analyzer import build_effusivity_field, build_local_chvorinov_c_field, compute_steiner_modulus
from core.sphere_lut import get_sphere_64_6
from core.types import BodyType


def test_sphere_64_6():
    dirs, adj = get_sphere_64_6()
    assert dirs.shape == (64, 3)
    assert adj.shape == (64, 6)
    # unit normals
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-12)
    # adjacency indices are distinct from the source direction
    for i in range(64):
        assert i not in adj[i]


def test_enthalpy_lut_eutectic_plateau():
    """AlSi7 has f_eut>0; the plateau must give Te over a range of H."""
    alloy = get_alloy("AlSi7")
    lut = build_H_T_fs_LUT(alloy, n=500)
    assert lut.H_table.size == 500
    assert np.all(np.diff(lut.H_table) > 0)
    # at the liquid end T ~ Tl, fs ~ 0
    H_liq = lut.H_table[-1]
    T, fs = interp_H_LUT(H_liq, lut.H_table, lut.T_table, lut.fs_table)
    assert abs(T - alloy.t_liquidus_c) < 1.0
    assert abs(fs) < 0.05
    # at the solid end T ~ Te, fs ~ 1
    H_sol = lut.H_table[0]
    T, fs = interp_H_LUT(H_sol, lut.H_table, lut.T_table, lut.fs_table)
    assert abs(T - alloy.t_eutectic_or_solidus_c) < 1.0
    assert abs(fs - 1.0) < 0.05
    # plateau: many points share Te
    te_count = np.sum(np.abs(lut.T_table - alloy.t_eutectic_or_solidus_c) < 0.5)
    assert te_count > 10


def test_enthalpy_lut_steel_no_plateau():
    alloy = get_alloy("42CrMo4")
    lut = build_H_T_fs_LUT(alloy, n=500)
    assert np.all(np.diff(lut.H_table) > 0)
    Tl = alloy.t_liquidus_c
    Ts = alloy.t_solidus_c
    T_top, _ = interp_H_LUT(lut.H_table[-1], lut.H_table, lut.T_table, lut.fs_table)
    T_bot, _ = interp_H_LUT(lut.H_table[0], lut.H_table, lut.T_table, lut.fs_table)
    assert abs(T_top - Tl) < 1.0
    assert abs(T_bot - Ts) < 1.0


def test_steiner_modulus():
    sdf = np.ones((10, 10, 10), dtype=np.float64) * 5.0
    mean_curv = np.zeros_like(sdf)
    gauss = np.zeros_like(sdf)
    M = compute_steiner_modulus(sdf, mean_curv, gauss)
    # flat plate -> M = SDF
    np.testing.assert_allclose(M, sdf)

    # concave spherical-like region (positive mean, positive gauss)
    # shape_factor < 1 -> M > SDF
    mean_curv[:] = 0.1
    gauss[:] = 0.01
    M = compute_steiner_modulus(sdf, mean_curv, gauss)
    assert np.all(M > sdf)


def test_local_c_and_effusivity_fields():
    grid = np.zeros((20, 20, 20), dtype=np.int32)
    grid[5:15, 5:15, 5:15] = int(BodyType.PART)
    body_index = np.full(grid.shape, -1, dtype=np.int32)
    alloy = get_alloy("AlSi7")
    mold = get_mold("sand")
    C = build_local_chvorinov_c_field(grid, body_index, [], mold, alloy)
    e = build_effusivity_field(grid, body_index, [], mold)
    assert C.shape == grid.shape
    assert e.shape == grid.shape
    # metal voxels should inherit nearest non-metal C
    part_mask = grid == BodyType.PART
    assert np.all(C[part_mask] == C[0, 0, 0])
    assert np.all(e[part_mask] == e[0, 0, 0])
