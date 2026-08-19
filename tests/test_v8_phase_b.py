import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from core.enthalpy_lut import build_H_T_fs_LUT, compute_H_field, effective_solidus_temp, interp_H_LUT
from core.materials import get_alloy


def test_compute_H_field_eutectic_plateau():
    """AlSi12: on the eutectic plateau T stays at Te while H drops."""
    alloy = get_alloy("AlSi12")
    # M_mod = 5 mm, C = 2.8 min/cm2 gives a local solidification time of a few 10s s
    M = np.array([5.0])
    C = np.array([2.8])
    t = np.linspace(0.0, 50.0, 500)
    H, T, fs = compute_H_field(M, C, t, alloy, t_pour_c=700.0, t_mold_c=25.0)

    Te = alloy.t_eutectic_or_solidus_c
    plateau = np.abs(T - Te) < 0.5
    assert plateau.sum() > 50, "eutectic plateau not reached"
    # Temperature is essentially constant on the plateau.
    assert T[plateau].max() - T[plateau].min() < 1.0
    # Enthalpy keeps falling across the plateau.
    H_plat = H[plateau]
    assert H_plat.max() - H_plat.min() > 1e4
    # Solid fraction keeps rising on the plateau.
    assert fs[plateau].max() - fs[plateau].min() > 0.1


def test_compute_H_field_consistent_with_lut():
    """H computed directly should be recoverable through the H-based LUT."""
    alloy = get_alloy("AlSi7")
    lut = build_H_T_fs_LUT(alloy, n=1000)
    M = np.array([4.0])
    C = np.array([3.0])
    t = np.array([2.0, 8.0, 20.0, 40.0])
    H, T, fs = compute_H_field(M, C, t, alloy, t_pour_c=700.0, t_mold_c=25.0)

    for i in range(t.size):
        T_lut, fs_lut = interp_H_LUT(H[i], lut.H_table, lut.T_table, lut.fs_table)
        assert abs(T_lut - T[i]) < 2.0
        assert abs(fs_lut - fs[i]) < 0.03


def test_compute_H_field_steel_no_plateau():
    """42CrMo4 has no eutectic plateau; H and T decrease monotonically."""
    alloy = get_alloy("42CrMo4")
    M = np.array([10.0])
    C = np.array([2.8])
    t = np.linspace(0.0, 200.0, 500)
    H, T, fs = compute_H_field(M, C, t, alloy, t_pour_c=1600.0, t_mold_c=25.0)

    # H monotonically non-increasing
    assert np.all(np.diff(H) <= 1e-3)
    # T monotonically non-increasing
    assert np.all(np.diff(T) <= 1e-3)
    # fs monotonically non-decreasing
    assert np.all(np.diff(fs) >= -1e-3)
    # reaches fully solid
    assert fs.max() > 0.99


def test_effective_solidus_temp_below_liquidus():
    alloy = get_alloy("AlSi7")
    Ts_eff = effective_solidus_temp(alloy)
    assert Ts_eff < alloy.t_liquidus_c
    assert Ts_eff <= alloy.t_eutectic_or_solidus_c
