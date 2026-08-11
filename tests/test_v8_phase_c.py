import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from core.materials import get_alloy
from core.peclet import effective_modulus_pe, peclet_correction, peclet_front_velocity_m_s


def test_peclet_correction_bounds():
    alloy = get_alloy("AlSi7")
    M = np.array([1.0, 5.0, 10.0])
    v = np.array([0.0, 0.5, 2.0])
    f = peclet_correction(
        M, v, alloy.thermal_diffusivity_m2_s, alloy.pe_flow_factor_c, alloy.pe_flow_factor_max
    )
    assert np.all(f >= 1.0)
    assert np.all(f <= alloy.pe_flow_factor_max)
    # zero velocity -> no correction
    assert abs(f[0] - 1.0) < 1e-9


def test_peclet_no_explosion_large_velocity():
    """V7 bug: high Pe could make M_eff explode; V8 clamps at Fmax."""
    alloy = get_alloy("AlSi7")
    M = np.array([5.0])
    v = np.array([10.0])  # very fast front
    M_eff = effective_modulus_pe(
        M, v, alloy.thermal_diffusivity_m2_s, alloy.pe_flow_factor_c, alloy.pe_flow_factor_max
    )
    assert np.all(M_eff <= M * alloy.pe_flow_factor_max * 1.001)
    assert np.all(M_eff >= M)


def test_peclet_front_velocity_from_fill_time():
    dx_mm = 1.0
    # fill_time gradient of 2 s/voxel -> v = dx_m / 2 = 0.0005 m/s
    ft = np.arange(10, dtype=np.float64) * 2.0
    v = peclet_front_velocity_m_s(ft, dx_mm)
    assert abs(v.mean() - 0.0005) < 1e-9


def test_peclet_higher_alpha_means_lower_correction():
    """Higher thermal diffusivity -> lower Pe -> smaller f_Pe."""
    alloy = get_alloy("AlSi7")
    M = np.array([5.0])
    v = np.array([1.0])
    f_low_alpha = peclet_correction(M, v, 1e-6, 0.08, 2.0)
    f_high_alpha = peclet_correction(M, v, 1e-4, 0.08, 2.0)
    assert f_high_alpha[0] < f_low_alpha[0]
