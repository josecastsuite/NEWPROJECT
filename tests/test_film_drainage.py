import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from core.film_drainage import (
    capillary_pressure,
    dynamic_pressure,
    effective_pressure,
    film_drainage_factor,
    film_thickness,
)


def test_dynamic_and_capillary_pressure():
    # AlSi7-like properties
    rho = 2660.0
    sigma = 0.9
    R = 1.0  # mm
    v = 0.1  # m/s
    Pd = dynamic_pressure(rho, v)
    cap = capillary_pressure(sigma, R)
    assert Pd < cap  # so dP_eff = 0
    assert effective_pressure(rho, v, sigma, R) == 0.0


def test_film_thickness_when_pd_less_than_capillary():
    """Fix1: if Pd <= 2*sigma/R then h_final = R and no NaN appears."""
    R_mm = 1.0
    v_rel = 0.1  # m/s
    sigma = 0.9
    rho = 2660.0
    mu = 0.0012
    t_film = 1.0
    h = film_thickness(R_mm, v_rel, sigma, rho, mu, t_film)
    # h_final should be the physical upper bound R [m]
    assert np.isfinite(h)
    assert abs(h - R_mm * 1e-3) < 1e-12


def test_film_drainage_factor_huge_thickness():
    """h >> h0 -> H_h ~ 1 (risk high, film not yet drained)."""
    h = 1e-3  # 1 mm
    Hh = film_drainage_factor(h)
    assert abs(Hh - 1.0) < 1e-3


def test_film_drainage_factor_thin_thickness():
    """h << h0 -> H_h ~ 0 (film already thin, metallic contact possible)."""
    h = 1e-11  # 0.01 nm
    Hh = film_drainage_factor(h)
    assert Hh < 0.01


def test_film_thickness_decreases_with_high_pressure():
    """When Pd > cap the film thins below R."""
    R_mm = 1.0
    sigma = 0.9
    rho = 2660.0
    mu = 0.0012
    t_film = 0.1
    # choose v so that Pd > cap
    v_rel = 5.0
    assert dynamic_pressure(rho, v_rel) > capillary_pressure(sigma, R_mm)
    h = film_thickness(R_mm, v_rel, sigma, rho, mu, t_film)
    assert h < R_mm * 1e-3
    assert h > 0.0
