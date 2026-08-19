import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.confluence_reeb import find_saddles_watershed
from core.confluence_sfer import compute_sfer_risk
from core.enthalpy_lut import build_H_T_fs_LUT
from core.materials import get_alloy
from core.sphere_lut import get_sphere_64_6


def test_find_saddles_watershed_two_basins():
    """Watershed fallback: two well-separated minima should give one persistent saddle."""
    shape = (40, 40, 40)
    ft = np.full(shape, 100.0, dtype=np.float64)
    part = np.zeros(shape, dtype=bool)
    part[5:35, 5:35, 5:35] = True

    x, y, z = np.indices(shape)
    d1 = np.sqrt((x - 15) ** 2 + (y - 20) ** 2 + (z - 20) ** 2)
    d2 = np.sqrt((x - 25) ** 2 + (y - 20) ** 2 + (z - 20) ** 2)
    ft[part] = np.minimum(d1[part], d2[part]) * 0.5

    from core.confluence_reeb import find_saddles_watershed
    coords, vals, pers = find_saddles_watershed(ft, part, persistence_thresh_s=0.1)
    assert coords.shape[0] >= 1, f"expected at least one saddle, got {coords.shape[0]}"
    assert np.all(pers > 0.1)
    assert 18 <= coords[0, 0] <= 22
    assert 18 <= coords[0, 1] <= 22
    assert 18 <= coords[0, 2] <= 22


def test_find_saddles_sublevel_merge_tree():
    """0-D sublevel persistence: two fronts meeting gives a saddle with persistence > 0."""
    from core.confluence_reeb import find_saddles_sublevel
    shape = (40, 40, 40)
    ft = np.full(shape, 100.0, dtype=np.float64)
    part = np.zeros(shape, dtype=bool)
    part[5:35, 5:35, 5:35] = True

    # two basins filling from opposite x-sides; they meet in the middle.
    x, y, z = np.indices(shape)
    d1 = np.sqrt((x - 12) ** 2 + (y - 20) ** 2 + (z - 20) ** 2)
    d2 = np.sqrt((x - 28) ** 2 + (y - 20) ** 2 + (z - 20) ** 2)
    ft[part] = np.minimum(d1[part], d2[part]) * 0.5

    coords, vals, pers = find_saddles_sublevel(ft, part, persistence_thresh_s=0.1, max_saddles=100)
    assert coords.shape[0] >= 1, f"expected at least one saddle, got {coords.shape[0]}"
    assert np.all(pers > 0.1)
    # saddle should be around the meeting plane x ~ 20
    assert 18 <= coords[0, 0] <= 22


def test_compute_sfer_risk_runs():
    """Smoke test for the Numba SFER kernel on a tiny synthetic part."""
    alloy = get_alloy("AlSi12")
    lut = build_H_T_fs_LUT(alloy, n=200)
    dirs, adj = get_sphere_64_6()

    shape = (24, 24, 24)
    part = np.zeros(shape, dtype=bool)
    part[4:20, 4:20, 4:20] = True
    ft = np.full(shape, 100.0, dtype=np.float64)
    x, y, z = np.indices(shape)
    # two well-separated minima so sigma=1.0 smoothing does not merge them
    d1 = np.sqrt((x - 8) ** 2 + (y - 12) ** 2 + (z - 12) ** 2)
    d2 = np.sqrt((x - 16) ** 2 + (y - 12) ** 2 + (z - 12) ** 2)
    ft[part] = np.minimum(d1[part], d2[part]) * 0.5

    coords, vals, pers = find_saddles_watershed(ft, part, persistence_thresh_s=0.05)
    assert coords.shape[0] >= 1

    H_field = np.full(shape, alloy.cp_j_kgk * alloy.t_pour_c + alloy.latent_heat_j_kg, dtype=np.float64)
    M_mod = np.full(shape, 5.0, dtype=np.float64)
    sdf = np.full(shape, 10.0, dtype=np.float64)
    C_field = np.full(shape, 2.8, dtype=np.float64)
    v_front = np.full(shape, 0.1, dtype=np.float64)
    velocity = np.zeros((3,) + shape, dtype=np.float64)
    velocity[0] = 0.1
    dist_feeder = np.full(shape, 100.0, dtype=np.float64)

    risk_cs, risk_lap, diagnostics = compute_sfer_risk(
        coords, ft, H_field, M_mod, sdf, C_field, v_front, velocity,
        dist_feeder, part, lut, dirs, adj, dx=1.0, alloy=alloy,
    )
    assert risk_cs.shape == shape
    assert risk_lap.shape == shape
    assert len(diagnostics) == coords.shape[0]
    # at least one saddle should produce non-zero cold-shot or lap risk
    assert risk_cs.max() > 0.0 or risk_lap.max() > 0.0
