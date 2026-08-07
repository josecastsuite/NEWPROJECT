"""Direct synthetic tests for core/filling_solver.py _compute_air_entrapment_risk."""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import ndimage
from core.filling_solver import _compute_air_entrapment_risk
from core.types import BodyType


s6 = ndimage.generate_binary_structure(3, 1)


def _make_base(shape=(24, 24, 24), dx_m=0.001):
    grid = np.full(shape, int(BodyType.EMPTY), dtype=np.int16)
    phi = np.zeros(shape, dtype=np.float64)
    fill_time = np.zeros(shape, dtype=np.float64)
    cavity = np.zeros(shape, dtype=bool)
    # define a cavity away from the domain boundary
    cavity[2:22, 2:22, 2:22] = True
    return grid, phi, fill_time, cavity


def _check_shapes(*args):
    shape = args[0].shape
    for a in args:
        assert a.shape == shape
    return shape


def test_closed_pocket_no_vent():
    grid, phi, fill_time, cavity = _make_base()
    shape = _check_shapes(grid, phi, fill_time, cavity)

    # A 5x5x5 air pocket in the centre, surrounded by metal.
    pocket = np.zeros(shape, dtype=bool)
    pocket[8:13, 8:13, 8:13] = True

    metal = cavity & ~pocket
    phi[metal] = 1.0          # metal
    phi[pocket] = 0.0         # empty air pocket
    grid[metal] = BodyType.PART

    # Provide a non-zero escape window so the test isolates the "no vent" path.
    fill_time[metal] = np.linspace(0.0, 0.01, int(metal.sum()))
    outlet_mask = np.zeros(shape, dtype=bool)

    risk, trapped_volume, _ = _compute_air_entrapment_risk(
        phi, fill_time, outlet_mask, grid, cavity, 0.001
    )

    # Risk is now written on the metal ceiling that seals the pocket, not inside
    # the empty pocket, because air entrapment is a surface/interface defect.
    metal = phi >= 0.5
    surface = ndimage.binary_dilation(pocket, structure=s6, iterations=1) & metal
    surface_risk = risk[surface]
    print("[test_closed_pocket_no_vent] surface risk values:", np.unique(surface_risk))
    print("[test_closed_pocket_no_vent] trapped_volume_m3:", trapped_volume)
    assert surface_risk.size > 0
    assert surface_risk.max() > 0.99, f"closed pocket ceiling risk should be ~1.0, got {surface_risk.max()}"
    assert trapped_volume > 0.0


def test_left_pocket_far_right_riser():
    grid, phi, fill_time, cavity = _make_base()
    shape = _check_shapes(grid, phi, fill_time, cavity)

    # Left-side closed air pocket (x = 4..8)
    pocket = np.zeros(shape, dtype=bool)
    pocket[4:9, 8:15, 8:15] = True

    # A RISER outlet far on the right (x = 14..18), separated by a 2-voxel metal wall.
    riser = np.zeros(shape, dtype=bool)
    riser[14:19, 8:15, 8:15] = True
    outlet_mask = riser.copy()

    metal = cavity & ~pocket & ~riser
    phi[metal] = 1.0
    phi[pocket] = 0.0
    phi[riser] = 0.0
    grid[metal] = BodyType.PART
    grid[riser] = BodyType.RISER

    fill_time[metal] = np.linspace(0.0, 0.01, int(metal.sum()))

    risk, trapped_volume, _ = _compute_air_entrapment_risk(
        phi, fill_time, outlet_mask, grid, cavity, 0.001
    )

    metal = phi >= 0.5
    surface = ndimage.binary_dilation(pocket, structure=s6, iterations=1) & metal
    surface_risk = risk[surface]
    print("[test_left_pocket_far_right_riser] surface risk values:", np.unique(surface_risk))
    print("[test_left_pocket_far_right_riser] trapped_volume_m3:", trapped_volume)
    assert surface_risk.size > 0
    assert surface_risk.max() > 0.99, f"geometric lock should keep ceiling risk ~1.0, got {surface_risk.max()}"
    assert trapped_volume > 0.0


def test_open_vent_reduces_risk():
    grid, phi, fill_time, cavity = _make_base()
    shape = _check_shapes(grid, phi, fill_time, cavity)

    # Pocket plus a directly adjacent 1-voxel-thick RISER vent slab.
    pocket = np.zeros(shape, dtype=bool)
    pocket[6:11, 8:15, 8:15] = True
    vent = np.zeros(shape, dtype=bool)
    vent[11:12, 8:15, 8:15] = True  # single-voxel slab touching the pocket
    outlet_mask = vent.copy()

    air_component = pocket | vent
    metal = cavity & ~air_component
    phi[metal] = 1.0
    phi[air_component] = 0.0
    grid[metal] = BodyType.PART
    grid[vent] = BodyType.RISER
    # grid for the pocket stays EMPTY (0)

    # Fill time gradient across the metal so t_window is 0.05 s.
    # x coordinate drives the gradient.
    fill_time[:] = 0.0
    x_grid = np.indices(shape, dtype=float)[0]
    fill_time[metal] = ((x_grid[metal] - 4.0) / 15.0) * 0.05
    fill_time[fill_time < 0] = 0.0
    fill_time[fill_time > 0.05] = 0.05

    risk, trapped_volume, _ = _compute_air_entrapment_risk(
        phi, fill_time, outlet_mask, grid, cavity, 0.001
    )

    metal = phi >= 0.5
    surface = ndimage.binary_dilation(air_component, structure=s6, iterations=1) & metal
    surface_risk = risk[surface]
    print("[test_open_vent_reduces_risk] pocket+vent surface risk values:", np.unique(surface_risk))
    print("[test_open_vent_reduces_risk] trapped_volume_m3:", trapped_volume)
    assert surface_risk.size > 0
    assert surface_risk.max() < 1.0, f"open vent should reduce ceiling risk below 1.0, got max={surface_risk.max()}"
    assert trapped_volume >= 0.0


if __name__ == "__main__":
    test_closed_pocket_no_vent()
    test_left_pocket_far_right_riser()
    test_open_vent_reduces_risk()
    print("ALL SYNTHETIC AIR ENTRAPMENT TESTS PASSED")
