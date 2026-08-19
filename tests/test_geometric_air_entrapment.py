"""Geometric air-entrapment tests independent of LBM/VOF."""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.filling_solver import compute_geometric_air_entrapment
from core.types import BodyType


def _box(shape=(50, 50, 50)):
    grid = np.zeros(shape, dtype=np.int16)
    grid[0, :, :] = grid[-1, :, :] = grid[:, 0, :] = grid[:, -1, :] = int(BodyType.PART)
    grid[:, :, 0] = grid[:, :, -1] = int(BodyType.PART)
    return grid


def test_closed_pocket_no_vent():
    """A fully enclosed cavity with no outlet must be risk=1."""
    grid = _box()
    # hollow cube shell at centre
    grid[15:35, 15:35, 10:30] = int(BodyType.PART)
    grid[16:34, 16:34, 11:29] = int(BodyType.EMPTY)
    risk, vol, cent = compute_geometric_air_entrapment(grid, np.zeros(3), 1.0)
    assert risk.max() == 1.0, f"closed pocket risk should be 1.0, got {risk.max()}"
    assert vol > 0.0


def test_far_riser_does_not_drain_side_pocket():
    """A side pocket separated by a metal wall must not use a distant riser."""
    grid = _box()
    # main cavity ceiling
    grid[10:40, 10:40, 30] = int(BodyType.PART)
    # side pocket with walls reaching the floor and a small bottom opening
    grid[5:16, 9:10, 0:21] = int(BodyType.PART)
    grid[5:16, 40:41, 0:21] = int(BodyType.PART)
    grid[5:6, 10:40, 0:21] = int(BodyType.PART)
    grid[15:16, 10:40, 0:21] = int(BodyType.PART)
    grid[5:16, 10:40, 20:21] = int(BodyType.PART)
    grid[5, 15:18, 1:3] = 0  # bottom opening
    # distant riser column
    grid[35:38, 35:38, 30:46] = int(BodyType.RISER)

    risk, vol, cent = compute_geometric_air_entrapment(
        grid, np.zeros(3), 1.0, gravity_vector=(0.0, 0.0, -1.0)
    )
    wall_risk = risk[5:16, 9:41, 0:21][grid[5:16, 9:41, 0:21] != 0]
    assert wall_risk.max() == 1.0, f"side pocket walls should be high risk, got {wall_risk.max()}"
    assert vol > 0.0


def test_open_riser_drains_main_cavity():
    """A cavity connected to a riser should have no trapped geometric air."""
    grid = _box()
    grid[10:40, 10:40, 30] = int(BodyType.PART)
    grid[35:38, 35:38, 30:46] = int(BodyType.RISER)
    risk, vol, cent = compute_geometric_air_entrapment(
        grid, np.zeros(3), 1.0, gravity_vector=(0.0, 0.0, -1.0)
    )
    assert risk.max() == 0.0, f"vented cavity should have zero risk, got {risk.max()}"
    assert vol == 0.0


if __name__ == "__main__":
    test_closed_pocket_no_vent()
    test_far_riser_does_not_drain_side_pocket()
    test_open_riser_drains_main_cavity()
    print("ALL GEOMETRIC AIR ENTRAPMENT TESTS PASSED")
