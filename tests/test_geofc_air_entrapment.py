"""Synthetic tests for the GeoFC geometric air-entrapment detector."""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.filling_solver import compute_air_entrapment_geofc
from core.types import BodyType


DX_MM = 5.0
GRAVITY = (0.0, 0.0, -1.0)
FILL_TIME_S = 0.05
Q_M3_S = 0.001


def _channel_with_pocket():
    """Main cavity along x, source at low x, riser at high x, side pocket.

    The pocket is a large chamber connected to the main channel by a single
    one-voxel throat so the vent capacity can not fully evacuate it.
    """
    # (x, y, z) axes; x is the primary flow direction.
    shape = (40, 22, 20)
    grid = np.full(shape, int(BodyType.EMPTY), dtype=np.int16)

    # Main channel along x at y=8..12, z=7..15.
    grid[4:36, 8:13, 7:15] = int(BodyType.PART)
    # Large side pocket in -y direction.
    grid[12:30, 2:7, 5:17] = int(BodyType.PART)
    # Solid wall at y=7 between pocket and main, except the one-cell throat.
    grid[12:30, 7, 5:17] = int(BodyType.EMPTY)
    grid[20, 7, 10] = int(BodyType.PART)  # single-cell throat
    # Sprue/source at low x.
    grid[1:5, 8:13, 7:15] = int(BodyType.SPRUE)
    # Riser at high x, open to atmosphere.
    grid[36:39, 8:13, 7:15] = int(BodyType.RISER)
    return grid


def test_open_riser_partial_drain():
    grid = _channel_with_pocket()
    risk, vol, cent = compute_air_entrapment_geofc(
        grid,
        np.zeros(3),
        DX_MM,
        gravity_vector=GRAVITY,
        fill_time_s=FILL_TIME_S,
        Q_m3_s=Q_M3_S,
    )
    pocket_mask = np.zeros_like(grid, dtype=bool)
    pocket_mask[12:30, 2:7, 5:17] = True
    main_mask = (grid == int(BodyType.PART)) & ~pocket_mask

    print("[open_riser] max risk:", risk.max())
    print("[open_riser] pocket max risk:", risk[pocket_mask].max())
    print("[open_riser] main max risk:", risk[main_mask].max())

    assert risk.max() > 0.0, "should detect trapped pocket"
    assert risk[pocket_mask].max() > 0.3, "pocket should have significant risk"
    assert risk[main_mask].max() < 0.3, "main channel should be largely drained"
    assert vol > 0.0


def test_closed_pocket_no_vent():
    grid = _channel_with_pocket()
    # Replace the riser with part metal so there is no vent.
    grid[grid == int(BodyType.RISER)] = int(BodyType.PART)
    risk, vol, cent = compute_air_entrapment_geofc(
        grid,
        np.zeros(3),
        DX_MM,
        gravity_vector=GRAVITY,
        fill_time_s=FILL_TIME_S,
        Q_m3_s=Q_M3_S,
    )
    print("[no_vent] max risk:", risk.max())
    print("[no_vent] trapped volume:", vol)
    assert risk.max() == 1.0, "closed cavity should be fully trapped"
    assert vol > 0.0


def test_far_riser_separated_by_wall():
    """A distant riser separated by solid (EMPTY/mold) cannot drain a pocket."""
    grid = _channel_with_pocket()
    # Put a solid wall between the main channel and the riser.
    grid[30:36, 8:13, 7:15] = int(BodyType.EMPTY)
    risk, vol, cent = compute_air_entrapment_geofc(
        grid,
        np.zeros(3),
        DX_MM,
        gravity_vector=GRAVITY,
        fill_time_s=FILL_TIME_S,
        Q_m3_s=Q_M3_S,
    )
    pocket_mask = np.zeros_like(grid, dtype=bool)
    pocket_mask[12:30, 2:7, 5:17] = True
    print("[separated_riser] pocket max risk:", risk[pocket_mask].max())
    assert risk[pocket_mask].max() > 0.9, "separated pocket must stay trapped"
    assert vol > 0.0


if __name__ == "__main__":
    test_open_riser_partial_drain()
    test_closed_pocket_no_vent()
    test_far_riser_separated_by_wall()
    print("ALL GEOFC AIR ENTRAPMENT TESTS PASSED")
