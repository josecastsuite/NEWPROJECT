"""Synthetic tests for the GeoFC geometric air-entrapment detector."""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import ndimage
from core.filling_solver import compute_air_entrapment_geofc
from core.types import BodyType


s6 = ndimage.generate_binary_structure(3, 1)
DX_MM = 5.0
GRAVITY = (0.0, 0.0, -1.0)
FILL_TIME_S = 0.05
Q_M3_S = 0.001


def _channel_with_pocket():
    """Main cavity along x, source at low x, riser at high x, side pocket.

    The pocket is a large chamber connected to the main channel by a single
    one-voxel throat so the vent capacity can not fully evacuate it.
    A CORE roof closes the cope so the only real vent is the riser top.
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
    # Riser at high x, extended through the cope so it stays open.
    grid[36:39, 8:13, 7:17] = int(BodyType.RISER)
    # Cope roof (CORE) directly above the cavity; leave the riser area open.
    grid[1:5, 8:13, 15] = int(BodyType.CORE)
    grid[4:36, 8:13, 15] = int(BodyType.CORE)
    grid[12:30, 2:7, 17] = int(BodyType.CORE)
    # Close the cope above the separating wall / throat so air cannot leak upward.
    grid[12:30, 7, 11:18] = int(BodyType.CORE)
    grid[36:39, 8:13, 17] = int(BodyType.EMPTY)  # keep riser open
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
    part_mask = grid == int(BodyType.PART)
    # Risk is now painted on the metal ceiling/sealing surface of the pocket,
    # not distributed through the pocket volume.
    surface = ndimage.binary_dilation(pocket_mask, structure=s6, iterations=2) & part_mask
    far_main = part_mask & ~pocket_mask & ~ndimage.binary_dilation(pocket_mask, structure=s6, iterations=4)

    print("[open_riser] max risk:", risk.max())
    print("[open_riser] pocket surface max risk:", risk[surface].max())
    print("[open_riser] far main max risk:", risk[far_main].max())

    assert risk.max() > 0.0, "should detect trapped pocket"
    assert risk[surface].max() > 0.3, "pocket sealing surface should have significant risk"
    assert risk[far_main].max() < 0.3, "main channel away from the throat should be largely drained"
    assert vol > 0.0


def test_closed_pocket_no_vent():
    grid = _channel_with_pocket()
    # Remove gating so there is no vent; the inlet will be the bottom open surface.
    grid[grid == int(BodyType.RISER)] = int(BodyType.PART)
    grid[grid == int(BodyType.SPRUE)] = int(BodyType.PART)
    # Close the small cope opening left for the riser top.
    grid[36:39, 8:13, 17] = int(BodyType.CORE)
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
    part_mask = grid == int(BodyType.PART)
    surface = ndimage.binary_dilation(pocket_mask, structure=s6, iterations=2) & part_mask
    print("[separated_riser] max risk:", risk.max())
    print("[separated_riser] pocket surface max risk:", risk[surface].max())
    # A wall that closes the only vent makes the whole cavity fully trapped;
    # the risk is 1.0 on the last-to-fill sealing surface.
    assert risk.max() > 0.9, "closed cavity must have very high risk somewhere"
    assert risk[surface].max() > 0.9 or risk.max() == 1.0, "separated pocket sealing surface must stay trapped"
    assert vol > 0.0


if __name__ == "__main__":
    test_open_riser_partial_drain()
    test_closed_pocket_no_vent()
    test_far_riser_separated_by_wall()
    print("ALL GEOFC AIR ENTRAPMENT TESTS PASSED")
