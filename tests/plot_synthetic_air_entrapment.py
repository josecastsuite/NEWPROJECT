"""Generate slice images of the synthetic air-entrapment test cases."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.filling_solver import _compute_air_entrapment_risk
from core.types import BodyType


def make_closed_pocket(shape=(24, 24, 24), dx_m=0.001):
    grid = np.full(shape, int(BodyType.EMPTY), dtype=np.int16)
    phi = np.zeros(shape, dtype=np.float64)
    fill_time = np.zeros(shape, dtype=np.float64)
    cavity = np.zeros(shape, dtype=bool)
    cavity[2:22, 2:22, 2:22] = True
    pocket = np.zeros(shape, dtype=bool)
    pocket[8:13, 8:13, 8:13] = True
    metal = cavity & ~pocket
    phi[metal] = 1.0
    phi[pocket] = 0.0
    grid[metal] = BodyType.PART
    fill_time[metal] = np.linspace(0.0, 0.01, int(metal.sum()))
    risk, _, _ = _compute_air_entrapment_risk(phi, fill_time, np.zeros(shape, bool), grid, cavity, dx_m)
    return risk, pocket, metal


def make_vented_pocket(shape=(24, 24, 24), dx_m=0.001):
    grid = np.full(shape, int(BodyType.EMPTY), dtype=np.int16)
    phi = np.zeros(shape, dtype=np.float64)
    fill_time = np.zeros(shape, dtype=np.float64)
    cavity = np.zeros(shape, dtype=bool)
    cavity[2:22, 2:22, 2:22] = True
    pocket = np.zeros(shape, dtype=bool)
    pocket[6:11, 8:15, 8:15] = True
    vent = np.zeros(shape, dtype=bool)
    vent[11:12, 8:15, 8:15] = True
    outlet_mask = vent.copy()
    air_component = pocket | vent
    metal = cavity & ~air_component
    phi[metal] = 1.0
    phi[air_component] = 0.0
    grid[metal] = BodyType.PART
    grid[vent] = BodyType.RISER
    x_grid = np.indices(shape, dtype=float)[0]
    fill_time[metal] = ((x_grid[metal] - 4.0) / 15.0) * 0.05
    fill_time[fill_time < 0] = 0.0
    fill_time[fill_time > 0.05] = 0.05
    risk, _, _ = _compute_air_entrapment_risk(phi, fill_time, outlet_mask, grid, cavity, dx_m)
    return risk, pocket, vent, metal


if __name__ == "__main__":
    out_dir = "/home/ubuntu/NEWPROJECT/NEWPROJECT-main/validation/results"
    os.makedirs(out_dir, exist_ok=True)

    risk_closed, pocket_c, metal_c = make_closed_pocket()
    risk_vent, pocket_v, vent_v, metal_v = make_vented_pocket()

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].imshow(risk_closed[10, :, :], vmin=0, vmax=1, cmap='coolwarm', origin='lower')
    ax[0].set_title('Closed pocket (risk = 1.0)')
    ax[0].set_xlabel('z'); ax[0].set_ylabel('y')
    ax[1].imshow(risk_vent[10, :, :], vmin=0, vmax=1, cmap='coolwarm', origin='lower')
    ax[1].set_title('Vented pocket (risk < 1.0)')
    ax[1].set_xlabel('z'); ax[1].set_ylabel('y')
    ax[2].imshow((risk_vent - risk_closed)[10, :, :], vmin=-1, vmax=1, cmap='coolwarm', origin='lower')
    ax[2].set_title('Difference (vent - closed)')
    ax[2].set_xlabel('z'); ax[2].set_ylabel('y')
    fig.tight_layout()
    path = os.path.join(out_dir, "synthetic_air_entrapment_slices.png")
    fig.savefig(path, dpi=120)
    print("saved", path)

    # print stats
    print("closed pocket risk:", np.unique(risk_closed[pocket_c]))
    print("vented pocket risk:", np.unique(risk_vent[pocket_v]))
