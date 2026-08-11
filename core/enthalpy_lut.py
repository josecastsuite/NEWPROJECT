"""Enthalpy lookup table for V8 cold-shut physics."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numba import njit


@dataclass
class EnthalpyLUT:
    """Monotonic H -> (T, fs) table."""

    H_table: np.ndarray
    T_table: np.ndarray
    fs_table: np.ndarray

    @property
    def n(self) -> int:
        return int(self.H_table.size)


def build_H_T_fs_LUT(alloy, n: int = 2000) -> EnthalpyLUT:
    """
    Build a monotonic H-based lookup table T(H) and fs(H).

    H = cp * T + (1 - fs) * L  [J/kg]

    The Scheil curve is adjusted so that the eutectic plateau starts at
    fs = 1 - f_eut and T = Te.  This avoids the T(H) multi-valued problem at
    the eutectic temperature.
    """
    cp = alloy.cp_j_kgk
    L = alloy.latent_heat_j_kg
    Tl = alloy.t_liquidus_c
    Te = alloy.t_eutectic_or_solidus_c
    k = max(min(alloy.partition_coefficient, 0.9999), 1e-6)
    f_eut = max(min(alloy.eutectic_fraction, 1.0), 0.0)

    fs = np.linspace(0.0, 1.0, n)
    T = np.empty(n, dtype=np.float64)

    fs_eut_start = 1.0 - f_eut
    exp = 1.0 - k

    # Effective solidus for the Scheil branch above the eutectic plateau.
    # We choose Ts_eff so that T(fs_eut_start) == Te.
    ts_eff = float(Te)
    if f_eut > 1e-9 and fs_eut_start > 1e-9 and Tl > Te:
        feut_exp = f_eut ** exp
        denom = 1.0 - feut_exp
        if abs(denom) > 1e-12:
            ts_eff = (Te - feut_exp * Tl) / denom
            ts_eff = min(ts_eff, Te)

    for i, f in enumerate(fs):
        if f <= fs_eut_start + 1e-12:
            ratio = (1.0 - f) ** exp
            T[i] = ts_eff + (Tl - ts_eff) * ratio
            # Clamp to the eutectic start (plateau).
            T[i] = max(T[i], Te)
        else:
            T[i] = Te

    H = cp * T + (1.0 - fs) * L

    # H is monotonically decreasing with fs; sort ascending for binary search.
    order = np.argsort(H)
    return EnthalpyLUT(H[order], T[order], fs[order])


@njit(cache=True)
def interp_H_LUT(
    H_target: float,
    H_table: np.ndarray,
    T_table: np.ndarray,
    fs_table: np.ndarray,
) -> Tuple[float, float]:
    """Binary-search interpolation in the H table."""
    n = H_table.size
    if H_target <= H_table[0]:
        return float(T_table[0]), float(fs_table[0])
    if H_target >= H_table[-1]:
        return float(T_table[-1]), float(fs_table[-1])

    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if H_table[mid] <= H_target:
            lo = mid
        else:
            hi = mid

    denom = H_table[hi] - H_table[lo]
    if denom <= 0.0:
        return float(T_table[lo]), float(fs_table[lo])
    t = (H_target - H_table[lo]) / denom
    T = T_table[lo] + t * (T_table[hi] - T_table[lo])
    fs = fs_table[lo] + t * (fs_table[hi] - fs_table[lo])
    return float(T), float(fs)
