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
    fs = 1 - f_eut and T = Te.  Superheated liquid (T > Tl, fs = 0) and
    fully-solid post-solid cooling (T < Te, fs = 1) are appended so the LUT
    covers the whole enthalpy range used by compute_H_field.
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

    # Superheat branch: T from Tl to Tl+200, fs = 0 (exclude Tl to avoid
    # duplicating the liquidus point already present in the Scheil branch).
    n_super = max(n // 10, 50)
    T_super = np.linspace(Tl, Tl + 200.0, n_super + 1)[1:]
    fs_super = np.zeros_like(T_super)
    H_super = cp * T_super + L

    # Post-solid branch: T from Te down to 20 C, fs = 1 (exclude Te to avoid
    # duplicating the fully-solid point already present in the Scheil branch).
    n_sub = max(n // 10, 50)
    T_sub = np.linspace(Te, 20.0, n_sub + 1)[:-1]
    fs_sub = np.ones_like(T_sub)
    H_sub = cp * T_sub

    H_all = np.concatenate([H, H_super, H_sub])
    T_all = np.concatenate([T, T_super, T_sub])
    fs_all = np.concatenate([fs, fs_super, fs_sub])

    # H is monotonically decreasing with fs on the Scheil/plateau branch;
    # the superheat/sub-solid branches extend the range.  Sort ascending and
    # remove any remaining duplicate H entries (e.g. for f_eut=0 plateau).
    order = np.argsort(H_all)
    H_all = H_all[order]
    T_all = T_all[order]
    fs_all = fs_all[order]
    keep = np.ones(H_all.size, dtype=bool)
    keep[1:] = H_all[1:] > H_all[:-1] + 1e-9
    return EnthalpyLUT(H_all[keep], T_all[keep], fs_all[keep])


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


def effective_solidus_temp(alloy) -> float:
    """
    Fictitious solidus for the Scheil branch above the eutectic plateau.

    The Scheil curve is forced to reach fs = 1 - f_eut exactly at T = Te,
    which is the start of the eutectic plateau.
    """
    Tl = alloy.t_liquidus_c
    Te = alloy.t_eutectic_or_solidus_c
    k = max(min(alloy.partition_coefficient, 0.9999), 1e-6)
    f_eut = max(min(alloy.eutectic_fraction, 1.0), 0.0)
    fs_eut_start = 1.0 - f_eut
    if f_eut > 1e-9 and fs_eut_start > 1e-9 and Tl > Te:
        exp = 1.0 - k
        feut_exp = f_eut ** exp
        denom = 1.0 - feut_exp
        if abs(denom) > 1e-12:
            return min((Te - feut_exp * Tl) / denom, Te)
    return float(Te)


def compute_H_field(
    M_mod: np.ndarray,
    C_field: np.ndarray,
    fill_time: np.ndarray,
    alloy,
    t_pour_c: float = None,
    t_mold_c: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-voxel meeting enthalpy H = cp*T + (1 - fs)*L [J/kg].

    The local solidification time is obtained from the Chvorinov rule using
    the Steiner-corrected modulus (M_mod) and the local mould constant
    (C_field).  T(t) is modelled as:

        - superheat removal: T from T_pour to T_liq, fs = 0
        - primary Scheil solidification: T from T_liq to Te
        - eutectic plateau: T = Te, fs from 1 - f_eut to 1
        - post-solid cooling: T from Te to T_mold, fs = 1

    This direct calculation preserves the eutectic plateau: T stays at Te
    while H keeps falling as the remaining eutectic liquid solidifies.
    """
    if t_pour_c is None:
        t_pour_c = alloy.t_pour_c
    cp = alloy.cp_j_kgk
    L = alloy.latent_heat_j_kg
    Tl = alloy.t_liquidus_c
    Te = alloy.t_eutectic_or_solidus_c
    k = max(min(alloy.partition_coefficient, 0.9999), 1e-6)
    f_eut = max(min(alloy.eutectic_fraction, 1.0), 0.0)
    Ts_eff = effective_solidus_temp(alloy)

    # Broadcast scalar/constant M_mod and C_field to the fill_time grid.
    t = np.asarray(fill_time, dtype=np.float64)
    M_mod = np.broadcast_to(np.asarray(M_mod, dtype=np.float64), t.shape)
    C_field = np.broadcast_to(np.asarray(C_field, dtype=np.float64), t.shape)

    # Local Chvorinov solidification time [s].
    M_cm = M_mod / 10.0
    t_solid = C_field * 60.0 * (M_cm * M_cm)
    t_solid = np.maximum(t_solid, 1e-9)

    t = np.where(np.isfinite(t), t, 0.0)

    H_pour = cp * t_pour_c + L
    H_solid = cp * Te
    denom = H_pour - H_solid
    denom = np.where(denom > 1e-9, denom, 1.0)

    # Time when the liquidus is reached (superheat removed).
    t_liq_start = t_solid * (cp * (t_pour_c - Tl)) / denom
    # Time when the eutectic plateau starts (primary solidification done).
    t_eut_start = t_solid * (cp * (t_pour_c - Te) + (1.0 - f_eut) * L) / denom
    t_liq_start = np.clip(t_liq_start, 0.0, t_solid)
    t_eut_start = np.clip(t_eut_start, t_liq_start, t_solid)

    T = np.full(t.shape, t_pour_c, dtype=np.float64)
    fs = np.zeros(t.shape, dtype=np.float64)

    # 1. Superheat removal: liquid, fs = 0, T drops linearly to Tl.
    mask_liq = t <= t_liq_start
    safe_liq = (t_liq_start > 1e-12) & mask_liq
    u_liq = np.zeros_like(t)
    u_liq[safe_liq] = t[safe_liq] / t_liq_start[safe_liq]
    u_liq = np.clip(u_liq, 0.0, 1.0)
    T[mask_liq] = t_pour_c - (t_pour_c - Tl) * u_liq[mask_liq]
    fs[mask_liq] = 0.0

    # 2. Primary solidification: Scheil curve from Tl down to Te.
    mask_prim = (~mask_liq) & (t < t_eut_start)
    delta_prim = np.where(t_eut_start > t_liq_start, t_eut_start - t_liq_start, 1e-9)
    u = np.clip((t - t_liq_start) / delta_prim, 0.0, 1.0)
    fs_prim = (1.0 - f_eut) * u
    ratio = np.power(np.clip(1.0 - fs_prim, 1e-12, 1.0), 1.0 - k)
    T_prim = Ts_eff + (Tl - Ts_eff) * ratio
    T_prim = np.clip(T_prim, Te, Tl)
    T[mask_prim] = T_prim[mask_prim]
    fs[mask_prim] = fs_prim[mask_prim]

    # 3. Eutectic plateau: T = Te, fs rises from 1 - f_eut to 1.
    mask_eut = t >= t_eut_start
    delta_eut = np.where(t_solid > t_eut_start, t_solid - t_eut_start, 1e-9)
    v = np.clip((t - t_eut_start) / delta_eut, 0.0, 1.0)
    fs_eut = (1.0 - f_eut) + f_eut * v
    T[mask_eut] = Te
    fs[mask_eut] = fs_eut[mask_eut]

    # 4. Post-solid cooling: fs = 1, T drops from Te to T_mold.
    mask_post = (t > t_solid) & mask_eut
    if np.any(mask_post):
        u_post = np.clip((t[mask_post] - t_solid[mask_post]) / (t_solid[mask_post] + 1e-9), 0.0, 1.0)
        T_post = Te + (t_mold_c - Te) * u_post
        T[mask_post] = np.clip(T_post, t_mold_c, Te)
        fs[mask_post] = 1.0

    H = cp * T + (1.0 - fs) * L
    return H, T, fs
