"""V8 SFER (Spherical Front Encounter Rate) cold-shot / lap risk kernel.

Fix3, Fix4 and the saddle loop from V8.  For each saddle found by the Reeb/
watershed detector, 64 directions on a sphere are sampled to find converging
fill fronts and compute the local cold-shut / lap risk.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit, prange

from core.film_drainage import _film_drainage_factor_numba, _film_thickness_numba


@njit(cache=True)
def _interp_H_LUT_scalar(H: float, H_table: np.ndarray, T_table: np.ndarray, fs_table: np.ndarray) -> Tuple[float, float]:
    """1-D binary search interpolation in an ascending H table."""
    n = H_table.size
    if H <= H_table[0]:
        return T_table[0], fs_table[0]
    if H >= H_table[-1]:
        return T_table[-1], fs_table[-1]
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if H_table[mid] < H:
            lo = mid
        else:
            hi = mid
    dh = H_table[hi] - H_table[lo]
    if abs(dh) < 1e-12:
        return T_table[lo], fs_table[lo]
    w = (H - H_table[lo]) / dh
    T = T_table[lo] + w * (T_table[hi] - T_table[lo])
    fs = fs_table[lo] + w * (fs_table[hi] - fs_table[lo])
    return T, fs


@njit(cache=True)
def _sfer64_one_saddle(
    i: int,
    j: int,
    k: int,
    ft: np.ndarray,
    H_field: np.ndarray,
    M_mod: np.ndarray,
    sdf: np.ndarray,
    C_field: np.ndarray,
    v_front: np.ndarray,
    velocity: np.ndarray,
    dist_feeder: np.ndarray,
    part_mask: np.ndarray,
    H_table: np.ndarray,
    T_table: np.ndarray,
    fs_table: np.ndarray,
    dirs: np.ndarray,
    adj: np.ndarray,
    shape: Tuple[int, int, int],
    dx: float,
    rho: float,
    sigma: float,
    mu: float,
    cp: float,
    L: float,
    alpha: float,
    c_pe: float,
    Fmax: float,
    Tl: float,
    Te: float,
    fs_crit: float,
    dt_crit: float,
    fs_crit_lap: float,
    dt_crit_lap: float,
    dT_crit: float,
    we_crit: float,
    h0: float,
    feed_k1: float,
    C_ref: float,
) -> Tuple[float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float]:
    """
    Compute cold-shot risk, lap risk, and dominant angle for one saddle.

    Returns (risk_cs, risk_lap, theta_deg, h_final_m, T_int_c, fs_int,
             We, Pe, M_eff_mm, dt_s, v_rel_m_s, N_front,
             d1x, d1y, d1z, d2x, d2y, d2z).
    """
    nx, ny, nz = shape
    if not part_mask[i, j, k]:
        return (
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )

    R = M_mod[i, j, k]
    three_dx = 3.0 * dx
    if R < three_dx:
        R = three_dx
    max_R = sdf[i, j, k] * 0.8
    if max_R < three_dx:
        max_R = three_dx
    if R > max_R:
        R = max_R

    R_m = R * 1e-3
    n_dir = dirs.shape[0]

    # sample fill time, enthalpy, front speed and velocity vector along each direction
    ft_s = np.empty(n_dir, dtype=np.float64)
    H_s = np.empty(n_dir, dtype=np.float64)
    vf_s = np.empty(n_dir, dtype=np.float64)
    vx_s = np.empty(n_dir, dtype=np.float64)
    vy_s = np.empty(n_dir, dtype=np.float64)
    vz_s = np.empty(n_dir, dtype=np.float64)
    valid = np.empty(n_dir, dtype=np.bool_)

    for s in range(n_dir):
        di = dirs[s, 0]
        dj = dirs[s, 1]
        dk = dirs[s, 2]
        ni = int(round(i + di * R / dx))
        nj = int(round(j + dj * R / dx))
        nk = int(round(k + dk * R / dx))
        if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
            valid[s] = False
            continue
        if not part_mask[ni, nj, nk]:
            valid[s] = False
            continue
        valid[s] = True
        ft_s[s] = ft[ni, nj, nk]
        H_s[s] = H_field[ni, nj, nk]
        vf_s[s] = v_front[ni, nj, nk]
        vx_s[s] = velocity[0, ni, nj, nk]
        vy_s[s] = velocity[1, ni, nj, nk]
        vz_s[s] = velocity[2, ni, nj, nk]

    # Choose the two valid directions on the sphere that are (a) lowest in
    # fill time and (b) most separated in angle.  This handles saddles (two
    # basins meeting) and maxima / closure points (one source band closing).
    saddle_ft = ft[i, j, k]
    best_s1 = -1
    best_s2 = -1
    best_score = 1e300
    best_cos = 1.0
    for s in range(n_dir):
        if not valid[s]:
            continue
        for t in range(s + 1, n_dir):
            if not valid[t]:
                continue
            cos_theta = (
                dirs[s, 0] * dirs[t, 0]
                + dirs[s, 1] * dirs[t, 1]
                + dirs[s, 2] * dirs[t, 2]
            )
            cos_theta = max(-1.0, min(1.0, cos_theta))
            # low ft_s and large angle (small 1 - cos) preferred
            separation = 1.0 - cos_theta
            if separation < 1e-6:
                continue
            score = (ft_s[s] + ft_s[t]) / separation
            if score < best_score:
                best_score = score
                best_s1 = s
                best_s2 = t
                best_cos = cos_theta

    if best_s1 < 0:
        return (
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )

    # precompute front speed for Péclet / M_eff at the saddle
    v_front_saddle = v_front[i, j, k]
    M_mm = M_mod[i, j, k]
    M_m = M_mm * 1e-3
    Pe = v_front_saddle * M_m / (alpha + 1e-12)
    if Pe < 0.0:
        Pe = 0.0
    sqrt_Pe = np.sqrt(Pe)
    f_pe = 1.0 + min(c_pe * sqrt_Pe, Fmax - 1.0)
    if f_pe < 1.0:
        f_pe = 1.0
    if f_pe > Fmax:
        f_pe = Fmax

    # feeder factor: closer to a feeder => lower risk
    dist_f = dist_feeder[i, j, k]
    L_feed = feed_k1 * M_mm * (C_field[i, j, k] / C_ref)
    if L_feed < 1e-6:
        L_feed = 1e-6
    f_feeder = 1.0 - 0.8 * np.exp(-dist_f / L_feed)
    if f_feeder < 0.2:
        f_feeder = 0.2
    if f_feeder > 1.0:
        f_feeder = 1.0

    M_eff = M_mm * f_pe * f_feeder

    # local solidification time for film drainage
    C_loc = C_field[i, j, k]
    t_solid = C_loc * 60.0 * (M_eff / 10.0) * (M_eff / 10.0)
    if t_solid < 1e-9:
        t_solid = 1e-9

    # Evaluate the dominant incoming-front pair.
    s1 = best_s1
    s2 = best_s2
    d1x = dirs[s1, 0]
    d1y = dirs[s1, 1]
    d1z = dirs[s1, 2]
    d2x = dirs[s2, 0]
    d2y = dirs[s2, 1]
    d2z = dirs[s2, 2]
    theta = np.arccos(best_cos)
    theta_deg = theta * 180.0 / np.pi

    v1_dot = vx_s[s1] * d1x + vy_s[s1] * d1y + vz_s[s1] * d1z
    v2_dot = vx_s[s2] * d2x + vy_s[s2] * d2y + vz_s[s2] * d2z
    v_rel = v1_dot - v2_dot
    if v_rel < 0.0:
        v_rel = -v_rel

    # film drainage
    dt = abs(ft_s[s1] - ft_s[s2])
    t_film = dt if dt < t_solid else t_solid
    h_final = _film_thickness_numba(R, v_rel, sigma, rho, mu, t_film)
    Hh = _film_drainage_factor_numba(h_final, h0)

    # Weber / inertial rupture of oxide film
    We = rho * v_rel * v_rel * R_m / (sigma + 1e-12)
    risk_We = 1.0 / (1.0 + np.exp(5.0 * (We - we_crit)))

    # enthalpy / solid fraction at the encounter (conservative: lower H)
    H_int = H_s[s1]
    if H_s[s2] < H_int:
        H_int = H_s[s2]
    T_int, fs_int = _interp_H_LUT_scalar(H_int, H_table, T_table, fs_table)

    # geometric factor: 1 for head-on, 0 for small angles
    geom = (1.0 - best_cos) / 2.0

    max_cs = 0.0
    max_lap = 0.0
    max_theta = 0.0
    h_final_out = 0.0

    # cold shut (>= 120 deg)
    if theta_deg >= 120.0:
        arg_dt = -10.0 * (dt - dt_crit)
        H_dt = 1.0 / (1.0 + np.exp(arg_dt))
        arg_T = -1.0 * ((Tl - T_int) - dT_crit)
        H_T = 1.0 / (1.0 + np.exp(arg_T))
        arg_fs = -20.0 * (fs_int - fs_crit)
        H_fs = 1.0 / (1.0 + np.exp(arg_fs))
        max_cs = H_dt * H_T * H_fs * Hh * risk_We * geom * f_feeder
        max_theta = theta_deg
        h_final_out = h_final

    # lap (45 - 120 deg)
    if 45.0 <= theta_deg < 120.0:
        arg_dt_lap = -10.0 * (dt - dt_crit_lap)
        H_dt_lap = 1.0 / (1.0 + np.exp(arg_dt_lap))
        arg_fs_lap = -20.0 * (fs_int - fs_crit_lap)
        H_fs_lap = 1.0 / (1.0 + np.exp(arg_fs_lap))
        max_lap = H_dt_lap * H_fs_lap * Hh * geom * f_feeder

    N_front = 0
    for s in range(n_dir):
        if valid[s]:
            N_front += 1

    return (
        max_cs, max_lap, max_theta, h_final_out,
        float(T_int), float(fs_int), float(We), float(Pe), float(M_eff),
        float(dt), float(v_rel), float(N_front),
        float(dirs[best_s1, 0]), float(dirs[best_s1, 1]), float(dirs[best_s1, 2]),
        float(dirs[best_s2, 0]), float(dirs[best_s2, 1]), float(dirs[best_s2, 2]),
    )


@njit(parallel=True, cache=True)
def _sfer64_kernel(
    saddles: np.ndarray,
    ft: np.ndarray,
    H_field: np.ndarray,
    M_mod: np.ndarray,
    sdf: np.ndarray,
    C_field: np.ndarray,
    v_front: np.ndarray,
    velocity: np.ndarray,
    dist_feeder: np.ndarray,
    part_mask: np.ndarray,
    H_table: np.ndarray,
    T_table: np.ndarray,
    fs_table: np.ndarray,
    dirs: np.ndarray,
    adj: np.ndarray,
    shape: Tuple[int, int, int],
    dx: float,
    rho: float,
    sigma: float,
    mu: float,
    cp: float,
    L: float,
    alpha: float,
    c_pe: float,
    Fmax: float,
    Tl: float,
    Te: float,
    fs_crit: float,
    dt_crit: float,
    fs_crit_lap: float,
    dt_crit_lap: float,
    dT_crit: float,
    we_crit: float,
    h0: float,
    feed_k1: float,
    C_ref: float,
    risk_cs: np.ndarray,
    risk_lap: np.ndarray,
    saddle_info: np.ndarray,
) -> None:
    """Parallel SFER over all saddles.  ``saddle_info`` is (N,18) for diagnostics."""
    n = saddles.shape[0]
    for idx in prange(n):
        i = int(saddles[idx, 0])
        j = int(saddles[idx, 1])
        k = int(saddles[idx, 2])
        (
            cs, lap, theta, h_final,
            T_int, fs_int, We, Pe, M_eff, dt, v_rel, N_front,
            d1x, d1y, d1z, d2x, d2y, d2z,
        ) = _sfer64_one_saddle(
            i, j, k,
            ft, H_field, M_mod, sdf, C_field, v_front, velocity, dist_feeder,
            part_mask, H_table, T_table, fs_table, dirs, adj,
            shape, dx, rho, sigma, mu, cp, L, alpha, c_pe, Fmax,
            Tl, Te, fs_crit, dt_crit, fs_crit_lap, dt_crit_lap,
            dT_crit, we_crit, h0, feed_k1, C_ref,
        )
        if cs > risk_cs[i, j, k]:
            risk_cs[i, j, k] = cs
        if lap > risk_lap[i, j, k]:
            risk_lap[i, j, k] = lap
        saddle_info[idx, 0] = cs
        saddle_info[idx, 1] = lap
        saddle_info[idx, 2] = theta
        saddle_info[idx, 3] = h_final
        saddle_info[idx, 4] = T_int
        saddle_info[idx, 5] = fs_int
        saddle_info[idx, 6] = We
        saddle_info[idx, 7] = Pe
        saddle_info[idx, 8] = M_eff
        saddle_info[idx, 9] = dt
        saddle_info[idx, 10] = v_rel
        saddle_info[idx, 11] = float(N_front)
        saddle_info[idx, 12] = d1x
        saddle_info[idx, 13] = d1y
        saddle_info[idx, 14] = d1z
        saddle_info[idx, 15] = d2x
        saddle_info[idx, 16] = d2y
        saddle_info[idx, 17] = d2z


def compute_sfer_risk(
    saddles: np.ndarray,
    ft: np.ndarray,
    H_field: np.ndarray,
    M_mod: np.ndarray,
    sdf: np.ndarray,
    C_field: np.ndarray,
    v_front: np.ndarray,
    velocity: np.ndarray,
    dist_feeder: np.ndarray,
    part_mask: np.ndarray,
    lut,
    dirs: np.ndarray,
    adj: np.ndarray,
    dx: float,
    alloy,
    h0: float = 100e-9,
    C_ref: float = 2.8,
    saddle_persistence: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """
    Compute V8 SFER cold-shot / lap risk for a list of saddle points.

    Returns per-voxel ``risk_cs`` and ``risk_lap`` arrays plus a list of
    saddle diagnostics.
    """
    shape = ft.shape
    risk_cs = np.zeros(shape, dtype=np.float64)
    risk_lap = np.zeros(shape, dtype=np.float64)
    if saddles.size == 0:
        return risk_cs, risk_lap, []

    n = saddles.shape[0]
    saddle_info = np.empty((n, 18), dtype=np.float64)

    # alloy properties
    rho = float(getattr(alloy, "rho_kg_m3", 7000.0))
    sigma = float(getattr(alloy, "surface_tension_n_m", 1.5))
    mu = float(getattr(alloy, "viscosity_pa_s", 0.0012))
    cp = float(alloy.cp_j_kgk)
    L = float(alloy.latent_heat_j_kg)
    alpha = float(alloy.thermal_diffusivity_m2_s)
    c_pe = float(alloy.pe_flow_factor_c)
    Fmax = float(alloy.pe_flow_factor_max)
    Tl = float(alloy.t_liquidus_c)
    Te = float(alloy.t_eutectic_or_solidus_c)
    fs_crit = float(alloy.fs_crit)
    dt_crit = float(alloy.dt_crit_s)
    fs_crit_lap = float(alloy.fs_crit_lap)
    dt_crit_lap = float(alloy.dt_crit_lap_s)
    dT_crit = float(alloy.dT_crit_c)
    we_crit = float(alloy.we_crit)
    feed_k1 = float(alloy.feed_k1)

    _sfer64_kernel(
        saddles, ft, H_field, M_mod, sdf, C_field, v_front, velocity,
        dist_feeder, part_mask,
        lut.H_table, lut.T_table, lut.fs_table,
        dirs, adj, shape, dx,
        rho, sigma, mu, cp, L, alpha, c_pe, Fmax,
        Tl, Te, fs_crit, dt_crit, fs_crit_lap, dt_crit_lap,
        dT_crit, we_crit, h0, feed_k1, C_ref,
        risk_cs, risk_lap, saddle_info,
    )

    f_eut = float(getattr(alloy, "eutectic_fraction", 0.0))
    diagnostics = []
    for idx in range(n):
        pers = float(saddle_persistence[idx]) if saddle_persistence is not None and idx < saddle_persistence.shape[0] else 0.0
        diagnostics.append({
            "i": int(saddles[idx, 0]),
            "j": int(saddles[idx, 1]),
            "k": int(saddles[idx, 2]),
            "risk_cs": float(saddle_info[idx, 0]),
            "risk_lap": float(saddle_info[idx, 1]),
            "theta_deg": float(saddle_info[idx, 2]),
            "h_final_m": float(saddle_info[idx, 3]),
            "T_int_c": float(saddle_info[idx, 4]),
            "fs": float(saddle_info[idx, 5]),
            "We": float(saddle_info[idx, 6]),
            "Pe": float(saddle_info[idx, 7]),
            "M_eff_mm": float(saddle_info[idx, 8]),
            "dt_s": float(saddle_info[idx, 9]),
            "v_rel_m_s": float(saddle_info[idx, 10]),
            "N_front": int(saddle_info[idx, 11]),
            "d1": [float(saddle_info[idx, 12]), float(saddle_info[idx, 13]), float(saddle_info[idx, 14])],
            "d2": [float(saddle_info[idx, 15]), float(saddle_info[idx, 16]), float(saddle_info[idx, 17])],
            "persistence_s": pers,
            "f_eut": f_eut,
        })
    return risk_cs, risk_lap, diagnostics
