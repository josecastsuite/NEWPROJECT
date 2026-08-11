"""V8 Péclet-limited effective modulus for cold-shut risk.

Fix2 from V8: front velocity and local modulus make the effective cooling time
longer than the static Chvorinov value, but bounded to avoid the 88s -> 1650s
explosion seen in V7.
"""
import numpy as np
from numba import njit


def peclet_front_velocity_m_s(
    fill_time_s: np.ndarray, dx_mm: float, axis: int = -1
) -> np.ndarray:
    """
    Estimate front velocity from the fill-time gradient.

    v_front = dx / |grad(fill_time)|  [m/s]

    ``fill_time_s`` is in seconds; ``dx_mm`` is the voxel size in mm.
    """
    dx_m = dx_mm / 1000.0
    grad = np.gradient(fill_time_s)
    if fill_time_s.ndim == 1:
        mag = np.abs(grad)
    else:
        # stack squared gradients across all axes
        mag = np.sqrt(sum(np.square(g) for g in grad))
    return dx_m / (mag + 1e-12)


def peclet_correction(
    M_mod_mm: np.ndarray,
    v_front_m_s: np.ndarray,
    alpha_m2_s: float,
    c_pe: float,
    Fmax: float,
) -> np.ndarray:
    """
    Péclet speed-up factor for the local modulus.

    M_m = M_mod_mm * 1e-3  [m]
    Pe = v_front * M_m / alpha
    f_Pe = 1 + min(c * Pe**0.5, Fmax - 1)

    The factor is >= 1 and <= Fmax.
    """
    M_m = M_mod_mm * 1e-3
    with np.errstate(divide="ignore", invalid="ignore"):
        Pe = v_front_m_s * M_m / max(alpha_m2_s, 1e-18)
        Pe = np.clip(Pe, 0.0, None)
        f_Pe = 1.0 + np.minimum(c_pe * np.sqrt(Pe), Fmax - 1.0)
    return np.clip(f_Pe, 1.0, Fmax)


def effective_modulus_pe(
    M_mod_mm: np.ndarray,
    v_front_m_s: np.ndarray,
    alpha_m2_s: float,
    c_pe: float,
    Fmax: float,
    f_feeder: np.ndarray = None,
) -> np.ndarray:
    """
    Effective modulus including Péclet speed-up and feeder effect.

    M_eff = M_mod * f_Pe * f_feeder  [mm]
    """
    f_Pe = peclet_correction(M_mod_mm, v_front_m_s, alpha_m2_s, c_pe, Fmax)
    if f_feeder is not None:
        f_Pe = f_Pe * f_feeder
    return M_mod_mm * f_Pe


@njit(cache=True)
def _peclet_correction_numba(
    M_mod_mm: float,
    v_front_m_s: float,
    alpha_m2_s: float,
    c_pe: float,
    Fmax: float,
) -> float:
    M_m = M_mod_mm * 1e-3
    Pe = v_front_m_s * M_m / max(alpha_m2_s, 1e-18)
    if Pe < 0.0:
        Pe = 0.0
    f_Pe = 1.0 + min(c_pe * np.sqrt(Pe), Fmax - 1.0)
    if f_Pe < 1.0:
        f_Pe = 1.0
    if f_Pe > Fmax:
        f_Pe = Fmax
    return f_Pe


@njit(cache=True)
def _effective_modulus_pe_numba(
    M_mod_mm: float,
    v_front_m_s: float,
    alpha_m2_s: float,
    c_pe: float,
    Fmax: float,
    f_feeder: float = 1.0,
) -> float:
    return M_mod_mm * _peclet_correction_numba(
        M_mod_mm, v_front_m_s, alpha_m2_s, c_pe, Fmax
    ) * f_feeder
