"""V8 film-drainage physics for cold-shut / confluence risk.

Fix1 from V8: the drainage model is regularised so Pd <= capillary pressure
gives h_final = R (no drainage).  The H_h factor is a dimensionless sigmoid
around h0 = 100 nm.
"""
import numpy as np
from numba import njit


H0_DEFAULT = 100e-9  # m


def dynamic_pressure(rho_kg_m3: float, v_rel_m_s: float) -> float:
    """Stagnation pressure: Pd = 0.5 * rho * v_rel^2 [Pa]."""
    return 0.5 * rho_kg_m3 * v_rel_m_s * v_rel_m_s


def capillary_pressure(sigma_n_m: float, R_mm: float) -> float:
    """Laplace pressure for a meniscus of radius R: 2*sigma/R [Pa]."""
    return 2.0 * sigma_n_m / (R_mm * 1e-3)


def effective_pressure(rho_kg_m3: float, v_rel_m_s: float, sigma_n_m: float, R_mm: float) -> float:
    """dP_eff = max(0, Pd - 2*sigma/R) [Pa]."""
    return max(0.0, dynamic_pressure(rho_kg_m3, v_rel_m_s) - capillary_pressure(sigma_n_m, R_mm))


def film_thickness(
    R_mm: float,
    v_rel_m_s: float,
    sigma_n_m: float,
    rho_kg_m3: float,
    mu_pa_s: float,
    t_film_s: float,
) -> float:
    """
    Film thickness between two converging liquid-metal fronts [m].

    Parameters
    ----------
    R_mm : float
        Local front radius / modulus [mm].
    v_rel_m_s : float
        Relative velocity magnitude of the two fronts [m/s].
    sigma_n_m : float
        Surface tension [N/m].
    rho_kg_m3 : float
        Liquid density [kg/m3].
    mu_pa_s : float
        Dynamic viscosity [Pa*s].
    t_film_s : float
        Film thinning time [s]; typically min(dt, t_solid).

    Returns
    -------
    h_final : float
        Final film thickness [m].  If dP_eff <= 0 the fronts cannot rupture
        the meniscus and the physical upper bound R [m] is returned.
    """
    dP_eff = effective_pressure(rho_kg_m3, v_rel_m_s, sigma_n_m, R_mm)
    R_m = R_mm * 1e-3
    if dP_eff <= 0.0:
        return R_m

    inv_h2 = 1.0 / (R_m * R_m) + (4.0 * dP_eff * t_film_s) / (3.0 * mu_pa_s * (R_m * R_m))
    if inv_h2 <= 0.0:
        return R_m
    return 1.0 / np.sqrt(inv_h2)


def film_drainage_factor(h_final_m: float, h0_m: float = H0_DEFAULT) -> float:
    """
    Dimensionless cold-shut risk factor from final film thickness.

    h >> h0  -> risk ~ 1 (film can still drain, fronts may not weld)
    h << h0  -> risk ~ 0 (film already thin enough for metallic contact)
    """
    ratio = h_final_m / h0_m
    return 1.0 / (1.0 + np.exp(-10.0 * (ratio - 1.0)))


@njit(cache=True)
def _film_thickness_numba(
    R_mm: float,
    v_rel_m_s: float,
    sigma_n_m: float,
    rho_kg_m3: float,
    mu_pa_s: float,
    t_film_s: float,
) -> float:
    """Numba scalar implementation for use in SFER loops."""
    Pd = 0.5 * rho_kg_m3 * v_rel_m_s * v_rel_m_s
    cap = 2.0 * sigma_n_m / (R_mm * 1e-3)
    dP_eff = max(0.0, Pd - cap)
    R_m = R_mm * 1e-3
    if dP_eff <= 0.0:
        return R_m
    inv_h2 = 1.0 / (R_m * R_m) + (4.0 * dP_eff * t_film_s) / (3.0 * mu_pa_s * (R_m * R_m))
    if inv_h2 <= 0.0:
        return R_m
    return 1.0 / np.sqrt(inv_h2)


@njit(cache=True)
def _film_drainage_factor_numba(h_final_m: float, h0_m: float = H0_DEFAULT) -> float:
    ratio = h_final_m / h0_m
    return 1.0 / (1.0 + np.exp(-10.0 * (ratio - 1.0)))
