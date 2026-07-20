"""Alloy and mould material database for JoseCast v7.2."""

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class MoldMaterial:
    """Mould / chill material for Chvorinov and heat transfer."""
    key: str
    name: str
    # Thermal properties (SI)
    k_w_mk: float           # thermal conductivity (W/m·K)
    rho_kg_m3: float        # density (kg/m³)
    cp_j_kgk: float         # specific heat (J/kg·K)
    t0_c: float             # initial mould temp (°C)
    # Chvorinov constant in s/mm² (empirical)
    chvorinov_c: float
    # Darcy / flow
    particle_size_mm: float = 0.25  # representative sand grain size
    permeability_proxy: float = 1.0

    @property
    def diffusivity_mm2_s(self) -> float:
        """Thermal diffusivity α = k / (ρ·c)  [mm²/s]."""
        return (self.k_w_mk / (self.rho_kg_m3 * self.cp_j_kgk)) * 1e6


@dataclass
class Alloy:
    """Cast alloy physical data."""
    key: str
    name: str
    # Metal properties
    rho_g_cm3: float
    rho_kg_m3: float
    latent_heat_j_kg: float
    t_liquidus_c: float
    t_solidus_c: float
    t_pour_c: float
    # Thermal (SI)
    k_w_mk: float
    cp_j_kgk: float
    # Scheil partition coefficient (k < 1)
    partition_coefficient: float = 0.5
    # Density alias for gating weight calculation
    density_g_cm3: float = 0.0
    # Flow / feeding coefficients
    viscosity_pa_s: float = 0.003
    particle_size_mm: float = 0.30
    # Feeding distance FD = feed_k1 * t_section (t_section = 2 * local modulus)
    # so feed_k1=4.5 gives the classic FD = 4.5 * wall_thickness for a plate.
    feed_k1: float = 4.5
    feed_k2: float = 0.0
    # Riser sizing
    riser_m_factor: float = 1.2
    riser_volume_factor: float = 0.3
    # Niyama thresholds (dimensionless as used by engine)
    niyama_macro: float = 0.775
    niyama_shrinkage: float = 1.5
    # Head loss
    elbow_loss_k: float = 0.9
    # Modulus resistance correction [mm per resistance unit]
    modulus_resistance_mm: float = 0.02
    # Solidification shrinkage (volume fraction)
    shrinkage_factor: float = 0.03
    # Secondary dendrite arm spacing [mm] for interdendritic permeability
    dendrite_spacing_mm: float = 0.12
    # Practical porosity acceptance limits [µm] and unavoidable gas/oxide baseline
    micro_pore_limit_um: float = 100.0
    macro_pore_limit_um: float = 1000.0
    gas_pore_baseline_um: float = 0.5
    # Porosity-shape exponent: (1 - N/N_thr)^n.  n>1 makes marginally-low-Niyama
    # regions much less porous, reflecting real-life feeding reservoirs/pressure.
    pore_niyama_exponent: float = 1.5
    feed_risk_exponent: float = 1.2
    # Unavoidable gas/oxide micro-porosity is modulated by local solidification
    # time (thicker section -> larger gas pores) and local Niyama risk.
    gas_pore_time_factor: float = 2.0
    gas_pore_niyama_factor: float = 0.5

    def __post_init__(self):
        if self.density_g_cm3 == 0.0:
            self.density_g_cm3 = self.rho_g_cm3

    @property
    def diffusivity_mm2_s(self) -> float:
        """Thermal diffusivity α = k / (ρ·c)  [mm²/s]."""
        return (self.k_w_mk / (self.rho_kg_m3 * self.cp_j_kgk)) * 1e6


MOLDS: Dict[str, MoldMaterial] = {
    "sand": MoldMaterial(
        key="sand",
        name="Kum Kalıp",
        k_w_mk=0.58,
        rho_kg_m3=1600.0,
        cp_j_kgk=1170.0,
        t0_c=25.0,
        chvorinov_c=2.8,
        particle_size_mm=0.25,
        permeability_proxy=1.0,
    ),
    "metal_mold": MoldMaterial(
        key="metal_mold",
        name="Metal Kalıp",
        k_w_mk=45.0,
        rho_kg_m3=7850.0,
        cp_j_kgk=460.0,
        t0_c=25.0,
        chvorinov_c=0.8,
        particle_size_mm=0.05,
        permeability_proxy=0.05,
    ),
    "ceramic": MoldMaterial(
        key="ceramic",
        name="Seramik Kalıp",
        k_w_mk=1.2,
        rho_kg_m3=2000.0,
        cp_j_kgk=1000.0,
        t0_c=25.0,
        chvorinov_c=2.2,
        particle_size_mm=0.15,
        permeability_proxy=0.7,
    ),
}


ALLOYS: Dict[str, Alloy] = {
    "AlSi7": Alloy(
        key="AlSi7",
        name="AlSi7 (Alüminyum)",
        rho_g_cm3=2.66,
        rho_kg_m3=2660.0,
        latent_heat_j_kg=3.97e5,
        t_liquidus_c=615.0,
        t_solidus_c=577.0,
        t_pour_c=700.0,
        k_w_mk=150.0,
        cp_j_kgk=900.0,
        partition_coefficient=0.12,
        viscosity_pa_s=0.0012,
        particle_size_mm=0.30,
        shrinkage_factor=0.07,
        dendrite_spacing_mm=0.08,
        feed_k1=3.5,
        feed_k2=0.0,
        niyama_macro=0.775,
        niyama_shrinkage=1.5,
    ),
    "GGG40": Alloy(
        key="GGG40",
        name="GGG40 (Dökme Demir)",
        rho_g_cm3=7.1,
        rho_kg_m3=7100.0,
        latent_heat_j_kg=2.3e5,
        t_liquidus_c=1200.0,
        t_solidus_c=1150.0,
        t_pour_c=1350.0,
        k_w_mk=36.0,
        cp_j_kgk=520.0,
        partition_coefficient=0.35,
        viscosity_pa_s=0.006,
        particle_size_mm=0.30,
        shrinkage_factor=0.02,
        dendrite_spacing_mm=0.15,
        feed_k1=4.5,
        feed_k2=0.0,
        niyama_macro=0.775,
        niyama_shrinkage=1.5,
    ),
    "42CrMo4": Alloy(
        key="42CrMo4",
        name="42CrMo4 (Çelik)",
        rho_g_cm3=7.85,
        rho_kg_m3=7850.0,
        latent_heat_j_kg=2.7e5,
        t_liquidus_c=1510.0,
        t_solidus_c=1410.0,
        t_pour_c=1600.0,
        k_w_mk=45.0,
        cp_j_kgk=460.0,
        partition_coefficient=0.20,
        viscosity_pa_s=0.005,
        particle_size_mm=0.25,
        shrinkage_factor=0.03,
        dendrite_spacing_mm=0.12,
        feed_k1=6.0,
        feed_k2=0.0,
        niyama_macro=0.775,
        niyama_shrinkage=1.5,
    ),
    "bronze": Alloy(
        key="bronze",
        name="Bronz",
        rho_g_cm3=8.8,
        rho_kg_m3=8800.0,
        latent_heat_j_kg=2.1e5,
        t_liquidus_c=1000.0,
        t_solidus_c=950.0,
        t_pour_c=1150.0,
        k_w_mk=60.0,
        cp_j_kgk=380.0,
        partition_coefficient=0.25,
        viscosity_pa_s=0.004,
        particle_size_mm=0.30,
        shrinkage_factor=0.06,
        dendrite_spacing_mm=0.10,
        feed_k1=4.0,
        feed_k2=0.0,
        niyama_macro=0.775,
        niyama_shrinkage=1.5,
    ),
}


def get_alloy(key: str) -> Alloy:
    return ALLOYS.get(key, ALLOYS["42CrMo4"])


def get_mold(key: str) -> MoldMaterial:
    return MOLDS.get(key, MOLDS["sand"])


# Backwards-compatible aliases
Material = Alloy
MATERIALS = ALLOYS


def get_material(key: str) -> Material:
    return get_alloy(key)


def chvorinov_c_from_properties(alloy: Alloy, mold: MoldMaterial) -> float:
    """
    Compute Chvorinov constant C in s/mm^2 from physical properties.
    C = (rho_m * L_eff / (T_m - T_0))^2 * (pi / (4 * k * rho * c))
    L_eff includes the superheat that must also be removed:
        L_eff = L + cp * (T_pour - T_liquidus)
    Returns a value in s/mm^2.
    """
    tm = (alloy.t_liquidus_c + alloy.t_solidus_c) / 2.0
    delta_t = max(tm - mold.t0_c, 1.0)
    # Effective latent heat including superheat [J/kg]
    l_eff = alloy.latent_heat_j_kg + alloy.cp_j_kgk * max(
        alloy.t_pour_c - alloy.t_liquidus_c, 0.0
    )
    # rho_m * L_eff / (T_m - T0)  [J/m^3 / K]
    numerator = alloy.rho_kg_m3 * l_eff / delta_t
    # k * rho * c  [W/mK * kg/m3 * J/kgK = W*J/(m^4 K^2)]
    denom = mold.k_w_mk * mold.rho_kg_m3 * mold.cp_j_kgk
    if denom <= 0:
        return mold.chvorinov_c
    c_si = (numerator ** 2) * (np.pi / (4.0 * denom))
    # Convert s/m^2 -> s/mm^2
    return float(c_si / 1e6)
