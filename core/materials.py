"""Alloy and mould material database for JoseCast v7.1."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class MoldMaterial:
    """Mould / chills used in Chvorinov solidification time."""
    key: str
    name: str
    # Thermal properties
    k_w_mk: float          # thermal conductivity  (W/m·K)
    rho_kg_m3: float       # density               (kg/m³)
    cp_j_kgk: float        # specific heat         (J/kg·K)
    t0_c: float            # initial mould temp    (°C)
    # Empirical Chvorinov C when M is in mm and t is in seconds
    chvorinov_c: float     # s/mm²
    # Permeability proxy for Darcy (m² scale, only relative)
    permeability_proxy: float = 1.0


@dataclass
class Alloy:
    """Cast alloy physical data."""
    key: str
    name: str
    # Metal properties
    rho_g_cm3: float       # density
    rho_kg_m3: float        # kg/m³
    latent_heat_j_kg: float
    t_liquidus_c: float
    t_solidus_c: float
    t_pour_c: float         # pouring temperature
    # Gating / feeding coefficients
    density_g_cm3: float    # same as rho_g_cm3 (alias for gating)
    feed_factor: float      # FD = feed_factor * t_section (t_section = wall thickness)
    riser_m_factor: float = 1.2
    riser_volume_factor: float = 0.3
    # Niyama thresholds (dimensionless as used by engine)
    niyama_macro: float = 0.775
    niyama_shrinkage: float = 1.5
    # Fluid viscosity proxy for Darcy (Pa·s relative)
    viscosity_proxy: float = 0.003


MOLDS: Dict[str, MoldMaterial] = {
    "sand": MoldMaterial(
        key="sand",
        name="Kum Kalıp",
        k_w_mk=0.58,
        rho_kg_m3=1600.0,
        cp_j_kgk=1170.0,
        t0_c=25.0,
        chvorinov_c=2.8,
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
        density_g_cm3=2.66,
        feed_factor=1.5,
        niyama_macro=0.775,
        niyama_shrinkage=1.5,
        viscosity_proxy=0.0012,
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
        density_g_cm3=7.1,
        feed_factor=2.0,
        niyama_macro=0.775,
        niyama_shrinkage=1.5,
        viscosity_proxy=0.006,
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
        density_g_cm3=7.85,
        feed_factor=2.25,
        niyama_macro=0.775,
        niyama_shrinkage=1.5,
        viscosity_proxy=0.005,
    ),
}


def get_alloy(key: str) -> Alloy:
    return ALLOYS.get(key, ALLOYS["42CrMo4"])


def get_mold(key: str) -> MoldMaterial:
    return MOLDS.get(key, MOLDS["sand"])


# Backwards-compatible alias
MATERIALS = ALLOYS


def chvorinov_c_from_properties(
    alloy: Alloy, mold: MoldMaterial
) -> float:
    """
    Compute Chvorinov constant in s/mm² from material properties.
    Formula: C = (ρ_m * L / (T_m - T_o))² * (π / (4 * k * ρ * c))
    """
    import numpy as np

    rho_m = alloy.rho_kg_m3
    L = alloy.latent_heat_j_kg
    delta_t = alloy.t_liquidus_c - mold.t0_c
    if delta_t <= 0:
        delta_t = 100.0
    k = mold.k_w_mk
    rho = mold.rho_kg_m3
    c = mold.cp_j_kgk
    c_si = (
        (rho_m * L / delta_t) ** 2
        * (np.pi / (4.0 * k * rho * c))
    )
    # convert from s/m² to s/mm²
    return c_si / 1e6


# Backwards-compatible alias used by old code paths
Material = Alloy

def get_material(key: str) -> Material:
    return get_alloy(key)
