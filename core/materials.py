"""Alloy and mould material database for JoseCast v8.0.

The actual material data lives in JSON files under ``data/materials/``:

* ``data/materials/alloys.json`` – cast alloy physical properties.
* ``data/materials/molds.json``  – mould / chill / sleeve properties.

This module loads those JSON files at import time and exposes the same
``ALLOYS``/``MOLDS`` dictionaries and helper functions as before.  If the JSON
files are missing or malformed, a small built-in fallback is used so the
program keeps running.
"""
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass
class MoldMaterial:
    """Mould / chill material for Chvorinov and heat transfer."""

    key: str
    name: str
    # Thermal properties (SI)
    k_w_mk: float  # thermal conductivity (W/m·K)
    rho_kg_m3: float  # density (kg/m³)
    cp_j_kgk: float  # specific heat (J/kg·K)
    t0_c: float  # initial mould temp (°C)
    # Chvorinov constant in dk/cm² (minutes per square centimetre, empirical)
    chvorinov_c: float
    # Darcy / flow
    particle_size_mm: float = 0.25  # representative sand grain size
    permeability_proxy: float = 1.0
    # Mold type category (sand, metal, ceramic, shell, investment, chill, ...)
    mold_type: str = "sand"
    is_sand: bool = True
    # Green-sand properties for particle-based permeability / air-leakage model
    afs_grain_size: float = 50.0  # AFS grain fineness number (GFN)
    moisture_percent: float = 4.0  # % moisture
    binder_percent: float = 2.0  # % bentonite / binder
    compactability_percent: float = 45.0  # % compactability
    mold_rigidity_factor: float = 1.0  # 0 = weak green sand, 1 = rigid metal/shell mold

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
    surface_tension_n_m: float = 0.0
    particle_size_mm: float = 0.30
    # Feeding distance FD = feed_k1 * t_section (t_section = 2 * local modulus)
    # so feed_k1=4.5 gives the classic FD = 4.5 * wall_thickness for a plate.
    feed_k1: float = 4.5
    feed_k2: float = 0.0
    # Riser sizing
    riser_m_factor: float = 1.2
    riser_volume_factor: float = 0.3
    # Exothermic mini-riser correction: the delivered metal is kept liquid longer,
    # so the required volume can be reduced while still supplying the same modulus.
    # A 0.45 yield means ~45% of a normal riser volume is needed; the equivalent
    # modulus boost (exothermic_head_factor) accounts for the extra liquid metal time.
    exothermic_volume_yield: float = 0.45
    exothermic_modulus_factor: float = 1.5
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
    # Thermomechanical stress defaults (simplified 1-D constrained shrinkage model)
    young_modulus_pa: float = 2.1e11
    thermal_expansion_cinv: float = 1.2e-5
    yield_strength_pa: float = 2.5e8
    room_temp_yield_pa: float = 4.0e8
    hot_tear_threshold_strain: float = 0.015
    # Material family for family-specific shrinkage / expansion models
    material_family: str = "steel"
    # Chemical composition (%) for cast-iron carbon equivalent: CE = C + (Si + P)/3
    C_pct: float = 0.0
    Si_pct: float = 0.0
    P_pct: float = 0.0
    carbon_equivalent: float = -1.0  # < 0 means auto-compute from C/Si/P
    # Graphite expansion model (cast irons)
    graphite_expansion_fraction: float = 0.0  # max volume expansion during eutectic
    inoculation_factor: float = 1.0  # 0 = no inoculation, 1 = strong inoculation
    # Practical porosity acceptance limits [µm] and unavoidable gas/oxide baseline
    micro_pore_limit_um: float = 50.0
    macro_pore_limit_um: float = 500.0
    gas_pore_baseline_um: float = 0.5
    # Porosity-shape exponent: (1 - N/N_thr)^n.  n>1 makes marginally-low-Niyama
    # regions much less porous, reflecting real-life feeding reservoirs/pressure.
    pore_niyama_exponent: float = 1.5
    feed_risk_exponent: float = 1.2
    # Unavoidable gas/oxide micro-porosity is modulated by local solidification
    # time (thicker section -> larger gas pores) and local Niyama risk.
    gas_pore_time_factor: float = 2.0
    gas_pore_niyama_factor: float = 0.5
    # Carlson-Beckermann dimensionless-Niyama calibration.
    # curve_key selects the published fit ("WCB", "A356", "AZ91D").
    # niyama_star_scale converts the engine's Niyama [sqrt(K s)/mm] to the
    # dimensionless Ny* used by the Carlson curve.  If <= 0 it is solved from
    # niyama_macro and macro_pore_limit_um so N=niyama_macro gives a reference
    # pore-volume percentage for risk calibration.
    # pore_size_um_per_porosity_pct is kept for that risk reference.
    carlson_curve_key: str = "WCB"
    niyama_star_scale: float = 0.0
    pore_size_um_per_porosity_pct: float = 400.0
    # Ingate/gate velocity above which surface turbulence entraps oxide films / air.
    # Campbell and Hojjat/Beckermann give ~0.45-0.5 m/s for Al; steels are less
    # oxide-sensitive, so a higher value is used for ferrous alloys.
    critical_entrainment_velocity_m_s: float = 0.5
    pore_entrainment_exponent: float = 1.5
    pore_entrainment_factor: float = 1.0
    # Physical size model: d = cbrt(6/pi * gp) * L, where L is the local
    # characteristic length (max(2*M_mod, SDAS)) scaled by this factor.
    pore_size_length_factor: float = 1.0
    pore_size_cube_root_factor: float = 1.0

    def __post_init__(self):
        if self.density_g_cm3 == 0.0:
            self.density_g_cm3 = self.rho_g_cm3
        if self.carbon_equivalent < 0.0 and (self.C_pct > 0.0 or self.Si_pct > 0.0 or self.P_pct > 0.0):
            self.carbon_equivalent = self.C_pct + (self.Si_pct + self.P_pct) / 3.0
        if self.carbon_equivalent < 0.0:
            self.carbon_equivalent = 0.0
        if self.niyama_star_scale <= 0.0:
            self.niyama_star_scale = self._niyama_star_scale()
        if self.surface_tension_n_m <= 0.0:
            family_defaults = {
                "aluminum": 0.90,
                "magnesium": 0.55,
                "cast_iron": 1.40,
                "steel": 1.70,
                "copper": 1.15,
                "superalloy": 1.75,
            }
            family = (self.material_family or "steel").lower().strip()
            self.surface_tension_n_m = family_defaults.get(family, 1.50)

    def _niyama_star_scale(self) -> float:
        """Solve the Carlson curve so N=niyama_macro gives the macro size proxy."""
        target_gp = self.macro_pore_limit_um / max(
            self.pore_size_um_per_porosity_pct, 1e-9
        )

        def gp_for_scale(s: float) -> float:
            ny = self.niyama_macro * s
            if ny <= 0:
                return float("inf")
            b0 = max(self.shrinkage_factor * 100.0, 1e-9)
            key = self.carlson_curve_key
            if key == "A356":
                if ny <= 1.43:
                    val = -2.068 * math.log10(ny) + 3.160
                elif ny <= 18.0:
                    val = 4.024 * (ny ** -0.9786)
                else:
                    val = 7.771 * (ny ** -1.206)
            elif key == "AZ91D":
                if ny <= 41.0:
                    val = -1.671 * math.log10(ny) + 3.483
                elif ny <= 45.2:
                    val = -10.81 * math.log10(ny) + 18.23
                else:
                    val = 73.01 * (ny ** -1.415)
            else:  # WCB default
                if ny <= 28.2:
                    val = -1.654 * math.log10(ny) + 3.052
                else:
                    val = 43.05 * (ny ** -1.254)
            return max(0.0, min(b0, val))

        lo, hi = 1e-3, 1e6
        if gp_for_scale(lo) <= target_gp:
            return lo
        if gp_for_scale(hi) >= target_gp:
            return hi
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if gp_for_scale(mid) > target_gp:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    @property
    def diffusivity_mm2_s(self) -> float:
        """Thermal diffusivity α = k / (ρ·c)  [mm²/s]."""
        return (self.k_w_mk / (self.rho_kg_m3 * self.cp_j_kgk)) * 1e6


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

ALLOYS: Dict[str, Alloy] = {}
MOLDS: Dict[str, MoldMaterial] = {}


def _json_path(name: str) -> Path:
    return Path(__file__).resolve().parent / "materials_data" / name


def _load_json_dict(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _default_alloys() -> Dict[str, Alloy]:
    """Minimal fallback if alloys.json is missing."""
    return {
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
            viscosity_pa_s=0.005,
            shrinkage_factor=0.03,
            critical_entrainment_velocity_m_s=0.9,
        ),
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
            viscosity_pa_s=0.0012,
            shrinkage_factor=0.07,
            dendrite_spacing_mm=0.08,
            feed_k1=3.5,
            carlson_curve_key="A356",
            critical_entrainment_velocity_m_s=0.5,
            young_modulus_pa=7.0e10,
            thermal_expansion_cinv=2.3e-5,
            yield_strength_pa=1.5e8,
            room_temp_yield_pa=2.5e8,
        ),
    }


def _default_molds() -> Dict[str, MoldMaterial]:
    """Minimal fallback if molds.json is missing."""
    return {
        "sand": MoldMaterial(
            key="sand",
            name="Kum Kalıp",
            k_w_mk=0.58,
            rho_kg_m3=1600.0,
            cp_j_kgk=1170.0,
            t0_c=25.0,
            chvorinov_c=2.8,
        ),
        "metal_mold": MoldMaterial(
            key="metal_mold",
            name="Metal Kalıp",
            k_w_mk=45.0,
            rho_kg_m3=7850.0,
            cp_j_kgk=460.0,
            t0_c=25.0,
            chvorinov_c=0.8,
            mold_type="metal",
            is_sand=False,
        ),
    }


def _load_materials() -> None:
    """Populate ALLOYS and MOLDS from JSON, with a tiny fallback."""
    ALLOYS.clear()
    MOLDS.clear()

    alloys_raw = _load_json_dict(_json_path("alloys.json"))
    if alloys_raw:
        for k, v in alloys_raw.items():
            try:
                ALLOYS[k] = Alloy(**v)
            except Exception as exc:
                print(f"[MATERIALS] skipping invalid alloy {k}: {exc}")
    if not ALLOYS:
        ALLOYS.update(_default_alloys())

    molds_raw = _load_json_dict(_json_path("molds.json"))
    if molds_raw:
        for k, v in molds_raw.items():
            try:
                MOLDS[k] = MoldMaterial(**v)
            except Exception as exc:
                print(f"[MATERIALS] skipping invalid mold {k}: {exc}")
    if not MOLDS:
        MOLDS.update(_default_molds())


_load_materials()


def get_alloy(key: str) -> Alloy:
    if key in ALLOYS:
        return ALLOYS[key]
    return ALLOYS.get("42CrMo4", next(iter(ALLOYS.values())))


def get_mold(key: str) -> MoldMaterial:
    if key in MOLDS:
        return MOLDS[key]
    return MOLDS.get("sand", next(iter(MOLDS.values())))


def save_molds(path: Optional[Path] = None) -> None:
    """Persist the current in-memory ``MOLDS`` dictionary back to JSON."""
    target = path or _json_path("molds.json")
    data = {key: asdict(mold) for key, mold in MOLDS.items()}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def make_effective_mold(
    mold: MoldMaterial,
    casting_params: Optional[object] = None,
    body: Optional[object] = None,
) -> MoldMaterial:
    """Return a copy of ``mold`` with GUI / per-body overrides applied."""
    # Use a per-CORE preset if the body provides one.
    base = mold
    overrides: Dict[str, float] = {}
    if body is not None:
        preset_key = getattr(body, "mold_preset", "")
        if preset_key and preset_key in MOLDS:
            base = MOLDS[preset_key]
        afs = getattr(body, "mold_afs_grain_size", 0.0) or 0.0
        moisture = getattr(body, "mold_moisture_percent", 0.0) or 0.0
        binder = getattr(body, "mold_binder_percent", 0.0) or 0.0
        compact = getattr(body, "mold_compactability_percent", 0.0) or 0.0
        rigidity = getattr(body, "mold_rigidity_factor", 0.0) or 0.0
        if afs:
            overrides["afs_grain_size"] = afs
        if moisture:
            overrides["moisture_percent"] = moisture
        if binder:
            overrides["binder_percent"] = binder
        if compact:
            overrides["compactability_percent"] = compact
        if rigidity > 0.0:
            overrides["mold_rigidity_factor"] = rigidity

    if casting_params is not None:
        if getattr(casting_params, "mold_afs_grain_size", 0.0):
            overrides["afs_grain_size"] = casting_params.mold_afs_grain_size
        if getattr(casting_params, "mold_moisture_percent", 0.0):
            overrides["moisture_percent"] = casting_params.mold_moisture_percent
        if getattr(casting_params, "mold_binder_percent", 0.0):
            overrides["binder_percent"] = casting_params.mold_binder_percent
        if getattr(casting_params, "mold_compactability_percent", 0.0):
            overrides["compactability_percent"] = casting_params.mold_compactability_percent
        rigidity = getattr(casting_params, "mold_rigidity_factor", -1.0)
        if rigidity >= 0.0:
            overrides["mold_rigidity_factor"] = rigidity

    if overrides:
        return replace(base, **overrides)
    return base


# Backwards-compatible aliases
Material = Alloy
MATERIALS = ALLOYS


def get_material(key: str) -> Material:
    return get_alloy(key)


def chvorinov_c_from_properties(alloy: Alloy, mold: MoldMaterial) -> float:
    """
    Return the Chvorinov constant C in dk/cm^2 (minutes per square centimetre).

    The materials database (mold.chvorinov_c) stores the empirical foundry
    constant, which is the authoritative value.  If it is missing, a physics-
    based estimate is computed from alloy/mould properties and converted to
    dk/cm^2.
    """
    # Authoritative empirical constant from the materials database.
    empirical = getattr(mold, "chvorinov_c", 0.0) or 0.0
    if empirical > 0.0:
        return float(empirical)

    tm = (alloy.t_liquidus_c + alloy.t_solidus_c) / 2.0
    delta_t = max(tm - mold.t0_c, 1.0)
    l_eff = alloy.latent_heat_j_kg + alloy.cp_j_kgk * max(
        alloy.t_pour_c - alloy.t_liquidus_c, 0.0
    )
    numerator = alloy.rho_kg_m3 * l_eff / delta_t
    denom = mold.k_w_mk * mold.rho_kg_m3 * mold.cp_j_kgk
    if denom <= 0:
        return 2.0
    c_si = (numerator ** 2) * (np.pi / (4.0 * denom))
    # Convert s/m^2 -> dk/cm^2  (1 min = 60 s, 1 m^2 = 10^4 cm^2)
    return float(c_si / (60.0 * 1e4))
