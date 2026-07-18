"""Material / alloy physical constants for JoseCast v7 Titan."""

from dataclasses import dataclass


@dataclass
class Material:
    name: str
    display_name: str
    # Chvorinov constant K (s/mm^2) for sand mould; material dependent.
    chvorinov_k: float
    # Metal density (g/cm3) for gating weight calculations.
    density_g_cm3: float = 7.2
    # Feeding distance safety factor: dist <= factor * M
    feed_factor: float = 4.5
    # Niyama critical thresholds (dimensionless as used by engine).
    niyama_macro: float = 0.5
    niyama_shrinkage: float = 1.0
    # Riser modulus must be >= riser_m_factor * hotspot_m
    riser_m_factor: float = 1.2
    # Riser volume must be >= riser_volume_factor * feed_region
    riser_volume_factor: float = 0.3


MATERIALS = {
    "steel": Material(
        name="steel",
        display_name="Çelik",
        chvorinov_k=3.2,
        density_g_cm3=7.85,
        feed_factor=4.5,
    ),
    "cast_iron": Material(
        name="cast_iron",
        display_name="Dökme Demir",
        chvorinov_k=2.9,
        density_g_cm3=7.1,
        feed_factor=4.2,
    ),
    "aluminum": Material(
        name="aluminum",
        display_name="Alüminyum",
        chvorinov_k=2.4,
        density_g_cm3=2.7,
        feed_factor=3.5,
    ),
    "bronze": Material(
        name="bronze",
        display_name="Bronz",
        chvorinov_k=2.7,
        density_g_cm3=8.8,
        feed_factor=4.0,
    ),
}


def get_material(key: str) -> Material:
    return MATERIALS.get(key, MATERIALS["steel"])
