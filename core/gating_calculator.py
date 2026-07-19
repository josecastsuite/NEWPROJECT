"""Pure numeric helpers extracted from gating_calculator_tr.py and Filling_time_tr.py.

These are the exact working equations the user supplied; they are kept in a
headless module so the JoseCast analyzer can call them without pulling in
tkinter / GUI code.
"""

import math
from typing import Tuple

G = 9.81  # m/s^2

RHO_REF = 7000.0  # kg/m³, reference density used by Filling_time_tr.py


def parse_ratio(text: str) -> Tuple[float, float, float]:
    """'1:2:2' -> (1.0, 2.0, 2.0)."""
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError("Gating oranı 1:2:2 formatında olmalı.")
    return tuple(float(p.replace(",", ".")) for p in parts)


def auto_fill_time(mat_name: str, W_part_kg: float) -> float:
    """Practical fill-time estimate from gating_calculator_tr.py.

    Small <= 5 kg, medium <= 20 kg, large > 20 kg.
    """
    m = mat_name.lower()

    if W_part_kg <= 5.0:
        size = "small"
    elif W_part_kg <= 20.0:
        size = "medium"
    else:
        size = "large"

    if "çelik" in m or "steel" in m or "42" in m:
        return {"small": 3.0, "medium": 5.0, "large": 8.0}.get(size, 5.0)
    if "gri pik" in m or "sfero" in m or "nodular" in m or "ggg" in m:
        return {"small": 2.5, "medium": 4.0, "large": 6.0}.get(size, 4.0)
    if "bronz" in m or "bronze" in m:
        return {"small": 2.0, "medium": 3.5, "large": 5.0}.get(size, 3.5)

    return {"small": 3.5, "medium": 5.5, "large": 8.5}.get(size, 5.5)


def head_reduction_fraction(W_part_kg: float) -> float:
    """Mass-dependent head reduction used in effective_head()."""
    m = max(0.0, W_part_kg)
    if m > 3000.0:
        m = 3000.0

    points = [
        (0.0, 0.00),
        (100.0, 0.40),
        (250.0, 0.50),
        (500.0, 0.60),
        (1000.0, 0.70),
        (3000.0, 0.75),
    ]

    for i in range(len(points) - 1):
        m0, r0 = points[i]
        m1, r1 = points[i + 1]
        if m0 <= m <= m1:
            if m1 == m0:
                return float(r1)
            t = (m - m0) / (m1 - m0)
            return float(r0 + t * (r1 - r0))

    return float(points[-1][1])


def effective_head(H_m: float, W_part_kg: float) -> float:
    """H_eff = H * (1 - reduction_fraction)."""
    if H_m <= 0.0:
        return H_m
    frac = head_reduction_fraction(W_part_kg)
    frac = max(0.0, min(frac, 0.9))
    return H_m * (1.0 - frac)


def compute_gating(
    W_kg: float,
    rho_kgm3: float,
    H_m: float,
    t_fill_s: float,
    Cd: float = 0.8,
    gating_ratio: Tuple[float, float, float] = (1.0, 2.0, 2.0),
    n_ingates: int = 2,
):
    """Exact gating area / diameter calculation from gating_calculator_tr.py."""
    As_ratio, Ar_ratio, Ag_ratio = gating_ratio

    # Choke velocity at sprue base
    Vc = math.sqrt(2 * G * H_m)  # m/s

    # Choke / sprue base area
    As = W_kg / (rho_kgm3 * Cd * t_fill_s * Vc)  # m²

    # Runner and gate areas from ratio
    Ar_total = As * (Ar_ratio / As_ratio)
    Ag_total = As * (Ag_ratio / As_ratio)
    Ag_each = Ag_total / max(n_ingates, 1)

    def area_to_diameter(area_m2: float) -> float:
        if area_m2 <= 0.0:
            return 0.0
        return math.sqrt(4 * area_m2 / math.pi)

    d_sprue_m = area_to_diameter(As)
    d_ingate_m = area_to_diameter(Ag_each)

    return {
        "As_m2": As,
        "Ar_total_m2": Ar_total,
        "Ag_total_m2": Ag_total,
        "Ag_each_m2": Ag_each,
        "Vc_ms": Vc,
        "d_sprue_m": d_sprue_m,
        "d_ingate_m": d_ingate_m,
    }


def compute_modulus_and_riser(
    W_part_kg: float,
    rho_kgm3: float,
    A_cast_m2: float,
    k_mod: float = 1.2,
):
    """Cylindrical riser pre-design from gating_calculator_tr.py."""
    if A_cast_m2 <= 0.0:
        raise ValueError("A_cast (soğuma yüzey alanı) 0'dan büyük olmalı.")

    V_cast_m3 = W_part_kg / rho_kgm3
    M_cast_m = V_cast_m3 / A_cast_m2
    M_riser_req_m = k_mod * M_cast_m

    D = 0.05
    D_max = 1.0
    step = 0.005

    best_D = None
    best_H = None
    best_M = None

    while D <= D_max:
        H = D
        V_r = math.pi * (D / 2.0) ** 2 * H
        A_r = math.pi * D * H + 2.0 * math.pi * (D / 2.0) ** 2
        M_r = V_r / A_r
        if M_r >= M_riser_req_m:
            best_D = D
            best_H = H
            best_M = M_r
            break
        D += step

    if best_D is None:
        best_D = D_max
        best_H = D_max
        V_r = math.pi * (best_D / 2.0) ** 2 * best_H
        A_r = math.pi * best_D * best_H + 2.0 * math.pi * (best_D / 2.0) ** 2
        best_M = V_r / A_r

    return {
        "V_cast_m3": V_cast_m3,
        "M_cast_m": M_cast_m,
        "M_riser_req_m": M_riser_req_m,
        "riser_D_m": best_D,
        "riser_H_m": best_H,
        "riser_M_m": best_M,
    }


def t_fill_base_piecewise(m_kg: float) -> Tuple[float, str]:
    """Campbell Table 4.1 piecewise log interpolation for base fill time."""
    m = max(m_kg, 1e-6)
    points = [
        (20.0, 2.0),
        (100.0, 6.0),
        (250.0, 14.0),
        (500.0, 30.0),
        (2000.0, 60.0),
        (3000.0, 130.0),
    ]

    def log_interp(m_val, mx1, tx1, mx2, tx2):
        return tx1 + (tx2 - tx1) * (math.log10(m_val / mx1) / math.log10(mx2 / mx1))

    if m <= points[0][0]:
        return points[0][1], f"m ≤ {points[0][0]} kg"
    for i in range(len(points) - 1):
        m1, t1 = points[i]
        m2, t2 = points[i + 1]
        if m1 < m <= m2:
            t = log_interp(m, m1, t1, m2, t2)
            detail = f"{m1}-{m2} kg arası log interp"
            return t, detail
    return points[-1][1], f"m > {points[-1][0]} kg"


def calc_campbell_parameters(m_kg: float, rho: float, thickness_mm: float, superheat_c: float):
    """Campbell-compatible fill time from Filling_time_tr.py."""
    t_base, t_base_detail = t_fill_base_piecewise(m_kg)

    f_rho = (RHO_REF / max(rho, 1e-6)) ** 0.35
    f_thick = (max(thickness_mm, 1.0) / 20.0) ** 0.2
    f_temp = (max(superheat_c, 10.0) / 100.0) ** 0.4

    t_fill = t_base * f_rho * f_thick * f_temp

    return {
        "t_base": t_base,
        "t_base_detail": t_base_detail,
        "f_rho": f_rho,
        "f_thick": f_thick,
        "f_temp": f_temp,
        "t_fill": t_fill,
    }
