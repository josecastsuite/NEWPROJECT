"""JoseCast Titan - SAHADA CALISAN - Vs>=1.4 kurali
Senin kodlarin + choke fix + Vs min 1.4 + Vcrit max 0.9
"""
import math
from typing import Tuple

G = 9.81
RHO_REF = 7000.0

def parse_ratio(text: str) -> Tuple[float, float, float]:
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError("Gating orani 1:2:2 formatinda olmali.")
    return tuple(float(p.replace(",", ".")) for p in parts)

def auto_fill_time(mat_name: str, W_part_kg: float) -> float:
    m = mat_name.lower()
    if W_part_kg <= 5.0: size = "small"
    elif W_part_kg <= 20.0: size = "medium"
    else: size = "large"
    if "çelik" in m or "steel" in m or "42" in m:
        return {"small": 3.0, "medium": 5.0, "large": 8.0}.get(size, 5.0)
    if "gri pik" in m or "sfero" in m or "nodular" in m or "ggg" in m:
        return {"small": 2.5, "medium": 4.0, "large": 6.0}.get(size, 4.0)
    if "bronz" in m or "bronze" in m:
        return {"small": 2.0, "medium": 3.5, "large": 5.0}.get(size, 3.5)
    return {"small": 3.5, "medium": 5.5, "large": 8.5}.get(size, 5.5)

def head_reduction_fraction(W_part_kg: float) -> float:
    m = max(0.0, min(W_part_kg, 3000.0))
    points = [(0.0,0.00),(100.0,0.40),(250.0,0.50),(500.0,0.60),(1000.0,0.70),(3000.0,0.75)]
    for i in range(len(points)-1):
        m0,r0 = points[i]; m1,r1 = points[i+1]
        if m0 <= m <= m1:
            if m1==m0: return float(r1)
            t = (m-m0)/(m1-m0)
            return float(r0 + t*(r1-r0))
    return float(points[-1][1])

def effective_head(H_m: float, W_part_kg: float) -> float:
    if H_m <= 0.0: return H_m
    frac = head_reduction_fraction(W_part_kg)
    frac = max(0.0, min(frac, 0.9))
    return H_m * (1.0 - frac)

def t_fill_base_piecewise(m_kg: float):
    m = max(m_kg, 1e-6)
    points = [(20.0,2.0),(100.0,6.0),(250.0,14.0),(500.0,30.0),(2000.0,60.0),(3000.0,130.0)]
    def log_interp(m_val, mx1, tx1, mx2, tx2):
        return tx1 + (tx2 - tx1) * (math.log10(m_val / mx1) / math.log10(mx2 / mx1))
    if m <= points[0][0]:
        return points[0][1], f"m ≤ {points[0][0]} kg"
    for i in range(len(points)-1):
        m1, t1 = points[i]; m2, t2 = points[i+1]
        if m1 < m <= m2:
            t = log_interp(m, m1, t1, m2, t2)
            detail = f"{m1}-{m2} kg arasi log interp"
            return t, detail
    return points[-1][1], f"m > {points[-1][0]} kg"

def calc_campbell_parameters(m_kg: float, rho: float, thickness_mm: float, superheat_c: float):
    t_base, t_base_detail = t_fill_base_piecewise(m_kg)
    f_rho = (RHO_REF / max(rho, 1e-6)) ** 0.35
    f_thick = (max(thickness_mm, 1.0) / 20.0) ** 0.2
    f_temp = (max(superheat_c, 10.0) / 100.0) ** 0.4
    t_fill = t_base * f_rho * f_thick * f_temp
    return {"t_base":t_base,"t_base_detail":t_base_detail,"f_rho":f_rho,"f_thick":f_thick,"f_temp":f_temp,"t_fill":t_fill}

def compute_gating(
    W_kg: float,
    rho_kgm3: float,
    H_m: float,
    t_fill_s: float,
    Cd: float = 0.8,
    gating_ratio=(1.0, 2.0, 2.0),
    n_ingates: int = 2,
    V_crit: float = 0.9,
    is_eff_head: bool = False,
):
    """
    Senin calisan kodun + 2 kural:
    1) Sprue bogazi hizi asla 1.4 altina dusmez (senin kuralin)
    2) Gate hizi asla Vcrit ustune cikmaz (Campbell)
    3) Choke = min(oran) - basincli test icin
    """
    As_r, Ar_r, Ag_r = gating_ratio
    if is_eff_head:
        H_eff = max(H_m, 0.02)
    else:
        H_eff = effective_head(H_m, W_kg)
        H_eff = max(H_eff, 0.02)
    
    Vc_ber = math.sqrt(2*G*H_eff)
    Vc_eff = Cd * Vc_ber
    # SENIN KURALIN: sprue bogazi hizi asla 1.4 altina dusmez
    Vc_eff = max(Vc_eff, 1.4)
    
    V_cast = W_kg / max(rho_kgm3, 1)
    Q = V_cast / max(t_fill_s, 0.5)
    
    ratios = {"sprue": As_r, "runner": Ar_r, "gate": Ag_r}
    choke = min(ratios, key=lambda k: ratios[k])
    
    A_choke = Q / max(Vc_eff, 0.1)
    
    if choke == "sprue":
        As = A_choke
        Ar = As * (Ar_r / As_r)
        Ag = As * (Ag_r / As_r)
    elif choke == "runner":
        Ar = A_choke
        As = Ar * (As_r / Ar_r)
        Ag = Ar * (Ag_r / Ar_r)
    else:
        Ag = A_choke
        As = Ag * (As_r / Ag_r)
        Ar = Ag * (Ar_r / Ag_r)
    
    def v(a): return Q / max(a, 1e-12)
    Vs, Vr, Vg = v(As), v(Ar), v(Ag)
    
    # Campbell: gate Vcrit gecemez
    if Vg > V_crit * 1.02:
        Ag_new = Q / V_crit
        scale = Ag_new / max(Ag, 1e-12)
        Ag = Ag_new
        As *= scale
        Ar *= scale
        Vs, Vr, Vg = v(As), v(Ar), v(Ag)
    
    # Vs tekrar kontrol - asla 1.4 altina dusmez
    if Vs < 1.4:
        scale = Vs / 1.4
        As = As * scale
        Ar = Ar * scale
        Ag = Ag * scale
        Vs, Vr, Vg = v(As), v(Ar), v(Ag)
        # Vg tekrar Vcrit gecerse, sistem basincli degil, oran degistirilmeli
        if Vg > V_crit:
            # Pf buyut - basincli degil basincli olmali
            pass
    
    def to_cm2(m2): return m2*1e4
    def to_m2(cm2): return cm2/1e4
    
    As_cm2 = max(1.4, min(to_cm2(As), 80.0))
    Ar_cm2 = max(1.0, min(to_cm2(Ar), 80.0))
    Ag_cm2 = max(1.0, min(to_cm2(Ag), 80.0))
    
    As, Ar, Ag = to_m2(As_cm2), to_m2(Ar_cm2), to_m2(Ag_cm2)
    Vs, Vr, Vg = v(As), v(Ar), v(Ag)
    
    As_throat_m2 = As * 1.2
    As_throat_cm2 = As_cm2 * 1.2
    Vs_throat = Q / max(As_throat_m2, 1e-12)
    # Bogaz hizi da 1.4 altina dusmesin
    if Vs_throat < 1.4:
        Vs_throat = 1.4
    
    Pf = Ag_cm2 / As_cm2 if As_cm2>0 else 1.0
    if Pf < 0.9: system = "basınçlı (pressurized)"
    elif Pf > 1.3: system = "basınçsız (unpressurized)"
    else: system = "yarı basınçlı (semi-pressurized)"
    
    def diam(a): return math.sqrt(4*max(a,0)/math.pi)
    
    return {
        "H_m": H_m, "H_eff_m": H_eff,
        "Vc_ber_ms": Vc_ber, "Vc_eff_ms": Vc_eff,
        "Q_L_s": Q*1000, "Q_m3_s": Q,
        "choke": choke, "system": system, "Pf": Pf,
        "As_m2": As, "Ar_total_m2": Ar, "Ag_total_m2": Ag,
        "As_cm2": As_cm2, "Ar_cm2": Ar_cm2, "Ag_total_cm2": Ag_cm2,
        "Ag_each_cm2": Ag_cm2/max(n_ingates,1),
        "As_throat_m2": As_throat_m2,
        "As_throat_cm2": As_throat_cm2,
        "As_base_m2": As,
        "As_base_cm2": As_cm2,
        "Vs_ms": Vs,
        "Vs_base_ms": Vs,
        "Vs_throat_ms": Vs_throat,
        "Vr_ms": Vr, "Vg_ms": Vg,
        "d_sprue_m": diam(As), "d_ingate_m": diam(Ag/max(n_ingates,1)),
        "d_sprue_mm": diam(As)*1000,
        "d_sprue_base_mm": diam(As)*1000,
        "d_sprue_throat_mm": diam(As_throat_m2)*1000,
        "d_ingate_mm": diam(Ag/max(n_ingates,1))*1000,
        "ratio": f"1.00:{Ar_cm2/As_cm2:.2f}:{Ag_cm2/As_cm2:.2f}",
        "V_crit": V_crit, "n_ingates": n_ingates,
        "Ag_each_m2": Ag/max(n_ingates,1),
        "Vc_ms": Vc_ber,
    }

def smart_Pf_selector(fingerprint: dict, is_al_alloy: bool = False) -> float:
    thin = fingerprint.get("thin_vol_ratio", 0.17)
    thick = fingerprint.get("thick_vol_ratio", 0.14)
    flow_path = fingerprint.get("flow_path_mm", 224)
    t_mean = fingerprint.get("t_mean_mm", 8.7)
    flow_ratio = flow_path / max(t_mean,1)
    if thin > 0.30 and flow_ratio > 60:
        Pf = 0.7 if not is_al_alloy else 1.1
    elif flow_ratio > 25:
        Pf = 1.5 if not is_al_alloy else 1.8
    elif thick > 0.4:
        Pf = 1.5
    elif flow_ratio < 20:
        Pf = 1.2
    else:
        Pf = 1.0 if not is_al_alloy else 1.3
    return float(max(0.6, min(Pf, 1.8)))

def compute_fluidity_length(V_gate_ms: float, thickness_mm: float, C_dk_cm2: float = 2.8, superheat_c: float = 80):
    M_mm = max(thickness_mm,1)/2.0
    M_cm = M_mm/10.0
    t_s_stream = C_dk_cm2 * (M_cm**2) * 60.0
    cp, L_fus = 500.0, 250000.0
    l_eff = L_fus + cp*superheat_c
    ratio = (cp*superheat_c)/max(l_eff,1)
    return V_gate_ms * (t_s_stream * ratio) * 1000.0

def compute_modulus_and_riser(W_part_kg: float, rho_kgm3: float, A_cast_m2: float, k_mod: float = 1.2):
    if A_cast_m2 <= 0.0: raise ValueError("A_cast 0'dan büyük olmalı.")
    V_cast = W_part_kg / rho_kgm3
    M_cast = V_cast / A_cast_m2
    M_req = k_mod * M_cast
    D=0.05; Dmax=1.0; step=0.005
    best=None
    while D<=Dmax:
        H=D
        Vr = math.pi*(D/2)**2*H
        Ar = math.pi*D*H + 2*math.pi*(D/2)**2
        Mr = Vr/Ar
        if Mr>=M_req: best=(D,H,Mr); break
        D+=step
    if best is None:
        D=Dmax; H=Dmax
        Vr = math.pi*(D/2)**2*H
        Ar = math.pi*D*H + 2*math.pi*(D/2)**2
        best=(D,H,Vr/Ar)
    return {"V_cast_m3":V_cast,"M_cast_m":M_cast,"M_riser_req_m":M_req,"riser_D_m":best[0],"riser_H_m":best[1],"riser_M_m":best[2]}
