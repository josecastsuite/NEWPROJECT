"""JoseCast v11 cumulative mold-erosion model.

Combines Finnie-Bitter solid-particle erosion, Darcy-Forchheimer wall shear,
Campbell bifilm / critical velocity, thermal binder degradation and a Miner-like
cumulative damage accumulator.  Per-gating-element risk is smoothly interpolated
along each gate body (centreline lerp) and Gaussian boundary injection blends the
gating field into the part surface.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from core.types import Body, BodyType, GatingNode

# Finnie-Bitter angular factor peaks at 30 deg from the wall surface and is
# zero at 0 (grazing) and 90 (normal) degrees.  alpha is the angle between the
# velocity vector and the wall *surface*.
_PEAK_ANGLE_RAD = math.radians(30.0)
_F_ANGLE_PEAK = (
    math.sin(_PEAK_ANGLE_RAD) * math.cos(_PEAK_ANGLE_RAD) ** 3
)


def _f_angle_from_sin(sin_alpha: np.ndarray) -> np.ndarray:
    """Finnie-Bitter angular weighting, 1.0 at 30 deg from surface."""
    s = np.clip(np.asarray(sin_alpha, dtype=np.float64), 0.0, 1.0)
    alpha = np.arcsin(s)
    f = np.sin(alpha) * np.cos(alpha) ** 3
    return np.clip(f / max(_F_ANGLE_PEAK, 1e-12), 0.0, 1.0)


def _binder_deg(T: np.ndarray) -> np.ndarray:
    """Bentonite binder degradation with metal-mould interface temperature."""
    T = np.asarray(T, dtype=np.float64)
    t0, t1 = 400.0, 700.0
    deg = np.where(
        T <= t0,
        1.0,
        np.where(T >= t1, 0.2, 1.0 - 0.8 * (T - t0) / (t1 - t0)),
    )
    return np.clip(deg, 0.2, 1.0)


def _turb_factor(Re: Any) -> Any:
    """Turbulence amplification from pore/pipe Reynolds number."""
    Re = np.asarray(Re, dtype=np.float64)
    Tu = np.where(Re > 5.0, 0.16 * np.power(Re / 5.0, -0.125), 0.0)
    Tu = np.clip(Tu, 0.0, 0.5)
    return 1.0 + 5.0 * Tu


def _kozeny_k_eff(mold: Any) -> float:
    """Intrinsic permeability corrected by grain size, porosity, binder, moisture."""
    d_p_mm = float(getattr(mold, "particle_size_mm", 0.0))
    afs = float(getattr(mold, "afs_grain_size", 0.0))
    if d_p_mm <= 0.0 and afs > 0.0:
        d_p_mm = 6.45 / math.sqrt(afs)
    if d_p_mm <= 0.0:
        d_p_mm = 0.25
    d_p_m = d_p_mm / 1000.0

    K_inf = float(getattr(mold, "K_inf", 1e-11))
    phi = min(max(float(getattr(mold, "phi_mold", 0.35)), 0.01), 0.99)
    binder = float(getattr(mold, "binder_percent", 0.0))
    moisture = float(getattr(mold, "moisture_percent", 0.0))

    d_ref = 2.5e-4  # 0.25 mm reference grain
    grain_factor = (d_p_m / d_ref) ** 2

    phi_factor = (phi / 0.35) ** 3 * ((1.0 - 0.35) / (1.0 - phi)) ** 2
    phi_factor = max(phi_factor, 0.1)

    # Higher binder / moisture blocks pore throats.
    perm_factor = max(0.1, 1.0 - 0.015 * binder - 0.025 * moisture)

    K_eff = max(K_inf, 1e-18) * grain_factor * phi_factor * perm_factor
    return max(K_eff, 1e-18)


def _sample_scalar(
    field: Optional[np.ndarray],
    pos_mm: Tuple[float, float, float],
    origin_mm: np.ndarray,
    dx_mm: float,
    order: int = 1,
) -> Optional[float]:
    """Trilinear (order=1) or nearest (order=0) sample of a 3-D scalar field."""
    if field is None or field.size == 0 or dx_mm <= 0.0 or origin_mm is None:
        return None
    pos = np.asarray(pos_mm, dtype=np.float64).reshape(3)
    origin = np.asarray(origin_mm, dtype=np.float64).reshape(3)
    idx = ((pos - origin) / dx_mm).reshape(3, 1)
    try:
        val = float(ndimage.map_coordinates(field, idx, order=order, mode="nearest"))
    except Exception:
        val = None
    return val


def _sample_vector(
    vec: Optional[np.ndarray],
    pos_mm: Tuple[float, float, float],
    origin_mm: np.ndarray,
    dx_mm: float,
) -> Optional[np.ndarray]:
    """Sample a (3, nx, ny, nz) vector field at a point."""
    if vec is None or vec.size == 0 or dx_mm <= 0.0 or origin_mm is None:
        return None
    if vec.ndim != 4 or vec.shape[0] != 3:
        return None
    pos = np.asarray(pos_mm, dtype=np.float64).reshape(3)
    origin = np.asarray(origin_mm, dtype=np.float64).reshape(3)
    idx = ((pos - origin) / dx_mm).reshape(3, 1)
    try:
        comps = [float(ndimage.map_coordinates(vec[i], idx, order=1, mode="nearest")) for i in range(3)]
    except Exception:
        return None
    return np.asarray(comps, dtype=np.float64)


def _section_risk(
    v: float,
    A_m2: float,
    A_up_m2: float,
    rho_m: float,
    mu: float,
    sigma: float,
    v_crit_eff: float,
    tau_crit: float,
    p_cap: float,
    K_eff: float,
    d_p_m: float,
    T_local: Optional[float] = None,
    fill_time_local: float = 0.0,
    t_fill_total: float = 1.0,
    f_angle: float = 1.0,
) -> float:
    """Power-law / soft-max building block for one 1-D gating section or voxel."""
    v = float(v)
    if v <= 1e-9 or A_m2 <= 1e-12 or v_crit_eff <= 1e-9 or tau_crit <= 0.0:
        return 0.0

    if T_local is not None and T_local > 0.0:
        tau_crit = tau_crit * float(_binder_deg(np.asarray([T_local]))[0])
    tau_crit = max(tau_crit, 1e3)

    D_h = math.sqrt(4.0 * A_m2 / math.pi)
    Re = rho_m * v * D_h / max(mu, 1e-12)

    eps = d_p_m
    if Re < 2300.0:
        f = 64.0 / max(Re, 1e-9)
    else:
        rough = eps / (3.7 * D_h) + 5.74 / (Re ** 0.9)
        if rough > 0.0 and math.log10(rough) != 0.0:
            f = 0.25 / (math.log10(rough) ** 2)
        else:
            f = 0.02
    f = min(max(f, 0.0001), 0.5)

    tau_w = (f / 8.0) * rho_m * v * v
    p_dyn = 0.5 * rho_m * v * v
    Re_K = rho_m * v * math.sqrt(K_eff) / max(mu, 1e-12)

    # Power-law damage terms (no hard threshold).
    D_shear = (tau_w / tau_crit) ** 1.5
    D_vel = (v / v_crit_eff) ** 2.0
    D_pen = (p_dyn / p_cap) ** 2.0
    D_ReK = (Re_K / 5.0) ** 1.5

    # Campbell / bifilm loss from sudden contraction/expansion.
    if A_up_m2 > 0.0 and abs(A_up_m2 - A_m2) > 1e-12:
        if A_up_m2 > A_m2:
            K_loss = 0.5 * (1.0 - A_m2 / A_up_m2)
        else:
            K_loss = (1.0 - A_up_m2 / A_m2) ** 2
    else:
        K_loss = 0.0
    D_loss = K_loss * (v / v_crit_eff) ** 2.0

    # Finnie-Bitter erosion with velocity exponent 2.5 and angular factor.
    D_FB = (v / v_crit_eff) ** 2.5 * max(f_angle, 0.0)

    turb = float(_turb_factor(np.asarray([Re]))[0])

    exposure = 1.0
    if t_fill_total > 1e-9:
        exposure = max(0.0, min(1.0, 1.0 - fill_time_local / t_fill_total))

    # p=4 soft-max instead of a hard max.
    combined = (D_shear ** 4 + D_vel ** 4 + D_pen ** 4 + D_ReK ** 4 + D_loss ** 4 + D_FB ** 4) ** 0.25
    return float(1.0 - math.exp(-turb * combined * exposure))


def _node_f_angle(
    n: GatingNode,
    velocity_m_s: Optional[np.ndarray],
    sdf: Optional[np.ndarray],
    origin_mm: np.ndarray,
    dx_mm: float,
) -> float:
    """Impact angle factor at a gating node using the local velocity and wall normal."""
    if velocity_m_s is None or sdf is None or dx_mm <= 0.0 or origin_mm is None:
        return 1.0
    vel = _sample_vector(velocity_m_s, tuple(n.centroid_mm), origin_mm, dx_mm)
    if vel is None:
        return 1.0
    v_norm = float(np.linalg.norm(vel))
    if v_norm <= 1e-9:
        return 1.0

    idx = (np.asarray(n.centroid_mm, dtype=np.float64) - np.asarray(origin_mm, dtype=np.float64).reshape(3)) / dx_mm
    idx = idx.reshape(3, 1)
    try:
        grad = np.stack(
            [float(ndimage.map_coordinates(sdf, idx + np.array([[0], [0], [0]]), order=1, mode="nearest")) for _ in range(3)],
            axis=0,
        )
    except Exception:
        return 1.0
    # Finite-difference gradient is more robust than np.gradient at a single point.
    step = 1.0
    grads = np.zeros(3, dtype=np.float64)
    idx = idx.reshape(3)
    for ax in range(3):
        plus = idx.copy()
        minus = idx.copy()
        plus[ax] += step
        minus[ax] -= step
        try:
            vplus = float(ndimage.map_coordinates(sdf, plus.reshape(3, 1), order=1, mode="nearest"))
            vminus = float(ndimage.map_coordinates(sdf, minus.reshape(3, 1), order=1, mode="nearest"))
            grads[ax] = (vplus - vminus) / (2.0 * step)
        except Exception:
            pass
    grad_norm = float(np.linalg.norm(grads))
    if grad_norm <= 1e-12:
        return 1.0
    # sdf increases inward in metal, so grad points inward (towards the wall normal).
    normal = grads / grad_norm
    sin_alpha = abs(float(np.dot(vel / v_norm, normal)))
    return float(_f_angle_from_sin(np.asarray([sin_alpha]))[0])


def _mask_gaussian_blur(arr: np.ndarray, mask: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """Mask-aware Gaussian smoothing; values do not leak outside the metal domain."""
    if sigma <= 0.0 or arr is None or mask is None:
        return arr
    arr = np.asarray(arr, dtype=np.float64)
    mask = mask.astype(bool, copy=False)
    blurred = ndimage.gaussian_filter(arr * mask, sigma=sigma, mode="constant")
    weight = ndimage.gaussian_filter(mask.astype(np.float64), sigma=sigma, mode="constant")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(mask & (weight > 1e-9), blurred / weight, arr)
    return out


def _body_area_cm2(body: Body) -> float:
    """Characteristic cross-sectional area of a gate body (cm^2)."""
    user = float(getattr(body, "section_area_cm2", 0.0))
    if user > 0.0:
        return user
    try:
        from core.gating import _characteristic_cross_section_area, _flow_axis

        axis = _flow_axis(body.mesh)
        return float(_characteristic_cross_section_area(body.mesh, axis, n=10)) / 100.0
    except Exception:
        # Fallback: area of a disc with the body radius.
        import trimesh

        try:
            tm = trimesh.Trimesh(vertices=body.vertices, faces=body.faces)
            r = math.sqrt(tm.area / math.pi) * 10.0  # mm -> cm radius? area mm^2 -> circle radius mm
            return math.pi * (r / 10.0) ** 2  # cm^2
        except Exception:
            return 1.0


def compute_gate_erosion_risk_v2(
    gating_nodes: Optional[List[GatingNode]],
    is_metal: np.ndarray,
    alloy: Any,
    mold: Any,
    body_index: Optional[np.ndarray] = None,
    bodies: Optional[List[Body]] = None,
    sdf: Optional[np.ndarray] = None,
    dx_mm: float = 1.0,
    origin_mm: Optional[np.ndarray] = None,
    velocity_m_s: Optional[np.ndarray] = None,
    fill_time: Optional[np.ndarray] = None,
    temperature: Optional[np.ndarray] = None,
    body_risk_out: Optional[Dict[str, float]] = None,
    impingement_out: Optional[List[Tuple[str, float, Tuple[float, float, float], float]]] = None,
) -> np.ndarray:
    """Per-gating-element cumulative erosion with centreline lerp and Gaussian boundary injection."""
    risk = np.zeros_like(is_metal, dtype=np.float64)
    if not gating_nodes or body_index is None or bodies is None:
        return risk

    is_sand = bool(getattr(mold, "is_sand", True))
    if not is_sand:
        return risk

    metal = is_metal.astype(bool, copy=False)
    if not metal.any():
        return risk

    shape = is_metal.shape

    # Material constants
    rho_m = float(getattr(alloy, "rho_kg_m3", 7000.0))
    mu = max(float(getattr(alloy, "viscosity_pa_s", 0.005)), 1e-12)
    sigma = float(getattr(alloy, "surface_tension_n_m", 1.0))
    if sigma <= 0.0:
        sigma = 1.0

    afs = float(getattr(mold, "afs_grain_size", 0.0))
    d_p_mm = float(getattr(mold, "particle_size_mm", 0.0))
    if d_p_mm <= 0.0 and afs > 0.0:
        d_p_mm = 6.45 / math.sqrt(afs)
    if d_p_mm <= 0.0:
        d_p_mm = 0.25
    d_p_m = d_p_mm / 1000.0

    rigidity = min(max(float(getattr(mold, "mold_rigidity_factor", 0.5)), 0.0), 1.0)
    binder = float(getattr(mold, "binder_percent", 0.0))
    moisture = float(getattr(mold, "moisture_percent", 0.0))
    compact = float(getattr(mold, "compactability_percent", 0.0))

    rho_sand = 2650.0
    g = 9.81
    theta_c = 0.03
    tau_gravity = theta_c * abs(rho_sand - rho_m) * g * d_p_m
    binder_factor = max(binder / 2.0, 0.5) if binder > 0.0 else 1.0
    compact_factor = compact / 45.0 if compact > 0.0 else 1.0
    moisture_factor = 1.0 if moisture < 0.5 else max(0.3, min(4.0 / moisture, 2.0))
    tau_cohesion = 20e3 * binder_factor * compact_factor * moisture_factor * (0.3 + 0.7 * rigidity)
    tau_crit = max(tau_gravity, tau_cohesion, 50e3 * rigidity)
    tau_crit = max(tau_crit, 1e3)

    v_crit = float(getattr(alloy, "critical_entrainment_velocity_m_s", 0.5))
    v_crit_eff = max(v_crit * (0.6 + 0.4 * rigidity), 0.1)
    p_cap = 4.0 * sigma / max(d_p_m, 1e-7)

    K_eff = _kozeny_k_eff(mold)

    # Effective interface temperature with surface burn correction.
    T_eff = None
    if temperature is not None and sdf is not None and temperature.shape == shape and sdf.shape == shape:
        T_metal = np.asarray(temperature, dtype=np.float64).reshape(shape)
        sdf_a = np.asarray(sdf, dtype=np.float64).reshape(shape)
        delta_T = max(2.0 * dx_mm, 1.0)  # mm, thermal boundary layer proxy
        T_eff = T_metal + (float(getattr(alloy, "t_pour_c", 700.0)) - T_metal) * np.exp(
            -np.clip(sdf_a, 0.0, None) / delta_T
        )

    t_fill_total = 1.0
    if fill_time is not None and fill_time.shape == shape:
        ft = np.asarray(fill_time, dtype=np.float64).reshape(shape)
        ft_metal = np.where(metal, ft, 0.0)
        t_fill_total = float(np.max(ft_metal)) if metal.any() else 1.0
        t_fill_total = max(t_fill_total, 1e-9)

    body_by_name = {b.name: b for b in bodies if b is not None and getattr(b, "name", "")}
    part_body = max(
        (b for b in bodies if b is not None and getattr(b, "body_type", None) == BodyType.PART),
        key=lambda b: float(getattr(b, "volume_cm3", 0.0) or 0.0),
        default=None,
    )

    gate_types = frozenset(
        [
            BodyType.SPRUE_THROAT,
            BodyType.SPRUE,
            BodyType.RUNNER,
            BodyType.DISTRIBUTOR,
            BodyType.INGATE,
            BodyType.POURING_BASIN,
            BodyType.COOLING_SPRUE,
            BodyType.FILTER,
            BodyType.CURUFLUK,
        ]
    )

    # Parse nodes and collect connection data.
    down_areas: Dict[str, float] = {}
    parsed: List[Tuple[GatingNode, str, str]] = []
    up_Q: Dict[str, float] = {}
    down_Q: Dict[str, float] = {}
    up_A: Dict[str, float] = {}
    down_A: Dict[str, List[float]] = {}

    for n in gating_nodes:
        name = str(n.name or "")
        if "→" in name:
            up, down = [s.strip() for s in name.split("→", 1)]
        else:
            up, down = "", name.strip()
        parsed.append((n, up, down))
        A = float(n.section_area_cm2)
        Q = float(n.flow_rate_m3_s)
        if down:
            down_areas[down] = max(down_areas.get(down, 0.0), A)
            down_Q[down] = down_Q.get(down, 0.0) + Q
            down_A.setdefault(down, []).append(A)
        if up:
            up_Q[up] = up_Q.get(up, 0.0) + Q
            up_A[up] = max(up_A.get(up, 0.0), A)

    # First pass: node risk values.
    node_infos = []
    for n, up, down in parsed:
        A_cm2 = float(n.section_area_cm2)
        if A_cm2 <= 0.0:
            continue
        A_up_cm2 = down_areas.get(up, A_cm2) if up else A_cm2
        A_m2 = A_cm2 * 1e-4
        A_up_m2 = A_up_cm2 * 1e-4
        v = max(float(n.velocity_m_s), float(getattr(n, "max_velocity_m_s", 0.0)))

        T_local = None
        if T_eff is not None:
            T_local = _sample_scalar(T_eff, tuple(n.centroid_mm), origin_mm, dx_mm)
        fill_time_local = 0.0
        if fill_time is not None:
            val = _sample_scalar(fill_time, tuple(n.centroid_mm), origin_mm, dx_mm, order=0)
            if val is not None and np.isfinite(val):
                fill_time_local = float(val)

        f_angle = _node_f_angle(n, velocity_m_s, sdf, origin_mm, dx_mm)
        node_risk = _section_risk(
            v,
            A_m2,
            A_up_m2,
            rho_m,
            mu,
            sigma,
            v_crit_eff,
            tau_crit,
            p_cap,
            K_eff,
            d_p_m,
            T_local=T_local,
            fill_time_local=fill_time_local,
            t_fill_total=t_fill_total,
            f_angle=f_angle,
        )
        if node_risk <= 1e-9:
            continue
        info = {
            "n": n,
            "up": up,
            "down": down,
            "A_cm2": A_cm2,
            "risk": node_risk,
            "centroid": np.asarray(n.centroid_mm, dtype=np.float64),
        }
        node_infos.append(info)

    up_nodes: Dict[str, List[dict]] = {}
    down_nodes: Dict[str, List[dict]] = {}
    for info in node_infos:
        if info["up"]:
            up_nodes.setdefault(info["up"], []).append(info)
        if info["down"]:
            down_nodes.setdefault(info["down"], []).append(info)

    # Coordinate grid (mm).
    coords_mm = None
    if origin_mm is not None and dx_mm > 0.0:
        idx = np.indices(shape, dtype=np.float64)
        coords_mm = np.stack([idx[i] * dx_mm + float(origin_mm[i]) for i in range(3)], axis=0)

    # Per-body centreline lerp.
    wall_decay = max(2.0 * d_p_mm, 0.5 * dx_mm)
    sdf_a = np.asarray(sdf, dtype=np.float64).reshape(shape) if sdf is not None and sdf.shape == shape else None

    for b in bodies:
        if b is None or b.body_type not in gate_types or b.name not in body_by_name:
            continue
        if body_index is None or body_index.shape != shape:
            continue
        body_mask = (body_index == b.index) & metal
        if not body_mask.any():
            continue

        Q_total = max(up_Q.get(b.name, 0.0), down_Q.get(b.name, 0.0))
        if Q_total <= 1e-12:
            continue
        A_body_cm2 = _body_area_cm2(b)
        connected = down_A.get(b.name, [])
        if connected:
            A_body_cm2 = max(A_body_cm2, min(connected))
        A_body_m2 = max(A_body_cm2, 1e-6) * 1e-4
        A_up_cm2 = up_A.get(b.name, A_body_cm2)
        A_up_m2 = max(A_up_cm2, 1e-6) * 1e-4
        v_body = Q_total / A_body_m2

        T_local = None
        if T_eff is not None:
            T_local = _sample_scalar(T_eff, tuple(b.center), origin_mm, dx_mm)
        fill_time_local = 0.0
        if fill_time is not None:
            val = _sample_scalar(fill_time, tuple(b.center), origin_mm, dx_mm, order=0)
            if val is not None and np.isfinite(val):
                fill_time_local = float(val)

        body_risk_val = _section_risk(
            v_body,
            A_body_m2,
            A_up_m2,
            rho_m,
            mu,
            sigma,
            v_crit_eff,
            tau_crit,
            p_cap,
            K_eff,
            d_p_m,
            T_local=T_local,
            fill_time_local=fill_time_local,
            t_fill_total=t_fill_total,
            f_angle=0.5,  # representative straight-channel Finnie factor
        )
        if body_risk_out is not None:
            body_risk_out[b.name] = max(body_risk_out.get(b.name, 0.0), body_risk_val)

        # Upstream / downstream risks and centroids.
        up_infos = up_nodes.get(b.name, [])
        down_infos = down_nodes.get(b.name, [])

        if up_infos:
            risk_up = float(np.mean([inf["risk"] for inf in up_infos]))
            up_centroid = np.mean([inf["centroid"] for inf in up_infos], axis=0)
        else:
            risk_up = body_risk_val
            up_centroid = np.asarray(b.center, dtype=np.float64)

        if down_infos:
            risk_down = float(np.mean([inf["risk"] for inf in down_infos]))
            down_centroid = np.mean([inf["centroid"] for inf in down_infos], axis=0)
        else:
            risk_down = body_risk_val
            down_centroid = np.asarray(b.center, dtype=np.float64)

        flow_vec = down_centroid - up_centroid
        L = float(np.linalg.norm(flow_vec))
        if L < 1e-3:
            try:
                from core.gating import _flow_axis

                axis = _flow_axis(b.mesh)
                L = float(np.ptp(np.asarray(b.vertices) @ axis))
                u = np.asarray(axis, dtype=np.float64)
                mid = np.asarray(b.center, dtype=np.float64)
                up_centroid = mid - 0.5 * L * u
                down_centroid = mid + 0.5 * L * u
                flow_vec = down_centroid - up_centroid
                L = float(np.linalg.norm(flow_vec))
            except Exception:
                L = 0.0

        coords = coords_mm[:, body_mask]  # (3, N)
        if L > 1e-3:
            u = flow_vec / L
            t = np.clip(np.dot(u, (coords - up_centroid.reshape(3, 1))) / L, 0.0, 1.0)
            risk_vals = risk_up * (1.0 - t) + risk_down * t
        else:
            risk_vals = np.full(coords.shape[1], body_risk_val, dtype=np.float64)

        if sdf_a is not None:
            sdf_vals = sdf_a[body_mask]
            profile = 1.0 + 0.1 * np.exp(-np.clip(sdf_vals, 0.0, None) / wall_decay)
            risk_vals = risk_vals * profile

        risk[body_mask] = np.maximum(risk[body_mask], risk_vals)

    # Gaussian boundary injection: node hotspots on gate bodies and part surface.
    for info in node_infos:
        n = info["n"]
        up = info["up"]
        down = info["down"]
        centroid = info["centroid"]
        A_cm2 = info["A_cm2"]
        node_risk = info["risk"]

        if coords_mm is None or node_risk <= 1e-9:
            continue

        jet_diam_mm = math.sqrt(4.0 * A_cm2 / math.pi)
        sigma_mm = max(jet_diam_mm / 3.0, dx_mm)
        radius_mm = max(3.0 * sigma_mm, 4.0 * dx_mm)
        sigma2 = max(sigma_mm * sigma_mm, 1e-6)
        dist2 = np.sum((coords_mm - centroid.reshape(3, 1, 1, 1)) ** 2, axis=0)
        g_blob = np.exp(-dist2 / (2.0 * sigma2))

        # Upstream gate body hotspot.
        target = body_by_name.get(up)
        if target is not None and target.body_type in gate_types and body_index is not None:
            mask = (body_index == target.index) & metal & (dist2 <= radius_mm * radius_mm)
            if mask.any():
                risk[mask] = np.maximum(risk[mask], node_risk * g_blob[mask])

        # Downstream impingement surface (part or next gate).
        down_type = None
        if down:
            down_type = getattr(body_by_name.get(down), "body_type", None)
        is_part = down == "Parça" or down == "PART" or down_type == BodyType.PART
        if is_part and part_body is not None and sdf_a is not None and body_index is not None:
            wall_layer = sdf_a <= max(4.0 * d_p_mm, 1.5 * dx_mm)
            mask = (
                (body_index == part_body.index)
                & metal
                & wall_layer
                & (dist2 <= radius_mm * radius_mm)
            )
            if mask.any():
                risk[mask] = np.maximum(risk[mask], node_risk * g_blob[mask])
            if impingement_out is not None:
                impingement_out.append(
                    (
                        part_body.name,
                        float(node_risk),
                        tuple(float(x) for x in centroid),
                        float(radius_mm),
                    )
                )
        elif down and body_by_name.get(down) is not None and body_index is not None:
            target = body_by_name[down]
            if target.body_type in gate_types:
                mask = (body_index == target.index) & metal & (dist2 <= radius_mm * radius_mm)
                if mask.any():
                    risk[mask] = np.maximum(risk[mask], node_risk * g_blob[mask])

    return np.clip(risk, 0.0, 1.0)


def compute_erosion_risk_v2(
    velocity_magnitude: Optional[np.ndarray],
    is_metal: np.ndarray,
    alloy: Any,
    mold: Any,
    velocity_m_s: Optional[np.ndarray] = None,
    sdf: Optional[np.ndarray] = None,
    dx_mm: float = 1.0,
    gating_nodes: Optional[List[GatingNode]] = None,
    body_index: Optional[np.ndarray] = None,
    bodies: Optional[List[Body]] = None,
    origin_mm: Optional[np.ndarray] = None,
    fill_time: Optional[np.ndarray] = None,
    temperature: Optional[np.ndarray] = None,
    body_risk_out: Optional[Dict[str, float]] = None,
    impingement_out: Optional[List[Tuple[str, float, Tuple[float, float, float], float]]] = None,
) -> np.ndarray:
    """Per-voxel cumulative mold erosion with Finnie-Bitter / Darcy-Forchheimer physics."""
    risk = np.zeros_like(is_metal, dtype=np.float64)
    if velocity_magnitude is None or velocity_magnitude.size == 0:
        return risk

    is_sand = bool(getattr(mold, "is_sand", True))
    if not is_sand:
        return risk

    shape = is_metal.shape
    metal = is_metal.astype(bool, copy=False)
    n_total = int(np.prod(shape))

    if velocity_magnitude.size != n_total:
        return risk

    # Material constants
    rho_m = float(getattr(alloy, "rho_kg_m3", 7000.0))
    mu = max(float(getattr(alloy, "viscosity_pa_s", 0.005)), 1e-12)
    sigma = float(getattr(alloy, "surface_tension_n_m", 1.0))
    if sigma <= 0.0:
        sigma = 1.0

    afs = float(getattr(mold, "afs_grain_size", 0.0))
    d_p_mm = float(getattr(mold, "particle_size_mm", 0.0))
    if d_p_mm <= 0.0 and afs > 0.0:
        d_p_mm = 6.45 / math.sqrt(afs)
    if d_p_mm <= 0.0:
        d_p_mm = 0.25
    d_p_m = d_p_mm / 1000.0

    rigidity = min(max(float(getattr(mold, "mold_rigidity_factor", 0.5)), 0.0), 1.0)
    binder = float(getattr(mold, "binder_percent", 0.0))
    moisture = float(getattr(mold, "moisture_percent", 0.0))
    compact = float(getattr(mold, "compactability_percent", 0.0))

    rho_sand = 2650.0
    g = 9.81
    theta_c = 0.03
    tau_gravity = theta_c * abs(rho_sand - rho_m) * g * d_p_m
    binder_factor = max(binder / 2.0, 0.5) if binder > 0.0 else 1.0
    compact_factor = compact / 45.0 if compact > 0.0 else 1.0
    moisture_factor = 1.0 if moisture < 0.5 else max(0.3, min(4.0 / moisture, 2.0))
    tau_cohesion = 20e3 * binder_factor * compact_factor * moisture_factor * (0.3 + 0.7 * rigidity)
    tau_crit = max(tau_gravity, tau_cohesion, 50e3 * rigidity)
    tau_crit = max(tau_crit, 1e3)

    v_crit = float(getattr(alloy, "critical_entrainment_velocity_m_s", 0.5))
    v_crit_eff = max(v_crit * (0.6 + 0.4 * rigidity), 0.1)
    p_cap = 4.0 * sigma / max(d_p_m, 1e-7)

    K_eff = _kozeny_k_eff(mold)

    v_mag = np.asarray(velocity_magnitude, dtype=np.float64).reshape(shape)
    v_mag = np.nan_to_num(v_mag, nan=0.0, posinf=0.0, neginf=0.0)

    # Wall-adjacent free-stream velocity (maximum in 3x3 neighbourhood).
    sdf_a = None
    wall_enhance = np.ones(shape, dtype=np.float64)
    if sdf is not None and sdf.shape == shape and dx_mm > 0.0:
        sdf_a = np.asarray(sdf, dtype=np.float64).reshape(shape)
        decay = max(2.0 * d_p_mm, 0.5 * dx_mm)
        wall_enhance = 1.0 + 0.1 * np.exp(-np.clip(sdf_a, 0.0, None) / max(decay, 1e-3))
        wall_layer = sdf_a <= max(2.0 * d_p_mm, 0.5 * dx_mm)
        if wall_layer.any():
            v_max_neigh = ndimage.maximum_filter(v_mag, size=3)
            v_eff = np.where(wall_layer & (v_max_neigh > v_mag), v_max_neigh, v_mag)
        else:
            v_eff = v_mag.copy()
    else:
        v_eff = v_mag.copy()

    # Normal velocity component and Finnie angle from the 3-D velocity field.
    v_n = v_eff.copy()
    f_angle = np.ones(shape, dtype=np.float64)
    if velocity_m_s is not None and velocity_m_s.size == 3 * n_total and sdf_a is not None and dx_mm > 0.0:
        vel = np.asarray(velocity_m_s, dtype=np.float64)
        if vel.ndim == 1:
            vel = vel.reshape((3,) + shape)
        elif vel.ndim == 4 and vel.shape[0] == 3 and vel.shape[1:] == shape:
            pass
        else:
            vel = None
        if vel is not None:
            grads = np.gradient(sdf_a, dx_mm)
            grad = np.stack(grads, axis=0)
            grad_norm = np.linalg.norm(grad, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                normal = grad / np.where(grad_norm > 1e-12, grad_norm, 1.0)
            normal = normal.reshape(3, -1)
            vel_flat = vel.reshape(3, -1)
            v_mag_flat = np.linalg.norm(vel_flat, axis=0)
            v_dot_n = np.einsum("ij,ij->j", vel_flat, normal)
            v_n_flat = np.abs(v_dot_n)
            sin_alpha = v_n_flat / np.maximum(v_mag_flat, 1e-9)
            sin_alpha = np.clip(sin_alpha, 0.0, 1.0)
            f_angle_flat = _f_angle_from_sin(sin_alpha)
            f_angle = f_angle_flat.reshape(shape)
            v_n = v_n_flat.reshape(shape)
            v_n = np.where(v_mag_flat.reshape(shape) > 1e-9, v_n, v_eff)

    # Thermal binder degradation at the metal-mould interface.
    if temperature is not None and temperature.shape == shape and sdf_a is not None:
        T_metal = np.asarray(temperature, dtype=np.float64).reshape(shape)
        delta_T = max(2.0 * dx_mm, 1.0)
        T_pour = float(getattr(alloy, "t_pour_c", 700.0))
        T_eff = T_metal + (T_pour - T_metal) * np.exp(-np.clip(sdf_a, 0.0, None) / delta_T)
        binder_deg = _binder_deg(T_eff)
    else:
        binder_deg = np.ones(shape, dtype=np.float64)

    # Darcy-Forchheimer wall shear.
    v_pore = v_eff / max(0.01, min(0.99, float(getattr(mold, "phi_mold", 0.35))))
    with np.errstate(divide="ignore", invalid="ignore"):
        dpdx_visc = mu * v_eff / K_eff
    phi = min(max(float(getattr(mold, "phi_mold", 0.35)), 0.01), 0.99)
    dpdx_inertial = (1.75 * rho_m * (1.0 - phi) / (phi ** 3 * d_p_m)) * (v_pore ** 2)
    tau_df = (dpdx_visc + dpdx_inertial) * d_p_m / 6.0
    tau_df = np.nan_to_num(tau_df, nan=0.0, posinf=0.0, neginf=0.0)

    D_shear = np.zeros(shape, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        D_shear = (tau_df / (tau_crit * binder_deg)) ** 1.5
    D_shear = np.nan_to_num(D_shear, nan=0.0, posinf=0.0, neginf=0.0)

    p_dyn = 0.5 * rho_m * v_n * v_n
    D_pen = (p_dyn / p_cap) ** 2.0
    D_pen = np.nan_to_num(D_pen, nan=0.0, posinf=0.0, neginf=0.0)

    D_vel = (v_eff / v_crit_eff) ** 2.0
    D_vel = np.nan_to_num(D_vel, nan=0.0, posinf=0.0, neginf=0.0)

    Re_K = rho_m * v_eff * np.sqrt(K_eff) / max(mu, 1e-12)
    D_ReK = (Re_K / 5.0) ** 1.5
    D_ReK = np.nan_to_num(D_ReK, nan=0.0, posinf=0.0, neginf=0.0)

    D_FB = (v_eff / v_crit_eff) ** 2.5 * f_angle
    D_FB = np.nan_to_num(D_FB, nan=0.0, posinf=0.0, neginf=0.0)

    # p=4 soft-max combination.
    combined = (D_shear ** 4 + D_vel ** 4 + D_pen ** 4 + D_ReK ** 4 + D_FB ** 4) ** 0.25
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)

    # Cumulative Miner-like normalisation by total fill time.
    exposure = np.ones(shape, dtype=np.float64)
    if fill_time is not None and fill_time.shape == shape:
        ft = np.asarray(fill_time, dtype=np.float64).reshape(shape)
        ft_safe = np.where(np.isfinite(ft), ft, 0.0)
        t_fill_total = float(np.max(ft_safe[metal])) if metal.any() else 1.0
        t_fill_total = max(t_fill_total, 1e-9)
        exposure = np.clip(1.0 - ft_safe / t_fill_total, 0.0, 1.0)
    else:
        t_fill_total = 1.0

    turb_factor = _turb_factor(Re_K)
    normalize = combined * wall_enhance * exposure
    field_risk = 1.0 - np.exp(-turb_factor * normalize)
    field_risk = np.where(metal, field_risk, 0.0)

    # Per-gating element contribution with boundary injection and centreline lerp.
    gate_risk = compute_gate_erosion_risk_v2(
        gating_nodes=gating_nodes,
        is_metal=is_metal,
        alloy=alloy,
        mold=mold,
        body_index=body_index,
        bodies=bodies,
        sdf=sdf,
        dx_mm=dx_mm,
        origin_mm=origin_mm,
        velocity_m_s=velocity_m_s,
        fill_time=fill_time,
        temperature=temperature,
        body_risk_out=body_risk_out,
        impingement_out=impingement_out,
    )

    risk = np.maximum(field_risk, gate_risk)
    risk = np.where(metal, risk, 0.0)
    risk = _mask_gaussian_blur(risk, metal, sigma=0.7)
    return np.clip(risk, 0.0, 1.0)
