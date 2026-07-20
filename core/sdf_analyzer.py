"""SDF-based geometric + pseudo-thermal casting analyzer - JoseCast v8.0."""

import math
from dataclasses import replace
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree
from scipy.special import erf
from skimage.feature import peak_local_max
from skimage.measure import marching_cubes
from skimage.morphology import skeletonize
from skimage.segmentation import watershed
from sklearn.cluster import DBSCAN

from core.materials import (
    Alloy,
    MoldMaterial,
    chvorinov_c_from_properties,
    get_alloy,
    get_mold,
)
from core.riser_designer import propose_risers
from core.thermal_solver import solve_3d_thermal
from core.voxelizer import build_part_grid
from core.types import (
    BODY_FEEDER_TYPES,
    BODY_METAL_TYPES,
    CHILL_BODY_TYPES,
    AnalysisResult,
    Body,
    BodyType,
    CastingParameters,
    HotSpot,
    RefinementRegion,
    RiserResult,
)


def _make_neighbors_26() -> Tuple[np.ndarray, np.ndarray]:
    """26-neighbor directions and voxel-center Euclidean costs."""
    neigh = []
    costs = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if di == dj == dk == 0:
                    continue
                d = np.sqrt(di * di + dj * dj + dk * dk)
                neigh.append((di, dj, dk))
                costs.append(d)
    return np.array(neigh, dtype=np.int32), np.array(costs, dtype=np.float64)


NEIGH_26, COST_26 = _make_neighbors_26()
NEIGH_6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def laplacian_smooth(
    field: np.ndarray, iterations: int = 3, sigma: float = 1.0
) -> np.ndarray:
    """Mild Gaussian smoothing (3 iteration = heavy physics feel)."""
    out = field.copy()
    for _ in range(iterations):
        out = ndimage.gaussian_filter(out, sigma=sigma, mode="nearest")
    return out


def compute_sdf(is_metal: np.ndarray, dx: float) -> np.ndarray:
    """Binary SDF: distance inside metal to nearest non-metal voxel."""
    return ndimage.distance_transform_edt(is_metal).astype(np.float64) * dx


def compute_subvoxel_sdf(
    is_metal: np.ndarray, dx: float, sub: int = 2
) -> np.ndarray:
    """
    Upsample the binary occupancy with linear interpolation, run EDT on the
    high-resolution grid and downsample to obtain a sub-voxel SDF.
    """
    if sub <= 1:
        return compute_sdf(is_metal, dx)
    zoom = float(sub)
    # Linear interpolation of 0/1 gives partial (0..1) boundary voxels.
    fine = ndimage.zoom(is_metal.astype(np.float64), zoom, order=1, mode="nearest")
    fine = (fine > 0.5).astype(np.uint8)
    fine_sdf = ndimage.distance_transform_edt(fine).astype(np.float64) * (dx / zoom)
    # Downsample by averaging (order=1) to keep smooth sub-voxel values.
    return ndimage.zoom(fine_sdf, 1.0 / zoom, order=1, mode="nearest")


def compute_curvature(sdf: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and Gaussian curvature from the SDF Hessian.
    Mean curvature is approximated by the trace of the Hessian (Laplacian of SDF);
    Gaussian curvature is the determinant of the Hessian.  Both are vectorised
    over the whole grid for speed.
    """
    gz, gy, gx = np.gradient(sdf, dx)
    hzz, hzy, hzx = np.gradient(gz, dx)
    hyz, hyy, hyx = np.gradient(gy, dx)
    hxz, hxy, hxx = np.gradient(gx, dx)

    # Mean curvature = Laplacian of SDF (trace of Hessian)
    mean_curv = hxx + hyy + hzz

    # Determinant of symmetric 3x3 Hessian
    gauss = (
        hxx * (hyy * hzz - hyz * hzy)
        - hxy * (hxy * hzz - hxz * hyz)
        + hxz * (hxy * hzy - hxz * hyy)
    )
    return mean_curv, gauss


def _marching_cubes_surface(
    bodies: List[Body], grid_shape: Tuple[int, int, int], origin: np.ndarray, dx: float
) -> Optional[trimesh.Trimesh]:
    """Create a watertight-ish combined metal surface for distance queries."""
    # Build a high-res label grid (1 = metal, 0 = empty) and run marching cubes.
    label = np.zeros(grid_shape, dtype=np.float64)
    for b in bodies:
        if b.body_type in (BodyType.EMPTY, BodyType.CORE):
            continue
        try:
            vox = trimesh.voxel.creation.voxelize(b.mesh, pitch=dx)
            if vox is None:
                continue
            mat = vox.fill().matrix
            off = (vox.transform[:3, 3] - origin) / dx
            off = np.round(off).astype(int)
            i0 = max(0, off[0])
            i1 = min(grid_shape[0], off[0] + mat.shape[0])
            j0 = max(0, off[1])
            j1 = min(grid_shape[1], off[1] + mat.shape[1])
            k0 = max(0, off[2])
            k1 = min(grid_shape[2], off[2] + mat.shape[2])
            li0 = i0 - off[0]
            lj0 = j0 - off[1]
            lk0 = k0 - off[2]
            region = mat[
                li0 : li0 + (i1 - i0), lj0 : lj0 + (j1 - j0), lk0 : lk0 + (k1 - k0)
            ]
            label[i0:i1, j0:j1, k0:k1][region.astype(bool)] = 1.0
        except Exception:
            continue
    if not label.any():
        return None
    try:
        verts, faces, *_ = marching_cubes(label, level=0.5)
        verts = verts * dx + origin
        if len(faces) == 0:
            return None
        return trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    except Exception:
        return None


def _scheil_fs(T_arr, t_liq, t_sol, k):
    """Vectorised Scheil solid fraction.

    fs = 1 - ((T - T_sol) / (T_liq - T_sol))^(1 / (1 - k))
    """
    fs = np.zeros_like(T_arr)
    mask_past = T_arr <= t_sol
    mask_liq = T_arr >= t_liq
    mask_mush = ~(mask_past | mask_liq)
    fs[mask_past] = 1.0
    fs[mask_liq] = 0.0
    if mask_mush.any():
        k = max(min(k, 0.999), 1e-6)
        ratio = (t_liq - T_arr[mask_mush]) / (t_liq - t_sol + 1e-9)
        ratio = np.clip(ratio, 0.0, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            fs[mask_mush] = 1.0 - np.power(1.0 - ratio, 1.0 / (1.0 - k))
            fs = np.clip(fs, 0.0, 1.0)
    return fs


def _temperature_from_erf(
    sdf: np.ndarray, t: Union[float, np.ndarray], alloy: Alloy, mold: MoldMaterial
) -> np.ndarray:
    """1-D semi-infinite solution of the Fourier heat equation in the normal direction."""
    alpha = mold.diffusivity_mm2_s
    if alpha <= 0:
        return np.full_like(sdf, alloy.t_pour_c)
    t = np.maximum(np.asarray(t, dtype=np.float64), 1e-9)
    arg = sdf / (2.0 * np.sqrt(alpha * t))
    T = mold.t0_c + (alloy.t_pour_c - mold.t0_c) * erf(arg)
    return np.clip(T, mold.t0_c, alloy.t_pour_c)


def _cooling_rate_from_erf(
    sdf: np.ndarray, t: Union[float, np.ndarray], alloy: Alloy, mold: MoldMaterial
) -> np.ndarray:
    """Time derivative dT/dt of the erf solution (always <= 0 for cooling)."""
    alpha = mold.diffusivity_mm2_s
    if alpha <= 0:
        return np.zeros_like(sdf)
    t = np.maximum(np.asarray(t, dtype=np.float64), 1e-9)
    sqrt_term = np.sqrt(alpha * t)
    arg = sdf / (2.0 * sqrt_term)
    exp = np.exp(-(arg * arg))
    denom = 2.0 * np.sqrt(np.pi * alpha) * (t ** 1.5)
    denom = np.where(denom > 0, denom, 1e-30)
    dTdt = - (alloy.t_pour_c - mold.t0_c) * sdf * exp / denom
    return np.where(sdf > 0, dTdt, 0.0)


def compute_thermal_field(
    grid: np.ndarray,
    is_metal: np.ndarray,
    alloy: Alloy,
    mold: MoldMaterial,
    dx: float,
    n_steps: int = 100,
    progress_callback: Optional[callable] = None,
    sdf: Optional[np.ndarray] = None,
    M_mod: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Analytical 1-D thermal field at the local Chvorinov solidification time:
        T(x,t_s) = T0 + (T_pour - T0) * erf( x / (2 sqrt(alpha*t_s)) )
    x is the signed distance to the nearest surface (SDF). Latent heat enters
    through the Scheil solid fraction. Returns (T, dT/dt, fs, div(∇T)).
    """
    if sdf is None:
        sdf = compute_subvoxel_sdf(is_metal, dx, sub=1)
    C = chvorinov_c_from_properties(alloy, mold)
    if M_mod is None:
        M_field = sdf
    else:
        M_field = M_mod
    t_s_field = np.maximum(compute_chvorinov_t(M_field, C), 1e-9)
    T = _temperature_from_erf(sdf, t_s_field, alloy, mold)
    cooling_rate = -_cooling_rate_from_erf(sdf, t_s_field, alloy, mold)
    solid_fraction = _scheil_fs(
        T, alloy.t_liquidus_c, alloy.t_solidus_c, alloy.partition_coefficient
    )
    thermal_divergence = ndimage.laplace(T) / (dx * dx)
    return T, cooling_rate, solid_fraction, thermal_divergence


def compute_chvorinov_t(M_field: np.ndarray, C: float) -> np.ndarray:
    """
    Chvorinov solidification time: t_s = C * M^2  [s].
    M is the local casting modulus (mm).
    """
    return C * np.maximum(M_field, 0.0) ** 2


def compute_niyama(
    sdf: np.ndarray,
    M_mod: np.ndarray,
    alloy: Alloy,
    mold: MoldMaterial,
    dx: float,
    is_metal: Optional[np.ndarray] = None,
    temperature: Optional[np.ndarray] = None,
    cooling_rate: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Physically-based Niyama criterion N = G / sqrt(R)  [K s^0.5 / mm].
    G is the metal-side temperature gradient required to remove latent +
    superheat, estimated from the Stefan velocity v = M_mod / t_s.  R is the
    Chvorinov cooling rate ΔT_solid / t_s.  The result is weighted by
    f = M_mod / sdf so that bulky, sphere-like regions (f < 1) report lower
    Niyama (higher shrinkage risk) than plates (f ≈ 1).
    """
    C = chvorinov_c_from_properties(alloy, mold)
    t_s = np.maximum(compute_chvorinov_t(M_mod, C), 1e-9)
    # Stefan velocity based on the local shape-corrected modulus [mm/s]
    v_solid = M_mod / t_s
    l_eff = alloy.latent_heat_j_kg + alloy.cp_j_kgk * max(
        alloy.t_pour_c - alloy.t_liquidus_c, 0.0
    )
    # Metal-side gradient required to carry away latent + superheat [K/mm]
    G = np.where(
        sdf > 0,
        alloy.rho_kg_m3 * l_eff * v_solid / (alloy.k_w_mk * 1e6),
        0.0,
    )
    # Cooling rate from Chvorinov [K/s]
    R = np.where(
        sdf > 0,
        (alloy.t_liquidus_c - alloy.t_solidus_c) / t_s,
        0.0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        niyama = G / np.sqrt(np.maximum(R, 1e-12))
    # Shape correction: sphere-like regions (f < 1) get lower Niyama
    shape_factor = M_mod / np.maximum(sdf, 1e-6)
    niyama = niyama * shape_factor
    niyama = np.nan_to_num(niyama, nan=0.0, posinf=0.0, neginf=0.0)
    if is_metal is not None:
        niyama = np.where(is_metal, niyama, 0.0)
        G = np.where(is_metal, G, 0.0)
        R = np.where(is_metal, R, 0.0)
    return G, R, niyama


def compute_niyama_variants(
    niyama: np.ndarray,
    G: np.ndarray,
    R: np.ndarray,
    t_s: np.ndarray,
    alloy: Alloy,
    max_time_s: float = 600.0,
) -> Dict[str, np.ndarray]:
    """
    Return physically meaningful Niyama indicators.

    - classical: G / sqrt(R)  [K sqrt(s) / mm]
    - macro_risk: max(0, 1 - N / N_macro)
    - shrinkage_risk: max(0, 1 - N / N_shrinkage)
    """
    niyama = np.nan_to_num(niyama, nan=0.0, posinf=0.0, neginf=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        macro_risk = np.clip(1.0 - niyama / max(alloy.niyama_macro, 1e-9), 0.0, 1.0)
        shrinkage_risk = np.clip(1.0 - niyama / max(alloy.niyama_shrinkage, 1e-9), 0.0, 1.0)
    return {
        "classical": niyama,
        "macro_risk": macro_risk,
        "shrinkage_risk": shrinkage_risk,
    }


def compute_niyama_ensemble(niyama: np.ndarray) -> np.ndarray:
    """Return the physical classical Niyama used for decisions."""
    return niyama


def compute_pore_size(
    niyama: np.ndarray,
    M_mod: np.ndarray,
    feed_risk: np.ndarray,
    alloy: Alloy,
    part_mask: np.ndarray,
    t_s: Optional[np.ndarray] = None,
    feeder_mask: Optional[np.ndarray] = None,
    dx: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate pore size (µm) from Niyama, local modulus and feeding deficit.

    Macro shrinkage pore:
        d_macro [mm] = t_section * shrinkage_factor * (1 - N/N_macro) * feed_risk
    Interdendritic micro pore:
        d_micro [µm] = SDAS_um * (1 - N/N_shrinkage) * feed_risk

    The shrinkage term is further reduced where the local solidification
    gradient points toward a feeder (directional feeding efficiency).  An
    unavoidable gas/oxide baseline is added so no region is artificially
    declared "zero porosity".

    Returns pore_size_um, pore_size_mm, macro_mask, micro_mask, fine_mask.
    """
    t_section_mm = 2.0 * np.maximum(M_mod, 0.0)

    raw_macro = np.clip(1.0 - niyama / max(alloy.niyama_macro, 1e-9), 0.0, 1.0)
    raw_micro = np.clip(1.0 - niyama / max(alloy.niyama_shrinkage, 1e-9), 0.0, 1.0)
    exp_n = max(alloy.pore_niyama_exponent, 1e-3)
    macro_factor = np.power(raw_macro, exp_n)
    micro_factor = np.power(raw_micro, exp_n)

    # Directional feeding: a feeder that is in the direction of the
    # solidification front is much more effective, but never removes all risk.
    _feeder = feeder_mask if feeder_mask is not None else np.zeros_like(part_mask)
    feed_eff = directional_feed_efficiency(t_s, _feeder, part_mask, dx)
    feed_factor = np.power(np.clip(feed_risk, 0.0, 1.0), alloy.feed_risk_exponent) * feed_eff

    d_macro_mm = t_section_mm * alloy.shrinkage_factor * macro_factor * feed_factor
    sdas_um = alloy.dendrite_spacing_mm * 1000.0
    d_micro_um = sdas_um * micro_factor * feed_factor

    # Real-life baseline: dissolved gas/oxides always leave some micro pores.
    # Thicker/slower-solidifying sections and lower-Niyama regions retain larger
    # gas pores, so the baseline is spatially modulated instead of uniform.
    baseline_min_um = max(alloy.gas_pore_baseline_um, sdas_um * 0.02)
    m_max = float(np.max(M_mod[part_mask])) if np.any(part_mask) else 1.0
    m_rel = np.clip(M_mod / max(m_max, 1e-9), 0.0, 1.0)
    # raw_micro already equals max(0, 1 - N/niyama_shrinkage)
    baseline_factor = (
        1.0
        + alloy.gas_pore_time_factor * m_rel
        + alloy.gas_pore_niyama_factor * raw_micro
    )
    baseline_um = baseline_min_um * np.clip(baseline_factor, 1.0, None)
    pore_size_um = np.where(
        part_mask,
        np.maximum(d_macro_mm * 1000.0 + d_micro_um, baseline_um),
        0.0,
    )
    pore_size_mm = pore_size_um / 1000.0

    pore_size_um = np.nan_to_num(pore_size_um, nan=0.0, posinf=0.0, neginf=0.0)
    pore_size_mm = np.nan_to_num(pore_size_mm, nan=0.0, posinf=0.0, neginf=0.0)

    macro_thr = alloy.macro_pore_limit_um
    micro_thr = alloy.micro_pore_limit_um
    macro_mask = (pore_size_um >= macro_thr) & part_mask
    micro_mask = (
        (pore_size_um >= micro_thr) & (pore_size_um < macro_thr) & part_mask
    )
    fine_mask = (pore_size_um > 0.0) & (pore_size_um < micro_thr) & part_mask

    return pore_size_um, pore_size_mm, macro_mask, micro_mask, fine_mask


def directional_feed_efficiency(
    t_s: np.ndarray,
    feeder_mask: np.ndarray,
    part_mask: np.ndarray,
    dx: float,
    min_eff: float = 0.05,
    max_reduction: float = 0.85,
) -> np.ndarray:
    """Return a per-voxel feeding-efficiency factor in [min_eff, 1.0].

    The local solidification time gradient points toward the last-freezing
    region.  A riser that lies in that direction can feed the voxel well, so
    shrinkage risk is reduced.  Where no gradient exists (uniform) the factor
    is neutral (0.5).  Far from any feeder the factor remains ~1.0.
    """
    if t_s is None or not feeder_mask.any():
        return np.ones_like(part_mask, dtype=np.float64)

    # Replace NaN/Inf in t_s so gradients are finite.
    t_safe = np.nan_to_num(t_s, nan=np.nanmax(t_s) if np.isfinite(t_s).any() else 0.0, posinf=0.0, neginf=0.0)

    gz, gy, gx = np.gradient(t_safe, dx)
    grad = np.stack([gz, gy, gx], axis=-1)
    grad_mag = np.linalg.norm(grad, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        grad_dir = np.where(grad_mag[..., None] > 1e-12, grad / grad_mag[..., None], 0.0)

    # Nearest feeder voxel for each voxel (feeder_mask True are features => pass inverted)
    _, nearest = ndimage.distance_transform_edt(~feeder_mask, return_indices=True)
    indices = np.indices(part_mask.shape)  # shape (3, *grid)
    diff = (nearest - indices) * dx  # vector from voxel to nearest feeder, shape (3, *grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        diff_norm = np.linalg.norm(diff, axis=0)
        diff_dir = np.where(diff_norm[None, ...] > 1e-9, diff / diff_norm[None, ...], 0.0)
    # grad_dir has channel last; bring diff_dir to same shape.
    diff_dir = np.moveaxis(diff_dir, 0, -1)

    # If the riser is in the direction of the solidification front (+grad t_s),
    # alignment is positive and feeding is more effective.
    alignment = np.einsum("...i,...i->...", diff_dir, grad_dir)
    alignment = np.where(grad_mag > 1e-12, alignment, 0.0)
    alignment = np.clip(alignment, 0.0, 1.0)

    # Strong alignment -> strong shrinkage reduction, but never zero (real life baseline).
    # Additionally, if the feeder solidifies before the part voxel it cannot feed it,
    # so the risk reduction is weakened.
    reduction = max_reduction * alignment
    t_feeder_max = 0.0
    if feeder_mask.any():
        feeder_times = t_safe[feeder_mask]
        finite_feeder_times = feeder_times[np.isfinite(feeder_times) & (feeder_times > 0.0)]
        if finite_feeder_times.size > 0:
            t_feeder_max = float(np.max(finite_feeder_times))
    if t_feeder_max > 0.0:
        time_factor = np.clip(t_feeder_max / np.maximum(t_safe, 1e-9), 0.0, 1.0)
        reduction = reduction * time_factor

    efficiency = 1.0 - reduction
    efficiency = np.clip(efficiency, min_eff, 1.0)
    return np.where(part_mask, efficiency, 1.0)


def _pore_size_class(
    pore_size_um: float,
    macro_threshold_um: float = 1000.0,
    micro_threshold_um: float = 100.0,
) -> str:
    if pore_size_um >= macro_threshold_um:
        return "macro"
    if pore_size_um >= micro_threshold_um:
        return "micro"
    if pore_size_um > 0.0:
        return "fine"
    return ""


def pore_size_threshold_um(
    pore_size_um: np.ndarray,
    noise_percent: float = 3.0,
    micro_pore_limit_um: float = 100.0,
    baseline_um: Optional[float] = None,
) -> float:
    """Return the display threshold for porosity.

    It is the larger of (a) the ``noise_percent`` percentile of positive pore
    sizes and (b) a material baseline floor.  The baseline floor is the material
    ``gas_pore_baseline_um`` (or the SDAS-derived 0.02*SDAS minimum) halved, so
    tiny gas pores are still represented without turning the whole part into a
    cloud.  If ``baseline_um`` is not provided, ``micro_pore_limit_um`` is used
    as a legacy fallback.
    """
    positive = np.asarray(pore_size_um[pore_size_um > 0.0], dtype=np.float64)
    if len(positive) == 0:
        return 0.0
    if noise_percent <= 0.0:
        return 0.0
    if noise_percent >= 100.0:
        return float(np.max(positive)) + 1.0
    p = 100.0 - noise_percent
    perc = float(np.percentile(positive, p))
    if baseline_um is not None and baseline_um > 0.0:
        floor = baseline_um * 0.5
    else:
        floor = micro_pore_limit_um * noise_percent / 100.0
    return max(perc, floor)


def _sdf_histogram(sdf: np.ndarray, mask: np.ndarray, bins: int = 50):
    """Histogram and dominant interior modulus."""
    vals = sdf[mask]
    if len(vals) == 0:
        return np.zeros(bins), np.linspace(0, 1, bins + 1), 0.0
    vmax = float(vals.max())
    hist, edges = np.histogram(vals, bins=bins, range=(0.0, vmax + 1e-6))
    interior = vals[vals > edges[1]]
    dominant_m = float(np.median(interior)) if len(interior) else float(np.median(vals))
    return hist, edges, dominant_m


def _histogram_stats(sdf: np.ndarray, mask: np.ndarray):
    vals = sdf[mask]
    if len(vals) == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    if std > 1e-9:
        skew = float(((vals - mean) ** 3).mean() / (std ** 3))
    else:
        skew = 0.0
    return mean, std, skew


def _local_section_thickness(
    sdf: np.ndarray,
    part_mask: np.ndarray,
    center_vox: np.ndarray,
    hotspot_m: float,
    dx: float,
) -> float:
    """Estimate local wall thickness (mm) around a hot spot via SDF median."""
    radius_vox = max(3.0 * hotspot_m / dx, 5.0)
    local_mask = _sphere_mask(part_mask.shape, center_vox, radius_vox) & part_mask
    vals = sdf[local_mask]
    if len(vals) == 0:
        return 2.0 * hotspot_m
    interior = vals[vals > dx * 2]
    m_local = float(np.median(interior)) if len(interior) else float(np.median(vals))
    return 2.0 * m_local


def _shape_factor(mask: np.ndarray, dx: float) -> float:
    """SF = V^2 / A^3; sphere gives ~0.0088, plates/rods give smaller."""
    volume = float(mask.sum()) * (dx ** 3)
    dilated = ndimage.binary_dilation(mask, iterations=1)
    surface = dilated & ~mask
    area = float(surface.sum()) * (dx ** 2)
    if area <= 0 or volume <= 0:
        return 0.0
    return (volume ** 2) / (area ** 3)


def find_hotspots(
    sdf: np.ndarray,
    part_mask: np.ndarray,
    dx: float,
    origin_mm: np.ndarray,
    curvature: Optional[np.ndarray] = None,
    use_skeleton: bool = True,
    min_size_mm: float = 2.0,
    cluster_eps_mm: float = 10.0,
    riser_mask: Optional[np.ndarray] = None,
    is_metal: Optional[np.ndarray] = None,
    feeder_mask: Optional[np.ndarray] = None,
    chvorinov_c: Optional[float] = None,
    n_time_steps: int = 40,
) -> List[HotSpot]:
    """Detect hot spots by pseudo-thermal solidification + CCL (Method 2).

    The metal is solidified in Chvorinov time layers.  At each layer, the
    remaining liquid metal is labelled with 26-connectivity.  Liquid pockets
    that are not connected to a feeder (riser / gating) are isolated; the last
    points to become isolated are the true hot spots.  A feeder/riser neck
    naturally solidifies earlier and breaks the connection, so the region under
    a riser is not reported as a part hot spot.
    """
    # Shape-corrected modulus
    if curvature is not None:
        # Concave regions (positive curvature) get a smaller shape factor,
        # increasing the local modulus to reflect heat accumulation at L/T/X
        # junctions.  f ≈ 0.77 gives up to ~30 % modulus boost.
        shape_factor_field = np.clip(
            1.0 - curvature * sdf, 0.77, 3.0
        )
    else:
        shape_factor_field = np.ones_like(sdf)
    M_mod = sdf / shape_factor_field

    if is_metal is None:
        is_metal = part_mask
    if feeder_mask is None:
        feeder_mask = riser_mask if riser_mask is not None else np.zeros_like(is_metal)
    if chvorinov_c is None or chvorinov_c <= 0:
        chvorinov_c = 1.0

    # Solidification time from shape-corrected modulus (Chvorinov)
    t_solid = chvorinov_c * M_mod * M_mod
    t_solid = np.nan_to_num(t_solid, nan=0.0, posinf=0.0, neginf=0.0)

    # Time horizon: use the part, fall back to all metal
    if part_mask.any():
        max_t = float(np.percentile(t_solid[part_mask], 99.9))
    else:
        max_t = float(np.percentile(t_solid[is_metal], 99.9))
    if max_t <= 0:
        return []

    # Quadratic time steps: denser near the end of solidification where pockets
    # shrink and disconnect.
    thresholds = max_t * (np.linspace(0.0, 1.0, n_time_steps + 1)[1:] ** 2)
    isolation_time = np.zeros_like(t_solid, dtype=np.float64)
    structure = np.ones((3, 3, 3), dtype=bool)

    for t in thresholds:
        liquid = is_metal & (t_solid > t)
        labeled, n = ndimage.label(liquid, structure=structure)
        if n == 0:
            continue
        # Labels that touch a feeder are considered fed, not isolated
        if feeder_mask.any():
            touch = np.unique(labeled[feeder_mask])
        else:
            touch = np.array([0], dtype=labeled.dtype)
        touch = set(int(x) for x in touch)
        isolated_labels = np.setdiff1d(np.arange(1, n + 1), list(touch), assume_unique=True)
        if isolated_labels.size == 0:
            continue
        isolated_mask = np.isin(labeled, isolated_labels)
        update = isolated_mask & (t > isolation_time)
        isolation_time[update] = t

    candidate_mask = part_mask & (isolation_time > 0.0)
    if not candidate_mask.any():
        return []

    # Each topologically distinct slow-solidifying pocket is segmented from the
    # isolation-time field.  Regional maxima separated by at least cluster_eps_mm
    # are used as watershed markers, so close local peaks in the same pocket are
    # merged while distinct pockets remain separate.  Gaussian smoothing breaks
    # the discrete threshold plateaus into natural peaks.
    min_vox = int(np.ceil((min_size_mm / max(dx, 0.01)) ** 3))
    size_vox = max(1, int(cluster_eps_mm / dx))
    sigma = max(0.8, size_vox / 3.0)
    iso_smooth = ndimage.gaussian_filter(isolation_time.astype(np.float64), sigma=sigma)
    regional_max = candidate_mask & (
        iso_smooth == ndimage.maximum_filter(iso_smooth, size=size_vox, mode="constant")
    )
    markers, n_markers = ndimage.label(regional_max, structure=structure)
    if n_markers == 0:
        # fallback: use the single highest-isolation voxel
        pos_vox = np.argwhere(isolation_time == isolation_time[candidate_mask].max())[0]
        m_value = float(M_mod[pos_vox[0], pos_vox[1], pos_vox[2]])
        return [
            HotSpot(
                position_mm=origin_mm + pos_vox * dx,
                m_value_mm=m_value,
                dist_to_riser_mm=np.inf,
                feed_ok=False,
                max_feeding_distance_mm=0.0,
            )
        ]

    # Watershed on the negative isolation time gives a basin for each pocket.
    labels = watershed(
        -iso_smooth,
        markers,
        mask=candidate_mask,
        connectivity=structure,
    )

    max_iso = float(isolation_time[candidate_mask].max())
    iso_threshold = 0.2 * max_iso
    hotspots: List[HotSpot] = []
    for lbl in range(1, n_markers + 1):
        mask = (labels == lbl) & candidate_mask
        voxel_count = int(mask.sum())
        if voxel_count < min_vox:
            continue
        comp_iso = isolation_time[mask]
        comp_max_iso = float(comp_iso.max())
        if comp_max_iso < iso_threshold:
            continue
        cand = np.argwhere(mask)
        vals = comp_iso
        m_vals = M_mod[cand[:, 0], cand[:, 1], cand[:, 2]]
        best_idx = int(np.argmax(vals * 1000.0 + m_vals))
        pos_vox = cand[best_idx]
        m_value = float(M_mod[pos_vox[0], pos_vox[1], pos_vox[2]])
        position_mm = origin_mm + pos_vox * dx
        hotspots.append(
            HotSpot(
                position_mm=position_mm,
                m_value_mm=m_value,
                dist_to_riser_mm=np.inf,
                feed_ok=False,
                max_feeding_distance_mm=0.0,
            )
        )

    # Final cleanup: merge hot spots that are close enough to be fed by one riser.
    if len(hotspots) > 1:
        positions = np.array([hs.position_mm for hs in hotspots], dtype=np.float64)
        clustering = DBSCAN(eps=cluster_eps_mm, min_samples=1, metric="euclidean").fit(
            positions
        )
        merged: List[HotSpot] = []
        for lbl in set(clustering.labels_):
            if lbl == -1:
                continue
            group = [hs for i, hs in enumerate(hotspots) if clustering.labels_[i] == lbl]
            group.sort(key=lambda h: h.m_value_mm, reverse=True)
            merged.append(group[0])
        hotspots = merged

    return hotspots


def feeding_distance_dijkstra(
    is_metal: np.ndarray,
    riser_mask: np.ndarray,
    dx: float,
    gravity_vector: Tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> np.ndarray:
    """Directed 26-neighbor weighted Dijkstra distance to the nearest riser.

    Upward steps (against gravity) are strongly penalised so that liquid metal
    cannot be fed uphill.  Returns distance in mm.
    """
    dist = np.full(is_metal.shape, np.inf, dtype=np.float64)
    if not (is_metal & riser_mask).any():
        return dist

    idx = np.full(is_metal.shape, -1, dtype=np.int64)
    metal_vox = np.argwhere(is_metal)
    n = int(metal_vox.shape[0])
    idx[tuple(metal_vox.T)] = np.arange(n)

    gx, gy, gz = gravity_vector
    norm = math.sqrt(gx * gx + gy * gy + gz * gz) + 1e-12
    gx, gy, gz = gx / norm, gy / norm, gz / norm
    UPWARD_PENALTY = 10.0
    DOWNWARD_BONUS = 0.7

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    vals: List[np.ndarray] = []

    for (di, dj, dk), c in zip(NEIGH_26, COST_26):
        ni = metal_vox[:, 0] + di
        nj = metal_vox[:, 1] + dj
        nk = metal_vox[:, 2] + dk
        mask = (
            (ni >= 0)
            & (ni < is_metal.shape[0])
            & (nj >= 0)
            & (nj < is_metal.shape[1])
            & (nk >= 0)
            & (nk < is_metal.shape[2])
        )
        if not mask.any():
            continue
        neighbor_idx = idx[ni[mask], nj[mask], nk[mask]]
        source_idx = np.arange(n)[mask]
        valid = neighbor_idx >= 0
        if not valid.any():
            continue
        dot = (di * gx + dj * gy + dk * gz) / math.sqrt(di * di + dj * dj + dk * dk)
        step = np.full(valid.sum(), c * dx, dtype=np.float64)
        if dot < 0:
            step *= (1.0 + UPWARD_PENALTY * (-dot))
        elif dot > 0:
            step *= max(0.5, 1.0 - 0.3 * dot)
        rows.append(source_idx[valid])
        cols.append(neighbor_idx[valid])
        vals.append(step.astype(np.float32))

    riser_flat = np.where(riser_mask[tuple(metal_vox.T)])[0]
    if len(riser_flat) == 0:
        return dist
    rows.append(np.full(len(riser_flat), n, dtype=np.int64))
    cols.append(riser_flat.astype(np.int64))
    vals.append(np.zeros(len(riser_flat), dtype=np.float32))

    graph = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n + 1, n + 1),
    ).tocsr()
    flat_dist = csgraph.dijkstra(graph, directed=True, indices=n, return_predecessors=False)
    dist[tuple(metal_vox.T)] = flat_dist[:n].astype(np.float64)
    return dist


def feeding_cost_dijkstra(
    is_metal: np.ndarray,
    riser_mask: np.ndarray,
    modulus: np.ndarray,
    dx: float,
    gravity_vector: Tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    26-neighbor directed Dijkstra where edge cost = dx / M (section modulus) of the
    voxel being entered.  Steps against the user-defined gravity vector are strongly
    penalised so liquid metal cannot be fed uphill.  Returns
    (cost_grid, predecessors, metal_vox).
    """
    cost = np.full(is_metal.shape, np.inf, dtype=np.float64)
    pred = np.full(is_metal.shape, -1, dtype=np.int64)
    if not (is_metal & riser_mask).any():
        return cost, pred, np.empty((0, 3), dtype=np.int64)

    idx = np.full(is_metal.shape, -1, dtype=np.int64)
    metal_vox = np.argwhere(is_metal)
    n = int(metal_vox.shape[0])
    idx[tuple(metal_vox.T)] = np.arange(n)

    gx, gy, gz = gravity_vector
    norm = math.sqrt(gx * gx + gy * gy + gz * gz) + 1e-12
    gx, gy, gz = gx / norm, gy / norm, gz / norm
    UPWARD_PENALTY = 10.0
    DOWNWARD_BONUS = 0.7

    rows, cols, vals = [], [], []
    for (di, dj, dk), c in zip(NEIGH_26, COST_26):
        ni = metal_vox[:, 0] + di
        nj = metal_vox[:, 1] + dj
        nk = metal_vox[:, 2] + dk
        mask = (
            (ni >= 0)
            & (ni < is_metal.shape[0])
            & (nj >= 0)
            & (nj < is_metal.shape[1])
            & (nk >= 0)
            & (nk < is_metal.shape[2])
        )
        if not mask.any():
            continue
        neighbor_idx = idx[ni[mask], nj[mask], nk[mask]]
        source_idx = np.arange(n)[mask]
        valid = neighbor_idx >= 0
        if not valid.any():
            continue
        # cost of moving into neighbor
        m_nb = np.clip(modulus[ni[mask][valid], nj[mask][valid], nk[mask][valid]], 0.1, None)
        step = (c * dx / m_nb).astype(np.float64)
        dot = (di * gx + dj * gy + dk * gz) / math.sqrt(di * di + dj * dj + dk * dk)
        if dot < 0:
            step *= (1.0 + UPWARD_PENALTY * (-dot))
        elif dot > 0:
            step *= max(0.5, 1.0 - 0.3 * dot)
        rows.append(source_idx[valid])
        cols.append(neighbor_idx[valid])
        vals.append(step.astype(np.float32))

    riser_flat = np.where(riser_mask[tuple(metal_vox.T)])[0]
    if len(riser_flat) == 0:
        return cost, pred, metal_vox
    rows.append(np.full(len(riser_flat), n, dtype=np.int64))
    cols.append(riser_flat.astype(np.int64))
    vals.append(np.zeros(len(riser_flat), dtype=np.float32))

    graph = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n + 1, n + 1),
    ).tocsr()
    flat_cost, flat_pred = csgraph.dijkstra(
        graph,
        directed=True,
        indices=n,
        return_predecessors=True,
    )
    flat_cost = flat_cost[:n].astype(np.float64)
    cost[tuple(metal_vox.T)] = flat_cost
    # flat_pred is 1D array of length n+1; map valid graph predecessors to flat voxel indices.
    fp = flat_pred[:n]
    pred_values = np.full(n, -1, dtype=np.int64)
    valid = (fp >= 0) & (fp < n)
    if valid.any():
        pv = metal_vox[fp[valid], 0] * is_metal.shape[1] * is_metal.shape[2]
        pv += metal_vox[fp[valid], 1] * is_metal.shape[2]
        pv += metal_vox[fp[valid], 2]
        pred_values[valid] = pv
    pred[tuple(metal_vox.T)] = pred_values
    return cost, pred, metal_vox


def _trace_cost_path(
    start_vox: np.ndarray,
    predecessors: np.ndarray,
    shape: Tuple[int, int, int],
) -> List[Tuple[int, int, int]]:
    """Walk predecessor map from a voxel back to the virtual riser source."""
    def _vox2flat(v):
        return int(v[0] * shape[1] * shape[2] + v[1] * shape[2] + v[2])

    path = []
    flat = _vox2flat(start_vox)
    visited = set()
    while True:
        if flat < 0 or flat in visited:
            break
        visited.add(flat)
        i = flat // (shape[1] * shape[2])
        rem = flat % (shape[1] * shape[2])
        j = rem // shape[2]
        k = rem % shape[2]
        path.append((i, j, k))
        pred = predecessors[i, j, k]
        if pred < 0 or pred == flat:
            break
        flat = pred
    return path


def _trace_path_to_riser(
    dist_to_riser: np.ndarray,
    part_mask: np.ndarray,
    start_vox: np.ndarray,
) -> List[Tuple[int, int, int]]:
    """Walk from start_vox toward decreasing distance-to-riser inside part."""
    shape = dist_to_riser.shape
    current = tuple(start_vox)
    if not part_mask[current]:
        return []
    path = [current]
    visited = {current}
    max_steps = shape[0] + shape[1] + shape[2]

    for _ in range(max_steps):
        i, j, k = current
        if dist_to_riser[i, j, k] <= 0:
            break
        best = None
        best_d = dist_to_riser[i, j, k]
        for di, dj, dk in NEIGH_6:
            ni, nj, nk = i + di, j + dj, k + dk
            if not (0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]):
                continue
            if not part_mask[ni, nj, nk]:
                continue
            d = dist_to_riser[ni, nj, nk]
            if d < best_d:
                best_d = d
                best = (ni, nj, nk)
        if best is None or best in visited:
            break
        visited.add(best)
        path.append(best)
        current = best
    return path


def trace_feeding_resistance(
    sdf: np.ndarray,
    dist_to_riser: np.ndarray,
    part_mask: np.ndarray,
    start_vox: np.ndarray,
) -> float:
    """Geometric feeding resistance along the path to the riser."""
    if not part_mask[start_vox[0], start_vox[1], start_vox[2]]:
        return 0.0
    m_hs = float(sdf[start_vox[0], start_vox[1], start_vox[2]])
    if m_hs <= 0:
        return 0.0
    path = _trace_path_to_riser(dist_to_riser, part_mask, start_vox)
    if not path:
        return 0.0
    resistance = 0.0
    for vox in path[1:]:
        m_i = float(sdf[vox[0], vox[1], vox[2]])
        if m_i > 0:
            resistance += max(0.0, (m_hs - m_i) / m_i)
    return resistance


def _liquid_fraction_at_time(
    sdf_i: float, t: float, alloy: Alloy, mold: MoldMaterial
) -> float:
    """Scheil liquid fraction at a point with signed distance sdf_i and time t."""
    if t <= 0:
        return 1.0
    T = float(
        _temperature_from_erf(
            np.array([sdf_i]), t, alloy, mold
        )[0]
    )
    if T >= alloy.t_liquidus_c:
        return 1.0
    if T <= alloy.t_solidus_c:
        return 0.0
    fs = _scheil_fs(
        np.array([T]),
        alloy.t_liquidus_c,
        alloy.t_solidus_c,
        alloy.partition_coefficient,
    )[0]
    return max(0.0, min(1.0, 1.0 - float(fs)))


def _path_darcy_and_directional(
    sdf: np.ndarray,
    M_mod: np.ndarray,
    cost_grid: np.ndarray,
    cost_pred: np.ndarray,
    part_mask: np.ndarray,
    start_vox: np.ndarray,
    dx: float,
    alloy: Alloy,
    mold: MoldMaterial,
    feeder_voxels: Optional[np.ndarray] = None,
    t_liq: Optional[np.ndarray] = None,
    t_sol: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, bool, bool, float, bool]:
    """
    Walk the lowest-resistance feeding path from start_vox to a riser and compute:
      * Darcy pressure drop through the mushy zone (Kozeny-Carman)
      * minimum neck modulus along the part path
      * t_s at the hot spot
      * directional solidification flag (using actual 3-D solidification times)
      * Heuver's circle flag
      * total feeding cost
      * darcy_ok flag
    """
    path = _trace_cost_path(start_vox, cost_pred, sdf.shape)
    if not path:
        return 0.0, 0.0, 0.0, True, True, 0.0, True

    m_hot = float(M_mod[start_vox[0], start_vox[1], start_vox[2]])
    # Use the actual 3-D transient solidification time if available
    if t_sol is not None and np.isfinite(t_sol[start_vox[0], start_vox[1], start_vox[2]]):
        t_s_hot = float(t_sol[start_vox[0], start_vox[1], start_vox[2]])
    else:
        C = chvorinov_c_from_properties(alloy, mold)
        t_s_hot = C * m_hot * m_hot

    # Hydrostatic head from feeders above the hot spot (if any)
    P_head = 0.0
    if feeder_voxels is not None and len(feeder_voxels) > 0:
        feeder_z = feeder_voxels[:, 2] * dx  # mm
        hot_z = start_vox[2] * dx
        dz_mm = np.max(feeder_z - hot_z)
        if dz_mm > 0:
            P_head = alloy.rho_kg_m3 * 9.81 * (dz_mm / 1000.0)

    # Feeding shrinkage demand: shrinkage of the last-liquid pocket at the hot spot
    hot_M = M_mod[start_vox[0], start_vox[1], start_vox[2]]
    V_hotspot_mm3 = (4.0 / 3.0) * np.pi * max(hot_M, 1e-3) ** 3
    V_shrink_mm3 = alloy.shrinkage_factor * V_hotspot_mm3
    Q_mm3_s = V_shrink_mm3 / max(t_s_hot, 1e-9)

    # Darcy pressure drop along the full metal path (part + runner + sprue)
    darcy = 0.0
    mu = max(alloy.viscosity_pa_s, 1e-6)
    d_dend = max(alloy.dendrite_spacing_mm, 0.01)
    f_l_stop = 0.10  # end of mass feeding / interdendritic flow
    feed_stopped = False
    has_thermal = t_liq is not None and t_sol is not None
    for vox in path[:-1]:
        M_i = max(float(M_mod[vox[0], vox[1], vox[2]]), 0.5)
        if has_thermal and np.isfinite(t_liq[vox[0], vox[1], vox[2]]):
            t_li = float(t_liq[vox[0], vox[1], vox[2]])
            t_si = float(t_sol[vox[0], vox[1], vox[2]])
            if t_s_hot <= t_li:
                f_l = 1.0
            elif t_s_hot >= t_si:
                f_l = 0.0
            else:
                f_l = max(0.0, min(1.0, (t_si - t_s_hot) / max(t_si - t_li, 1e-12)))
        else:
            sdf_i = max(float(sdf[vox[0], vox[1], vox[2]]), 0.5)
            f_l = _liquid_fraction_at_time(sdf_i, t_s_hot, alloy, mold)
        if f_l <= f_l_stop:
            feed_stopped = True
            break
        # Kozeny-Carman permeability in the mushy zone [mm²]
        fl_c = max(min(f_l, 0.97), 0.05)
        K_mm2 = (d_dend ** 2 / 180.0) * (fl_c ** 3) / ((1.0 - fl_c) ** 2)
        # Cross-section approximated as a disk of the local modulus
        A_mm2 = np.pi * M_i * M_i
        v_mms = Q_mm3_s / max(A_mm2, 1e-6)
        v_ms = v_mms / 1000.0
        darcy += mu * v_ms * (dx / 1000.0) / max(K_mm2 / 1e6, 1e-15)

    # Minimum driving head: 1000 Pa ≈ 0.01 atm / ~13 mm metal head
    darcy_ok = (not feed_stopped) and (darcy < max(P_head, 1000.0))

    # Heuver / directional checks on the PART portion of the path
    part_path = [v for v in path if part_mask[v[0], v[1], v[2]]]
    if len(part_path) < 2:
        part_path = path

    m_part = np.array([M_mod[v[0], v[1], v[2]] for v in part_path])
    min_neck_m = float(m_part.min()) if len(m_part) else m_hot

    # Heuver (vector): modulus gradient projected on the feeding direction must be
    # non-negative.  A negative dot product means the modulus is decreasing toward
    # the feeder (geometric choking).
    heuvers_ok = True
    if len(part_path) > 4:
        grad_M = np.stack(np.gradient(M_mod), axis=0)  # mm / voxel
        tol = max(0.05 * m_hot, 0.1)
        for i in range(1, len(part_path)):
            a = np.array(part_path[i - 1], dtype=np.float64)
            b = np.array(part_path[i], dtype=np.float64)
            step = b - a
            step_len = float(np.linalg.norm(step)) + 1e-12
            u = step / step_len
            v = part_path[i - 1]
            g = np.array(
                [
                    float(grad_M[0][v[0], v[1], v[2]]),
                    float(grad_M[1][v[0], v[1], v[2]]),
                    float(grad_M[2][v[0], v[1], v[2]]),
                ]
            )
            if float(np.dot(g, u)) < -tol:
                heuvers_ok = False
                break

    # Directional solidification: solidification time must increase (or stay)
    # toward the feeder.  A drop means a cold pocket blocking feeding.
    directional_ok = True
    if has_thermal and len(part_path) > 4:
        t_path = np.array([float(t_sol[v[0], v[1], v[2]]) for v in part_path])
        tol = max(0.05 * t_s_hot, 1.0)
        if np.any(np.diff(t_path[1:]) < -tol):
            directional_ok = False

    feeding_cost = float(cost_grid[start_vox[0], start_vox[1], start_vox[2]])
    return darcy, min_neck_m, t_s_hot, directional_ok, heuvers_ok, feeding_cost, darcy_ok


def _sphere_mask(
    shape: Tuple[int, int, int], center: np.ndarray, radius_vox: float
) -> np.ndarray:
    """Boolean spherical mask inside the grid."""
    z, y, x = np.indices(shape, dtype=np.float64)
    z -= center[0]
    y -= center[1]
    x -= center[2]
    return z * z + y * y + x * x <= (radius_vox * radius_vox)


def _ingate_contact_m(
    grid: np.ndarray, sdf: np.ndarray, part_mask: np.ndarray, dx: float
) -> float:
    """Average SDF (modulus) of part voxels touching an ingate."""
    ingate = grid == BodyType.INGATE
    if not ingate.any():
        return 0.0
    touch = np.zeros_like(part_mask)
    for di, dj, dk in NEIGH_6:
        rolled = np.roll(ingate, (di, dj, dk), axis=(0, 1, 2))
        if di > 0:
            rolled[-1, :, :] = False
        elif di < 0:
            rolled[0, :, :] = False
        if dj > 0:
            rolled[:, -1, :] = False
        elif dj < 0:
            rolled[:, 0, :] = False
        if dk > 0:
            rolled[:, :, -1] = False
        elif dk < 0:
            rolled[:, :, 0] = False
        touch |= rolled & part_mask
    vals = sdf[touch]
    if len(vals) == 0:
        return 0.0
    return float(vals.mean())


def _refine_region(
    bodies: List[Body],
    hotspot: HotSpot,
    hotspot_index: int,
    coarse_origin: np.ndarray,
    coarse_dx: float,
    alloy: Alloy,
    mold: MoldMaterial,
    base_res: int,
    max_res: int,
    progress_callback: Optional[callable] = None,
) -> Optional[RefinementRegion]:
    """Create a high-resolution local grid around a hot spot."""
    from core.voxelizer import build_voxel_grid

    m = hotspot.m_value_mm
    half = max(4 * m, 3 * coarse_dx)
    local_min = hotspot.position_mm - half
    local_max = hotspot.position_mm + half
    local_size = local_max - local_min
    max_size = float(local_size.max())

    desired_dx = max_size / max_res
    # v8.2: dense local refine above ~96³ is too heavy for normal RAM budgets;
    # keep the 2040 dial for future sparse octree implementations.
    max_local_dim = min(max_res, 96)
    dx_fine = max(desired_dx, max_size / max_local_dim)

    cropped_bodies: List[Body] = []
    for b in bodies:
        bmin = b.mesh.bounds[0]
        bmax = b.mesh.bounds[1]
        if np.any(bmax < local_min - 1.0) or np.any(bmin > local_max + 1.0):
            continue
        box = trimesh.creation.box(
            extents=local_size,
            transform=trimesh.transformations.translation_matrix(hotspot.position_mm),
        )
        try:
            cropped = (
                b.mesh.intersection(box, engine="scad")
                if hasattr(b.mesh, "intersection")
                else b.mesh
            )
        except Exception:
            cropped = b.mesh
        if cropped is None or len(cropped.faces) == 0:
            continue
        cb = Body(
            index=b.index,
            name=b.name,
            vertices=cropped.vertices,
            faces=cropped.faces,
            mesh=cropped,
            body_type=b.body_type,
            volume_cm3=cropped.volume / 1000.0,
            center=cropped.center_mass if cropped.is_watertight else cropped.centroid,
        )
        cropped_bodies.append(cb)

    if not cropped_bodies:
        return None

    target_dim = max(32, int(max_size / dx_fine))
    grid, origin, dx, _ = build_voxel_grid(
        cropped_bodies,
        target_dim=target_dim,
        progress_callback=progress_callback,
    )
    is_metal = np.isin(grid, BODY_METAL_TYPES)
    sdf = compute_sdf(is_metal, dx)
    C = chvorinov_c_from_properties(alloy, mold)
    mean_curv, _ = compute_curvature(sdf, dx)
    shape_factor_field = np.clip(
        1.0 - mean_curv * sdf, 0.77, 3.0
    )
    M_mod = sdf / shape_factor_field
    t_s = compute_chvorinov_t(M_mod, C)
    T, R, fs, _ = compute_thermal_field(
        grid, is_metal, alloy, mold, dx, sdf=sdf, M_mod=M_mod
    )
    G, R, niyama = compute_niyama(
        sdf, M_mod, alloy, mold, dx, is_metal=is_metal,
        temperature=T, cooling_rate=R,
    )
    niyama = np.where(grid == BodyType.PART, niyama, 0.0)

    niyama_risk = np.clip(1.0 - niyama / alloy.niyama_shrinkage, 0.0, 1.0)
    FD_field = alloy.feed_k1 * (2.0 * M_mod)
    # Smooth local feeding risk: 0 when local thickness << FD, 0.5 at equality, -> 1 when much thicker than FD
    feed_risk = sdf / (sdf + np.maximum(FD_field, 1.0))
    feed_risk = np.clip(feed_risk, 0.0, 1.0)
    risk = 1.0 - (1.0 - niyama_risk) * (1.0 - feed_risk)
    part_mask_local = grid == BodyType.PART
    risk = np.where(part_mask_local, risk, 0.0)
    risk = np.nan_to_num(risk, nan=0.0, posinf=0.0, neginf=0.0)

    return RefinementRegion(
        hotspot_index=hotspot_index,
        origin_mm=origin,
        dx_mm=dx,
        grid=grid,
        sdf=sdf,
        niyama=niyama,
        risk=risk,
    )


def _high_res_part_hotspots(
    bodies: List[Body],
    feeder_mask: np.ndarray,
    origin_mm: np.ndarray,
    coarse_dx: float,
    part_voxels_target: int,
    part_max_dim: int,
    chvorinov_c: float,
    progress_callback: Optional[callable] = None,
) -> Optional[List[HotSpot]]:
    """Build a high-resolution grid containing the PART and connected casting-metal
    bodies (gating/riser) and detect hot spots.

    Including gating/riser geometry in the local high-resolution grid lets the
    pseudo-thermal CCL see the real metal connectivity, so gate/riser-fed part
    pockets are not reported as isolated hot spots.
    """
    try:
        part_grid, part_origin, part_dx, _ = build_part_grid(
            bodies,
            target_voxels=part_voxels_target,
            max_dim=part_max_dim,
        )
    except Exception:
        return None

    if part_grid is None or part_grid.size == 0:
        return None

    part_mask = part_grid == BodyType.PART
    part_is_metal = np.isin(part_grid, [int(t) for t in BODY_METAL_TYPES])
    if not part_mask.any() or part_is_metal.sum() < 1000:
        return None

    if progress_callback:
        progress_callback(83)

    # SDF and curvature on the casting-metal high-res grid (sub=1 to avoid 8x blowup).
    # Using the union of part + gating/riser for the SDF means a thin part region
    # that is directly attached to a thick runner/riser gets a larger effective
    # modulus and stays connected longer during the pseudo-thermal CCL.
    part_sdf = compute_subvoxel_sdf(part_is_metal, part_dx, sub=1)
    mean_curv, _ = compute_curvature(part_sdf, part_dx)
    shape_factor_field = np.clip(
        1.0 - mean_curv * part_sdf, 0.77, 3.0
    )
    part_M_mod = part_sdf / shape_factor_field

    # Derive feeder mask directly from the high-res grid.  Only dedicated
    # RISER bodies are true feeders; gates/runners/sprues are not.
    part_feeder_mask = part_grid == BodyType.RISER

    # Merge with the resampled coarse feeder mask as a safety net.
    if feeder_mask is not None and feeder_mask.size:
        idx = np.indices(part_grid.shape, dtype=np.float64)
        coarse_coords = (
            part_origin[:, None, None, None]
            + idx * part_dx
            - origin_mm[:, None, None, None]
        ) / coarse_dx
        coarse_resampled = (
            ndimage.map_coordinates(
                feeder_mask.astype(np.float32),
                coarse_coords,
                order=0,
                mode="constant",
                cval=0.0,
            )
            > 0.5
        )
        part_feeder_mask = part_feeder_mask | coarse_resampled

    # Guard against a completely missing feeder: in that case the CCL cannot mark
    # anything as fed and every liquid pocket becomes isolated, which is fine.
    max_part_sdf = float(part_sdf[part_mask].max()) if part_mask.any() else 0.0
    hotspot_min_size_mm = min(2.0, max(0.5, 0.5 * max_part_sdf))
    hotspot_cluster_mm = max(12.0, 2.0 * part_dx)

    if progress_callback:
        progress_callback(83)

    part_hotspots = find_hotspots(
        part_sdf,
        part_mask,
        part_dx,
        part_origin,
        curvature=mean_curv,
        use_skeleton=True,
        min_size_mm=hotspot_min_size_mm,
        cluster_eps_mm=hotspot_cluster_mm,
        is_metal=part_is_metal,
        feeder_mask=part_feeder_mask,
        chvorinov_c=chvorinov_c,
    )
    return part_hotspots


def _feeder_time_factor(body: Body) -> float:
    """Return the solidification-time multiplier for a RISER body.

    Values > 1 keep the feeder liquid longer; values < 1 solidify it faster.
    """
    ftype = (body.feeder_type or "conventional").lower().strip()
    if not ftype:
        ftype = "conventional"
    return {
        "conventional": 1.0,
        "exothermic": 1.5,
        "insulated": 1.2,
        "sleeve": 1.1,
        "chilled": 0.8,
    }.get(ftype, 1.0)


def _apply_feeder_sleeve_time_factor(
    grid: np.ndarray,
    t_liq: np.ndarray,
    t_sol: np.ndarray,
    bodies: List[Body],
) -> None:
    """Scale solidification times of RISER voxels by the weighted feeder type."""
    riser_bodies = [b for b in bodies if b.body_type == BodyType.RISER]
    if not riser_bodies:
        return
    total_volume = sum(max(b.volume_cm3, 1e-9) for b in riser_bodies)
    if total_volume <= 0.0:
        return
    factor = sum(
        _feeder_time_factor(b) * max(b.volume_cm3, 1e-9) / total_volume
        for b in riser_bodies
    )
    riser_mask = grid == BodyType.RISER
    if not riser_mask.any():
        return
    finite_liq = riser_mask & np.isfinite(t_liq)
    finite_sol = riser_mask & np.isfinite(t_sol)
    t_liq[finite_liq] = t_liq[finite_liq] * factor
    t_sol[finite_sol] = t_sol[finite_sol] * factor


def analyze(
    bodies: List[Body],
    grid: np.ndarray,
    origin_mm: np.ndarray,
    dx: float,
    alloy_key: str = "42CrMo4",
    mold_key: str = "sand",
    base_res: int = 160,
    max_res: int = 2040,
    refine_local: bool = True,
    sub_voxel: int = 2,
    thermal_max_time_s: float = 600.0,
    thermal_downsample: int = 2,
    casting_params: Optional[CastingParameters] = None,
    progress_callback: Optional[callable] = None,
    part_voxels_target: int = 10_000_000,
    part_max_dim: int = 600,
) -> AnalysisResult:
    """Run the full v8.0+ geometric + 3-D transient thermal analysis pipeline."""
    import time

    t_start = time.time()
    if progress_callback:
        progress_callback(5)

    alloy = get_alloy(alloy_key)
    mold = get_mold(mold_key)
    if casting_params is not None:
        # User overrides for this run
        alloy = replace(
            alloy,
            t_pour_c=casting_params.t_pour_c,
            t_liquidus_c=casting_params.t_liquidus_c,
            t_solidus_c=casting_params.t_solidus_c,
            rho_kg_m3=casting_params.rho_liquid_kg_m3,
            viscosity_pa_s=casting_params.viscosity_pa_s,
        )
        mold = replace(mold, t0_c=casting_params.t_mold_c)
    chvorinov_c = chvorinov_c_from_properties(alloy, mold)
    gravity_vector = (
        casting_params.gravity_vector
        if casting_params is not None
        else (0.0, 0.0, -1.0)
    )
    bbox_size = np.array(grid.shape) * dx

    is_metal = np.isin(grid, BODY_METAL_TYPES)
    if is_metal.sum() < 1000:
        raise ValueError(
            "Model çok küçük. Çözünürlüğü artırın veya modelin mm biriminde olduğundan emin olun."
        )

    # v8.2 memory guard: very large grids at sub>1 explode RAM; fall back to sub=1.
    if grid.size > 2_000_000 and sub_voxel > 1:
        sub_voxel = 1

    part_mask = grid == BodyType.PART
    riser_mask = grid == BodyType.RISER

    # v8.6: exposed part surface area (mold contact) and volume for modulus/riser calculations.
    part_pad = np.pad(part_mask, 1, constant_values=False)
    metal_pad = np.pad(is_metal, 1, constant_values=False)
    exposed_faces = np.zeros_like(part_pad, dtype=int)
    for di, dj, dk in NEIGH_6:
        exposed_faces += part_pad & ~np.roll(metal_pad, (di, dj, dk), axis=(0, 1, 2))
    voxel_surface_area_mm2 = float(exposed_faces[1:-1, 1:-1, 1:-1].sum()) * dx * dx
    mesh_surface_area_mm2 = sum(
        b.surface_area_cm2 for b in bodies if b.body_type == BodyType.PART
    ) * 100.0
    part_surface_area_mm2 = (
        mesh_surface_area_mm2 if mesh_surface_area_mm2 > 0.0 else voxel_surface_area_mm2
    )
    part_volume_mm3 = float(part_mask.sum()) * dx ** 3

    # Only dedicated risers are true feeding sources during solidification.
    # Gating (sprue/runner/ingate) supplies metal during filling but is not a
    # feeder; if no riser exists, the part is effectively un-fed.
    if riser_mask.any():
        feeder_mask = riser_mask
        no_riser = False
    else:
        feeder_mask = np.zeros_like(riser_mask)
        no_riser = True

    # AŞAMA 2: SDF (sub-voxel) + histogram + curvature + shape factor
    sdf = compute_subvoxel_sdf(is_metal, dx, sub=sub_voxel)
    if progress_callback:
        progress_callback(18)

    mean_curv, gauss_curv = compute_curvature(sdf, dx)
    # Shape factor from mean curvature: f=1 for plates, f≈2 for cylinders,
    # f≈3 for spheres, f<1 for concave L/T/X junctions (heat accumulation).
    shape_factor_field = np.clip(
        1.0 - mean_curv * sdf, 0.77, 3.0
    )
    M_mod = sdf / shape_factor_field
    if progress_callback:
        progress_callback(25)

    _, _, dominant_m = _sdf_histogram(M_mod, part_mask, bins=50)
    wall_thickness = 2.0 * dominant_m if dominant_m > 0 else 0.0
    m_mean, m_std, m_skew = _histogram_stats(M_mod, part_mask)
    if part_surface_area_mm2 > 0.0 and part_volume_mm3 > 0.0:
        shape_factor_global = (part_volume_mm3 ** 2) / (part_surface_area_mm2 ** 3)
    else:
        shape_factor_global = _shape_factor(part_mask, dx)
    if progress_callback:
        progress_callback(28)

    # AŞAMA 3: Full 3-D transient enthalpy thermal solver (downsampled for speed)
    if progress_callback:
        progress_callback(30)
        progress_callback(31)
    temperature, solid_fraction, t_liq, t_s, G, cooling_rate, niyama = solve_3d_thermal(
        grid, alloy, mold, dx,
        max_time_s=thermal_max_time_s,
        downsample=thermal_downsample,
        progress_callback=progress_callback,
    )
    # v9.3: account for feeder sleeves/exothermic/chilled type by scaling the
    # solidification time of RISER voxels.  A single weighted factor per model is
    # used because the voxel grid stores body-type IDs; per-body factors require
    # a body-index grid.  This is a robust first-order correction.
    _apply_feeder_sleeve_time_factor(grid, t_liq, t_s, bodies)

    # Fallback for thick regions that did not reach solidus within max_time_s:
    # use the analytical Chvorinov/Stefan Niyama so hot spots are not reported as 0.
    G_ana, R_ana, niyama_ana = compute_niyama(
        sdf, M_mod, alloy, mold, dx, is_metal=is_metal
    )
    solidified = np.isfinite(t_s) & (t_s > 0.0) & (niyama > 0.0)
    niyama = np.where(solidified, niyama, niyama_ana)
    G = np.where(solidified, G, G_ana)
    cooling_rate = np.where(solidified, cooling_rate, R_ana)

    thermal_divergence = ndimage.laplace(temperature) / (dx * dx)
    if progress_callback:
        progress_callback(60)

    # AŞAMA 4: Niyama family from the 3-D thermal solution
    niyama_variants = compute_niyama_variants(niyama, G, cooling_rate, t_s, alloy, max_time_s=thermal_max_time_s)
    niyama = compute_niyama_ensemble(niyama)
    # v8.5: porosity / Niyama display should be restricted to the part,
    # not to risers/gating, to avoid meaningless artifacts.
    for k in list(niyama_variants.keys()):
        niyama_variants[k] = np.where(part_mask, niyama_variants[k], 0.0)
    niyama = np.where(part_mask, niyama, 0.0)
    if progress_callback:
        progress_callback(65)

    # AŞAMA 5: Hot spot detection (medial axis + DBSCAN + curvature)
    max_part_sdf = float(sdf[part_mask].max()) if part_mask.any() else 0.0
    hotspot_min_size_mm = min(2.0, max(0.5, 0.5 * max_part_sdf))
    hotspot_cluster_mm = max(12.0, 2.0 * dx)
    hotspots = find_hotspots(
        sdf, part_mask, dx, origin_mm, curvature=mean_curv, use_skeleton=True,
        min_size_mm=hotspot_min_size_mm,
        cluster_eps_mm=hotspot_cluster_mm,
        is_metal=is_metal,
        feeder_mask=feeder_mask,
        chvorinov_c=chvorinov_c,
    )
    if progress_callback:
        progress_callback(75)

    # AŞAMA 6: 26-neighbor Dijkstra feeding distance and lowest-resistance cost path
    dist_feed = feeding_distance_dijkstra(is_metal, feeder_mask, dx, gravity_vector=gravity_vector)
    cost_feed, cost_pred, _ = feeding_cost_dijkstra(
        is_metal, feeder_mask, M_mod, dx, gravity_vector=gravity_vector
    )
    if progress_callback:
        progress_callback(82)

    # AŞAMA 6.5: High-resolution PART-only grid (hybrid voxelization).
    # The gating/riser geometry is already handled by CAD cross-sections and the
    # coarse global grid is only used for connectivity/feeding distance; the
    # critical part geometry is resolved at ~10 M voxels for accurate hot spots.
    if part_voxels_target > 0:
        part_hotspots = _high_res_part_hotspots(
            bodies,
            feeder_mask,
            origin_mm,
            dx,
            part_voxels_target,
            part_max_dim,
            chvorinov_c,
            progress_callback,
        )
        if part_hotspots is not None and part_hotspots:
            hotspots = part_hotspots

    if progress_callback:
        progress_callback(84)

    # Nearest-part-voxel lookup: high-resolution hotspots may map to a coarse
    # voxel that is just outside the part (boundary discretisation).  Snap to
    # the closest metal voxel so Niyama / feeding values are not lost.
    _, nearest_part_vox = ndimage.distance_transform_edt(part_mask, return_indices=True)

    def _snap_to_part(voxel):
        v = np.clip(voxel, 0, np.array(grid.shape) - 1).astype(int)
        if part_mask[v[0], v[1], v[2]]:
            return (int(v[0]), int(v[1]), int(v[2]))
        return tuple(int(x) for x in nearest_part_vox[:, v[0], v[1], v[2]])

    # v8.7: part voxels within the feeder's feeding distance are fed by that feeder
    # and should not be reported as part hot spots (e.g., directly under a riser).
    # Feeding distance FD = feed_k1 * t_section = feed_k1 * 2 * M_feeder.
    if feeder_mask.any():
        max_feeder_m = float(M_mod[feeder_mask & (M_mod > 0)].max()) if (feeder_mask & (M_mod > 0)).any() else 0.0
        feeder_fd_mm = alloy.feed_k1 * 2.0 * max_feeder_m
        influence_mm = max(2.0 * dx, feeder_fd_mm, 2.0)
        influence_vox = int(np.ceil(influence_mm / dx))
        dilated_feeder = ndimage.binary_dilation(feeder_mask, iterations=influence_vox)
        fed_zone = dilated_feeder & part_mask
        filtered_hotspots: List[HotSpot] = []
        for hs in hotspots:
            vox_raw = np.round((hs.position_mm - origin_mm) / dx).astype(int)
            vox = _snap_to_part(vox_raw)
            if not fed_zone[vox[0], vox[1], vox[2]]:
                filtered_hotspots.append(hs)
        hotspots = filtered_hotspots

    # AŞAMA 7: Hot-spot physics
    feeder_voxels = np.argwhere(feeder_mask)

    for hs in hotspots:
        vox_raw = np.round((hs.position_mm - origin_mm) / dx).astype(int)
        vox = _snap_to_part(vox_raw)
        if 0 <= vox[0] < grid.shape[0] and 0 <= vox[1] < grid.shape[1] and 0 <= vox[2] < grid.shape[2]:
            hs.dist_to_riser_mm = float(dist_feed[vox[0], vox[1], vox[2]])
            hs.niyama_min = float(niyama[vox[0], vox[1], vox[2]])
            hs.niyama_variants = {
                k: float(v[vox[0], vox[1], vox[2]]) for k, v in niyama_variants.items()
            }
            hs.niyama_ensemble = float(niyama[vox[0], vox[1], vox[2]])
            hs.local_sdf_max = float(sdf[vox[0], vox[1], vox[2]])
            hs.m_uncertainty_mm = dx / 2.0
            hs.feeding_cost = float(cost_feed[vox[0], vox[1], vox[2]])

            darcy, min_neck_m, t_hs, directional_ok, heuvers_ok, feeding_cost, darcy_ok = _path_darcy_and_directional(
                sdf,
                M_mod,
                cost_feed,
                cost_pred,
                part_mask,
                vox,
                dx,
                alloy,
                mold,
                feeder_voxels=feeder_voxels,
                t_liq=t_liq,
                t_sol=t_s,
            )
            hs.darcy_resistance = darcy
            hs.min_neck_m_mm = min_neck_m
            hs.directional_ok = directional_ok
            hs.heuvers_ok = heuvers_ok
            hs.darcy_ok = darcy_ok
            hs.curvature_mean = float(mean_curv[vox[0], vox[1], vox[2]])
            hs.curvature_gaussian = float(gauss_curv[vox[0], vox[1], vox[2]])

            # Section thickness = 2 * local modulus (equivalent wall thickness)
            hs.t_section_mm = 2.0 * hs.m_value_mm
            hs.width_mm = 2.0 * hs.local_sdf_max
            hs.shape_factor = _shape_factor(
                _sphere_mask(part_mask.shape, vox, max(3.0 * hs.m_value_mm / dx, 5.0)) & part_mask,
                dx,
            )

            if len(feeder_voxels) > 0:
                feeder_positions_mm = feeder_voxels * dx + origin_mm
                dz = feeder_positions_mm[:, 2] - hs.position_mm[2]
                closest_feeder_idx = int(
                    np.argmin(np.linalg.norm(feeder_positions_mm - hs.position_mm, axis=1))
                )
                dz_closest = dz[closest_feeder_idx]
                hs.gravity_factor = 1.0 + 0.3 * max(
                    0.0, dz_closest / max(hs.dist_to_riser_mm, 1.0)
                )
            else:
                dz_closest = 0.0
                hs.gravity_factor = 1.0

            # FD = feed_k1 * t_section (t_section = 2 * M_mod)
            base_fd = alloy.feed_k1 * hs.t_section_mm
            hs.max_feeding_distance_mm = base_fd * hs.gravity_factor
            feed_cost_ok = hs.feeding_cost < 30.0
            hs.feed_ok = (
                (not np.isinf(hs.dist_to_riser_mm))
                and (hs.dist_to_riser_mm <= hs.max_feeding_distance_mm)
                and hs.directional_ok
                and hs.heuvers_ok
                and feed_cost_ok
                and hs.darcy_ok
            )
        else:
            hs.feed_ok = False

    if progress_callback:
        progress_callback(88)

    # AŞAMA 8: Riser sufficiency with resistance-corrected modulus transfer
    riser_results: List[RiserResult] = []
    labeled, num = ndimage.label(riser_mask)
    # v9.1: per-voxel riser-size factor for effective feeding distance.
    # ID 0 means "no riser" and keeps the default factor 1.0.
    riser_factor_map = np.zeros(grid.shape, dtype=np.int32)
    factor_by_id = [1.0]
    g_unit = np.asarray(gravity_vector, dtype=np.float64)
    g_norm = float(np.linalg.norm(g_unit))
    if g_norm > 1e-12:
        g_unit = g_unit / g_norm
    else:
        g_unit = np.array([0.0, 0.0, -1.0])
    for body in bodies:
        if body.body_type != BodyType.RISER:
            continue
        body_center_vox = np.round((body.center - origin_mm) / dx).astype(int)
        best_label = 1
        best_dist = np.inf
        for lbl in range(1, num + 1):
            pts = np.argwhere(labeled == lbl)
            centroid = pts.mean(axis=0)
            d = np.linalg.norm(centroid - body_center_vox)
            if d < best_dist:
                best_dist = d
                best_label = lbl
        component_mask = labeled == best_label
        voxel_count = int(component_mask.sum())
        if voxel_count == 0:
            continue

        volume_mm3 = voxel_count * (dx ** 3)
        volume_cm3 = volume_mm3 / 1000.0

        dilated = ndimage.binary_dilation(component_mask, iterations=1)
        # Cooling surface must exclude contact with part/runner/gating metal;
        # only faces exposed to the mould count for Chvorinov/modulus.
        surface_mask = dilated & (grid == 0)
        surface_mm2 = float(surface_mask.sum()) * dx * dx
        m_riser = volume_mm3 / surface_mm2 if surface_mm2 > 0 else 0.0

        riser_centroid_vox = np.array(np.argwhere(component_mask).mean(axis=0))
        nearest_hs = None
        nearest_m = 0.0
        nearest_pos = np.zeros(3)
        nearest_resistance = 0.0
        if hotspots:
            hs_positions_vox = np.array(
                [(hs.position_mm - origin_mm) / dx for hs in hotspots]
            )
            tree = cKDTree(hs_positions_vox.astype(np.float32))
            d, idx = tree.query(riser_centroid_vox.astype(np.float32), k=1)
            nearest_hs = hotspots[idx]
            nearest_m = nearest_hs.m_value_mm
            nearest_pos = nearest_hs.position_mm
            nearest_resistance = nearest_hs.darcy_resistance

        # v9.3: apply user-defined feeder type / modulus and compute effective modulus.
        feeder_type = (body.feeder_type or "conventional").lower().strip()
        if not feeder_type or feeder_type == "":
            feeder_type = "conventional"
        # Multiplier on the riser modulus that accounts for exothermic/insulated sleeves.
        # Chilled feeders lose metal quickly, so their effective modulus is reduced.
        feeder_modulus_factor = {
            "conventional": 1.0,
            "exothermic": getattr(alloy, "exothermic_modulus_factor", 1.5),
            "insulated": 1.2,
            "sleeve": 1.1,
            "chilled": 0.8,
        }.get(feeder_type, 1.0)
        # Volume yield: exothermic mini-risers supply the same modulus with less metal.
        feeder_volume_yield = {
            "conventional": 1.0,
            "exothermic": getattr(alloy, "exothermic_volume_yield", 0.45),
            "insulated": 0.8,
            "sleeve": 0.9,
            "chilled": 1.2,
        }.get(feeder_type, 1.0)

        # If the user entered an explicit feeder modulus, use it as the base modulus.
        if body.feeder_m_mm > 0.0:
            m_riser_base = float(body.feeder_m_mm)
        else:
            m_riser_base = m_riser
        m_riser_eff = m_riser_base * feeder_modulus_factor

        # v8.6: existing riser must satisfy both the local hotspot and the global part modulus.
        m_cast_mm = part_volume_mm3 / part_surface_area_mm2 if part_surface_area_mm2 > 0 else 0.0
        local_m_required = alloy.riser_m_factor * nearest_m
        global_m_required = alloy.riser_m_factor * m_cast_mm
        m_required = max(local_m_required, global_m_required)
        # v9.1: use the actual gravity vector, not only Z.
        riser_centroid_mm = riser_centroid_vox * dx + origin_mm
        if nearest_hs is not None:
            dz = float(np.dot(nearest_pos - riser_centroid_mm, g_unit))
        else:
            dz = 0.0
        gravity = max(0.85, 1.0 - 0.005 * max(0, -dz))
        # v9.1: convert Darcy pressure drop (Pa) to an equivalent metal head (mm).
        rho_g = max(alloy.rho_kg_m3 * 9.81, 1e-6)
        resistance_correction = (nearest_resistance / rho_g) * 1000.0
        effective_m_required = m_required * gravity + resistance_correction
        # Allow 5% engineering tolerance.
        large_enough = m_riser_eff >= 0.95 * effective_m_required if m_required > 0 else True

        # v9.1: larger-than-required risers extend effective feeding distance;
        # undersized ones shorten it.  Store a per-component factor.
        if effective_m_required > 0.0 and m_riser_eff > 0.0:
            ratio = m_riser_eff / effective_m_required
            if large_enough:
                size_factor = min(2.0, 1.0 + 0.5 * max(0.0, ratio - 1.0))
            else:
                size_factor = max(0.5, min(1.0, ratio))
        else:
            size_factor = 1.0
        factor_id = len(factor_by_id)
        factor_by_id.append(float(size_factor))
        riser_factor_map[component_mask] = factor_id

        required_volume_cm3 = 0.0
        volume_ratio_ok = True
        if nearest_hs is not None:
            radius_mm = 2.0 * nearest_m
            radius_vox = radius_mm / dx
            feed_region = (
                _sphere_mask(
                    grid.shape, (nearest_hs.position_mm - origin_mm) / dx, radius_vox
                )
                & part_mask
            )
            feed_volume_mm3 = float(feed_region.sum()) * (dx ** 3)
            required_volume_cm3 = alloy.riser_volume_factor * feed_volume_mm3 / 1000.0
            volume_ratio_ok = volume_cm3 >= required_volume_cm3 * feeder_volume_yield

        part_volume_cm3 = part_volume_mm3 / 1000.0
        riser_mass_kg = volume_cm3 * alloy.density_g_cm3 / 1000.0
        feed_to_part_volume_ratio = (
            volume_cm3 / part_volume_cm3 if part_volume_cm3 > 0.0 else 0.0
        )

        riser_results.append(
            RiserResult(
                body_index=body.index,
                name=body.name,
                volume_cm3=volume_cm3,
                surface_area_cm2=surface_mm2 / 100.0,
                m_value_mm=m_riser,
                target_hotspot_m_mm=nearest_m,
                large_enough=large_enough,
                volume_ratio_ok=volume_ratio_ok,
                nearest_hotspot_position_mm=nearest_pos,
                gravity_factor=gravity,
                effective_m_required=effective_m_required,
                required_volume_cm3=required_volume_cm3,
                resistance_correction_mm=resistance_correction,
                mass_kg=riser_mass_kg,
                feed_to_part_mass_ratio=feed_to_part_volume_ratio,
                feed_to_part_volume_ratio=feed_to_part_volume_ratio,
                feeder_type=feeder_type,
                feeder_m_user_mm=body.feeder_m_mm,
                effective_m_value_mm=m_riser_eff,
            )
        )

    if progress_callback:
        progress_callback(92)

    # v9.1: propagate nearest-riser size factor to every voxel.
    _, nearest_riser = ndimage.distance_transform_edt(riser_factor_map == 0, return_indices=True)
    nearest_riser_id = riser_factor_map[tuple(nearest_riser)]
    riser_factor_field = np.asarray(factor_by_id, dtype=np.float64)[nearest_riser_id]

    # AŞAMA 9: Risk map (Niyama risk scaled by feeding deficit)
    # A low-Niyama region is dangerous only if it cannot be fed.  If a riser/
    # gating source is close enough, the Niyama risk is strongly suppressed.
    with np.errstate(divide="ignore", invalid="ignore"):
        FD_field = alloy.feed_k1 * (2.0 * M_mod) * riser_factor_field
        # feed_risk -> 0 at the feeder, -> 1 far beyond the feeding distance.
        feed_risk = dist_feed / (dist_feed + np.maximum(FD_field, 1.0))
        feed_risk = np.clip(np.nan_to_num(feed_risk, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)
        # Macro shrinkage risk (Niyama < alloy.niyama_macro) scaled by feeding.
        niyama_macro_risk = np.clip(1.0 - niyama / alloy.niyama_macro, 0.0, 1.0)
        niyama_micro_risk = np.clip(1.0 - niyama / alloy.niyama_shrinkage, 0.0, 1.0)
        macro_risk = niyama_macro_risk * feed_risk
        micro_risk = niyama_micro_risk * feed_risk
        risk = 1.0 - (1.0 - macro_risk) * (1.0 - micro_risk)
        # v8.6: risk belongs to the part only; risers/gating/chills are not part porosity.
        risk = np.where(part_mask, risk, 0.0)
        risk = np.nan_to_num(risk, nan=0.0, posinf=0.0, neginf=0.0)
    risk_norm = risk

    # v8.8: estimate pore size from Niyama, local modulus and feeding deficit.
    pore_size_um, pore_size_mm, pore_macro_mask, pore_micro_mask, pore_fine_mask = compute_pore_size(
        niyama, M_mod, feed_risk, alloy, part_mask, t_s=t_s, feeder_mask=feeder_mask, dx=dx
    )
    # v8.9: per-class display filters (macro top 60%, micro top 40%, fine top 20%).
    baseline_min_um = max(alloy.gas_pore_baseline_um, alloy.dendrite_spacing_mm * 1000.0 * 0.02)
    pore_macro_percent = 60.0
    pore_micro_percent = 40.0
    pore_fine_percent = 20.0
    pore_threshold_um = pore_size_threshold_um(
        pore_size_um,
        pore_macro_percent,
        micro_pore_limit_um=alloy.micro_pore_limit_um,
        baseline_um=baseline_min_um,
    )
    pore_macro_threshold_um = pore_size_threshold_um(
        np.where(pore_macro_mask, pore_size_um, 0.0),
        pore_macro_percent,
        micro_pore_limit_um=alloy.micro_pore_limit_um,
        baseline_um=baseline_min_um,
    )
    pore_micro_threshold_um = pore_size_threshold_um(
        np.where(pore_micro_mask, pore_size_um, 0.0),
        pore_micro_percent,
        micro_pore_limit_um=alloy.micro_pore_limit_um,
        baseline_um=baseline_min_um,
    )
    pore_fine_threshold_um = pore_size_threshold_um(
        np.where(pore_fine_mask, pore_size_um, 0.0),
        pore_fine_percent,
        micro_pore_limit_um=alloy.micro_pore_limit_um,
        baseline_um=baseline_min_um,
    )

    # Assign pore-size estimate to each hot spot and re-evaluate feed_ok with
    # the riser-size-modulated feeding distance.
    for hs in hotspots:
        vox_raw = np.round((hs.position_mm - origin_mm) / dx).astype(int)
        vox = _snap_to_part(vox_raw)
        if 0 <= vox[0] < grid.shape[0] and 0 <= vox[1] < grid.shape[1] and 0 <= vox[2] < grid.shape[2]:
            ps_um = float(pore_size_um[vox[0], vox[1], vox[2]])
            hs.pore_size_um = ps_um
            hs.pore_size_mm = ps_um / 1000.0
            hs.pore_size_class = _pore_size_class(
                ps_um,
                macro_threshold_um=alloy.macro_pore_limit_um,
                micro_threshold_um=alloy.micro_pore_limit_um,
            )
            # v9.1: effective feeding distance depends on the nearest riser size.
            factor = float(riser_factor_field[vox[0], vox[1], vox[2]])
            hs.max_feeding_distance_mm = hs.max_feeding_distance_mm * factor
            feed_cost_ok = hs.feeding_cost < 30.0
            hs.feed_ok = (
                (not np.isinf(hs.dist_to_riser_mm))
                and (hs.dist_to_riser_mm <= hs.max_feeding_distance_mm)
                and hs.directional_ok
                and hs.heuvers_ok
                and feed_cost_ok
                and hs.darcy_ok
            )

    if progress_callback:
        progress_callback(95)

    # AŞAMA 10: Local refinement around hot spots
    local_regions: List[RefinementRegion] = []
    if refine_local and hotspots:
        for idx, hs in enumerate(hotspots):
            if progress_callback:
                progress_callback(95 + int((idx + 1) / len(hotspots) * 5))
            region = _refine_region(
                bodies,
                hs,
                idx,
                origin_mm,
                dx,
                alloy,
                mold,
                base_res,
                max_res,
                progress_callback,
            )
            if region is not None:
                local_regions.append(region)

    result = AnalysisResult(
        grid=grid,
        origin_mm=origin_mm,
        dx_mm=dx,
        is_metal=is_metal,
        sdf=sdf,
        dist_to_riser=dist_feed,
        risk=risk_norm,
        solidification_time=t_s,
        niyama=niyama,
        gradient_magnitude=G,
        hotspots=hotspots,
        riser_results=riser_results,
        gate_result=None,
        local_regions=local_regions,
        alloy_key=alloy_key,
        mold_key=mold_key,
        alloy_name=alloy.name,
        mold_name=mold.name,
        chvorinov_c=chvorinov_c,
        unit_scale=1.0,
        dominant_m_mm=dominant_m,
        wall_thickness_mm=wall_thickness,
        temperature=temperature,
        cooling_rate=cooling_rate,
        solid_fraction=solid_fraction,
        curvature_mean=mean_curv,
        curvature_gaussian=gauss_curv,
        subvoxel_sdf=sdf,
        shape_factor_global=shape_factor_global,
        m_mean_mm=m_mean,
        m_std_mm=m_std,
        m_skewness=m_skew,
        niyama_variants=niyama_variants,
        elapsed_s=time.time() - t_start,
        casting_params=casting_params,
        thermal_divergence=thermal_divergence,
        bbox_size_mm=bbox_size,
        part_volume_mm3=part_volume_mm3,
        part_surface_area_mm2=part_surface_area_mm2,
        pore_size_um=pore_size_um,
        pore_size_mm=pore_size_mm,
        pore_size_macro_mask=pore_macro_mask,
        pore_size_micro_mask=pore_micro_mask,
        pore_size_fine_mask=pore_fine_mask,
        pore_size_noise_percent=pore_macro_percent,
        pore_size_threshold_um=pore_macro_threshold_um,
        pore_size_macro_percent=pore_macro_percent,
        pore_size_macro_threshold_um=pore_macro_threshold_um,
        pore_size_micro_percent=pore_micro_percent,
        pore_size_micro_threshold_um=pore_micro_threshold_um,
        pore_size_fine_percent=pore_fine_percent,
        pore_size_fine_threshold_um=pore_fine_threshold_um,
    )

    result.riser_proposals = propose_risers(result, alloy, existing_riser_count=len(riser_results))

    # AŞAMA 11: Connect gating solver so the report receives mass / sprue / runner data.
    try:
        from core.gating import analyze_gating
        result.gate_result = analyze_gating(
            result, casting_params=casting_params, bodies=bodies
        )
    except Exception as exc:
        result.recommendations.append(f"Gating analizi atlandı: {exc}")

    result.recommendations = _build_recommendations(result, alloy, mold)
    if result.gate_result and result.gate_result.gating_system_reason:
        result.recommendations.append(result.gate_result.gating_system_reason)
    return result


def _build_recommendations(
    result: AnalysisResult, alloy: Alloy, mold: MoldMaterial
) -> List[str]:
    recs: List[str] = []

    # v9.0: 1:1 scale sanity check.
    part_volume_cm3 = result.part_volume_mm3 / 1000.0
    bbox_max_mm = float(np.max(result.bbox_size_mm))
    if bbox_max_mm < 0.1 or bbox_max_mm > 10000.0 or part_volume_cm3 < 0.01 or part_volume_cm3 > 1e6:
        recs.append(
            f"UYARI: Ölçek / 1:1 kontrolü yapılmalı. Parça hacmi = {part_volume_cm3:.3f} cm³, "
            f"kutu boyu = {bbox_max_mm:.2f} mm. STEP dosyasının mm biriminde ve 1:1 ölçekte olduğundan emin olun."
        )

    has_riser = (result.grid == BodyType.RISER).any()
    if not has_riser:
        recs.append(
            "Ayrı besleyici (riser) atanmamış; döküm ağzı / yolluk / meme kaynağından "
            "besleme mesafesi ve yol maliyeti hesaplandı. Soğuk birleşme ve çekinti riski "
            "daha yüksek olabilir; kritik bölgeler için besleyici eklenmesi önerilir."
        )

    recs.append(
        f"Malzeme: {alloy.name} | Kalıp: {mold.name} | Chvorinov C = {result.chvorinov_c:.4f} s/mm² | "
        f"Baskın M = {result.dominant_m_mm:.2f} mm (t ≈ {result.wall_thickness_mm:.2f} mm) | "
        f"Şekil faktörü SF = {result.shape_factor_global:.6f}"
    )
    recs.append(
        f"Modül istatistikleri: ortalama M = {result.m_mean_mm:.2f} mm, std = {result.m_std_mm:.2f} mm, "
        f"çarpıklık = {result.m_skewness:.2f}. "
        + ("Parça duvar kalınlığı dengesiz." if abs(result.m_skewness) > 1.0 else "Kalınlık dağılımı nispeten dengeli.")
    )

    if not result.hotspots:
        recs.append(
            "Kritik sıcak nokta (hot spot) tespit edilmedi. Model çok ince veya geometri düzgün okunamamış olabilir."
        )
        return recs

    for hs in result.hotspots:
        t = hs.t_section_mm
        W = hs.width_mm
        fd = alloy.feed_k1 * t
        unc = hs.m_uncertainty_mm
        recs.append(
            f"Hot spot M = {hs.m_value_mm:.2f} ± {unc:.2f} mm, t = {t:.2f} mm, "
            f"W = {W:.2f} mm, şekil faktörü = {hs.shape_factor:.6f}"
        )
        if hs.dist_to_riser_mm > hs.max_feeding_distance_mm:
            recs.append(
                f"Hot spot: besleme mesafesi {hs.dist_to_riser_mm:.1f} mm > limit {hs.max_feeding_distance_mm:.1f} mm (FD={fd:.1f} mm). "
                f"Besleyiciyi yakın taşı veya kesiti büyütün."
            )
        if not hs.directional_ok:
            recs.append(
                f"Hot spot: yönlü katılaşma bozuk, yolda daralma (boyun M={hs.min_neck_m_mm:.1f} mm). "
                f"Meme/besleyici arasındaki geometriyi kalınlaştırın."
            )
        if not hs.heuvers_ok:
            recs.append(
                "Hot spot: Heuver çemberi kuralı ihlali - besleme yolunda kesit daralıyor, "
                "ara bölge daha ince/sıcak. Meme konumunu/kalınlığını gözden geçirin."
            )
        if not hs.darcy_ok:
            if hs.darcy_resistance < 0.01:
                recs.append(
                    "Hot spot: Besleme yolunda eriyik oranı çok düşük, katılaşmış bölge geçilemiyor. "
                    "Mesafeyi kısaltın, kesiti büyütün veya yerel besleyici ekleyin."
                )
            else:
                recs.append(
                    f"Hot spot: Darcy basınç kaybı ({hs.darcy_resistance:.2f} Pa) mevcut hidrostatik basıncı aşıyor. "
                    f"Mushy-zone geçirgenliği yetersiz; meme/yol kesitini büyütün veya kısa yol seçin."
                )

        niy = hs.niyama_ensemble
        if niy < alloy.niyama_macro:
            if hs.feed_ok and hs.darcy_ok:
                recs.append(
                    f"Hot spot: Niyama {niy:.2f} < {alloy.niyama_macro} ama "
                    f"besleyici ile beslenebiliyor. Mikro çekinti/porozite için "
                    f"besleyici hacim/boyun kontrolü yapın."
                )
            else:
                recs.append(
                    f"Hot spot: Niyama {niy:.2f} < {alloy.niyama_macro} -> "
                    f"makro shrinkage / çekinti riski yüksek; besleme yetersiz."
                )
        elif niy < alloy.niyama_shrinkage:
            if hs.feed_ok and hs.darcy_ok:
                recs.append(
                    f"Hot spot: Niyama {niy:.2f} < {alloy.niyama_shrinkage}; "
                    f"besleyici var ancak mikro gözenek / shrinkage porozite riski "
                    f"takip edilmeli."
                )
            else:
                recs.append(
                    f"Hot spot: Niyama {niy:.2f} < {alloy.niyama_shrinkage} -> "
                    f"mikro gözenek / shrinkage porozite riski."
                )

    for rr in result.riser_results:
        eff_m = max(rr.effective_m_value_mm, rr.m_value_mm)
        type_text = f" ({rr.feeder_type})" if rr.feeder_type else ""
        if not rr.large_enough:
            increase = (
                (rr.effective_m_required / max(eff_m, 1e-6) - 1.0) * 100.0
            )
            recs.append(
                f"{rr.name}{type_text}: M_besleyici={eff_m:.2f} mm (gerçek {rr.m_value_mm:.2f} mm) < gerekli {rr.effective_m_required:.2f} mm. "
                f"Besleyici modülünü %{int(increase)} büyütün."
            )
        if not rr.volume_ratio_ok:
            short = rr.required_volume_cm3 - rr.volume_cm3
            recs.append(
                f"{rr.name}{type_text}: hacim yetersiz (V={rr.volume_cm3:.2f} cm³, gerekli {rr.required_volume_cm3:.2f} cm³). "
                f"En az {short:.2f} cm³ daha hacim ekleyin."
            )

    for idx, proposal in enumerate(result.riser_proposals):
        pos = f"({proposal.placement_mm[0]:.1f}, {proposal.placement_mm[1]:.1f}, {proposal.placement_mm[2]:.1f})"
        if proposal.infeasible:
            recs.append(
                f"UYARI {idx + 1}: Hotspot #{proposal.target_hotspot_index + 1} için önerilen "
                f"besleyici/çıkıcı parça geometrisine sığmıyor. "
                f"M={proposal.m_required_mm:.2f} mm, çap={proposal.diameter_mm:.1f} mm, "
                f"V={proposal.volume_cm3:.2f} cm³. Konum {pos} mm. "
                f"{proposal.warning if proposal.warning else 'Çözüm kullanıcı kararıdır.'}"
            )
        elif proposal.shape == "chill":
            recs.append(
                f"ÖNERİ {idx + 1}: çıkıcı (chill) ekle -> "
                f"çap={proposal.diameter_mm:.1f} mm, yükseklik={proposal.height_mm:.1f} mm, "
                f"V={proposal.volume_cm3:.2f} cm³. "
                f"Konum {pos} mm. Neden: {proposal.reason}."
            )
        elif proposal.exothermic:
            recs.append(
                f"ÖNERİ {idx + 1}: ekzotermik mini besleyici ekle -> "
                f"çap={proposal.diameter_mm:.1f} mm, yükseklik={proposal.height_mm:.1f} mm, "
                f"V={proposal.volume_cm3:.2f} cm³, M={proposal.m_required_mm:.2f} mm. "
                f"Konum {pos} mm. Neden: {proposal.reason}."
            )
        else:
            recs.append(
                f"ÖNERİ {idx + 1}: konvansiyonel silindirik besleyici ekle -> "
                f"çap={proposal.diameter_mm:.1f} mm, yükseklik={proposal.height_mm:.1f} mm, "
                f"V={proposal.volume_cm3:.2f} cm³, M={proposal.m_required_mm:.2f} mm. "
                f"Konum {pos} mm. Neden: {proposal.reason}."
            )

    all_feed_ok = all(hs.feed_ok for hs in result.hotspots)
    if all_feed_ok and all(rr.large_enough for rr in result.riser_results):
        if has_riser:
            recs.append(
                "Tüm sıcak noktalar besleyici menzili içinde ve besleyici boyutları yeterli görünüyor."
            )
        else:
            recs.append(
                "Tüm sıcak noktalar gating kaynağı menzili içinde, ancak ayrı besleyici olmadan "
                "shrinkage riski tamamen giderilemeyebilir."
            )

    return recs
