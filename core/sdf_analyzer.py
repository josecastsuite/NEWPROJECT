"""SDF-based geometric + pseudo-thermal casting analyzer - JoseCast v8.0."""

import math
import mmap
import os
import sys
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

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
    BODY_PRESETS,
    MOLDS,
    chvorinov_c_from_properties,
    get_alloy,
    get_body_preset,
    get_mold,
    make_effective_mold,
)
from core.riser_designer import propose_risers
from core.enthalpy_lut import build_H_T_fs_LUT, compute_H_field
from core.confluence_reeb import (
    find_local_maxima,
    find_saddles_gate_watershed,
    find_saddles_skeleton,
    find_saddles_sublevel,
    find_saddles_watershed,
)
from core.confluence_sfer import compute_sfer_risk, splat_saddle_risks
from core.confluence_lines import extract_confluence_lines
from core.peclet import peclet_front_velocity_m_s
from core.sphere_lut import get_sphere_64_6
from core.thermal_solver import _alloy_to_dict, _dscheil_dT, _scheil_fs, solve_3d_thermal
from core.voxel_arena import VoxelArena
from core.voxelizer import build_part_grid


def _exposed_surface_area_mm2(mask: np.ndarray, grid: np.ndarray, dx: float) -> float:
    """Compute the exact 6-neighbour exposed surface area of ``mask``.

    Only faces whose neighbour cell is empty (``grid == 0``) count for
    Chvorinov/modulus; this replaces the approximate ``dilated & (grid==0)``
    voxel-shell count that can over/underestimate the cooling area by a
    factor of several for small or thin bodies.
    """
    if not mask.any() or dx <= 0.0:
        return 0.0
    pad_m = np.pad(mask.astype(bool), 1, constant_values=False)
    pad_e = np.pad((grid == 0), 1, constant_values=True)
    face_area = dx * dx
    area = 0.0

    # Axis 0 (z)
    area += float(np.sum(pad_m[:-1, :, :] & ~pad_m[1:, :, :] & pad_e[1:, :, :])) * face_area
    area += float(np.sum(pad_m[1:, :, :] & ~pad_m[:-1, :, :] & pad_e[:-1, :, :])) * face_area
    # Axis 1 (y)
    area += float(np.sum(pad_m[:, :-1, :] & ~pad_m[:, 1:, :] & pad_e[:, 1:, :])) * face_area
    area += float(np.sum(pad_m[:, 1:, :] & ~pad_m[:, :-1, :] & pad_e[:, :-1, :])) * face_area
    # Axis 2 (x)
    area += float(np.sum(pad_m[:, :, :-1] & ~pad_m[:, :, 1:] & pad_e[:, :, 1:])) * face_area
    area += float(np.sum(pad_m[:, :, 1:] & ~pad_m[:, :, :-1] & pad_e[:, :, :-1])) * face_area

    return float(area)


USE_CPP_POROSITY = (
    os.environ.get("JOSECAST_USE_CPP_POROSITY", "1").lower() in ("1", "true", "yes")
)
if USE_CPP_POROSITY:
    from core.cpp_bridge import JOSECAST_CORE
else:
    JOSECAST_CORE = None
from core.types import (
    BODY_FEEDER_TYPES,
    BODY_METAL_TYPES,
    CHILL_BODY_TYPES,
    AnalysisResult,
    Body,
    BodyType,
    CastingParameters,
    GatingVelocityError,
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


def compute_steiner_modulus(
    sdf: np.ndarray, mean_curv: np.ndarray, gauss_curv: np.ndarray, clip_min: float = 0.5
) -> np.ndarray:
    """
    Steiner shape-corrected local modulus.

    M_mod = SDF / (1 - 2*H*SDF + K*SDF^2)
    where H is mean curvature and K is Gaussian curvature.
    The implementation uses the SDF Laplacian (trace of Hessian) as 2*H and
    the determinant of the Hessian as K, so the expression becomes:
        shape_factor = 1 - mean_curv*SDF + gauss_curv*SDF^2
    SDF and M_mod are in the same length units (mm).

    clip_min is bounded below by 0.5 to prevent the 10x inflation that
    occurred with the old 0.1 limit in concave/discretised regions.
    """
    shape_factor = 1.0 - mean_curv * sdf + gauss_curv * (sdf ** 2)
    shape_factor = np.clip(shape_factor, clip_min, None)
    M_mod = sdf / shape_factor
    # Guard negative or tiny SDF values.
    M_mod = np.where(sdf > 0.0, M_mod, 0.0)
    return M_mod


def _body_mold_material(body: Body, mold: MoldMaterial) -> MoldMaterial:
    """Resolve the mould/contact material represented by a non-metal body."""
    preset = (getattr(body, "mold_preset", None) or "").strip()
    if preset:
        mat = get_body_preset(preset)
        if mat is None:
            mat = get_mold(preset)
        if mat is not None:
            return mat
    # Fallback by body type / feeder type
    if body.body_type in (BodyType.CHILL, BodyType.COOLING_SPRUE):
        return get_body_preset("steel_chill") or mold
    if body.body_type == BodyType.FILTER:
        return get_body_preset("ceramic_foam_filter") or mold
    if body.body_type == BodyType.SLEEVE:
        return get_body_preset("insulating_sleeve") or mold
    if body.body_type == BodyType.RISER:
        ftype = (getattr(body, "feeder_type", None) or "").lower()
        if "exothermic" in ftype:
            return get_body_preset("exothermic_sleeve") or mold
        if "insulated" in ftype or "insulating" in ftype:
            return get_body_preset("insulating_sleeve") or mold
        if "chill" in ftype or "chilled" in ftype:
            return get_body_preset("steel_chill") or mold
        # conventional riser: use the main mould
        return mold
    return mold


def build_local_chvorinov_c_field(
    grid: np.ndarray,
    body_index: Optional[np.ndarray],
    bodies: List[Body],
    mold: MoldMaterial,
    alloy: Alloy,
) -> np.ndarray:
    """
    Per-voxel Chvorinov constant C [dk/cm^2].

    Non-metal voxels (mould + inserts) get the C computed from their material.
    Metal voxels inherit the C of the nearest non-metal voxel, so chills or
    sleeves correctly accelerate/slow down local solidification.
    """
    is_metal = np.isin(grid, BODY_METAL_TYPES)
    C_base = chvorinov_c_from_properties(alloy, mold)
    C_grid = np.full(grid.shape, C_base, dtype=np.float64)

    if body_index is not None and bodies:
        for body in bodies:
            if body is None:
                continue
            mat = _body_mold_material(body, mold)
            C_mat = chvorinov_c_from_properties(alloy, mat)
            mask = (body_index == body.index) & (~is_metal)
            if mask.any():
                C_grid[mask] = C_mat

    C_field = C_grid.copy()
    if is_metal.any():
        # nearest non-metal voxel (is_metal=0 is the feature)
        _, nearest = ndimage.distance_transform_edt(is_metal, return_indices=True)
        C_field[is_metal] = C_grid[nearest[0][is_metal], nearest[1][is_metal], nearest[2][is_metal]]
    return C_field


def build_effusivity_field(
    grid: np.ndarray,
    body_index: Optional[np.ndarray],
    bodies: List[Body],
    mold: MoldMaterial,
) -> np.ndarray:
    """Per-voxel thermal effusivity e = sqrt(k * rho * cp) [J/(m^2 K s^0.5)]."""
    is_metal = np.isin(grid, BODY_METAL_TYPES)
    e_base = math.sqrt(max(mold.k_w_mk * mold.rho_kg_m3 * mold.cp_j_kgk, 0.0))
    e_grid = np.full(grid.shape, e_base, dtype=np.float64)

    if body_index is not None and bodies:
        for body in bodies:
            if body is None:
                continue
            mat = _body_mold_material(body, mold)
            e_mat = math.sqrt(max(mat.k_w_mk * mat.rho_kg_m3 * mat.cp_j_kgk, 0.0))
            mask = (body_index == body.index) & (~is_metal)
            if mask.any():
                e_grid[mask] = e_mat

    e_field = e_grid.copy()
    if is_metal.any():
        _, nearest = ndimage.distance_transform_edt(is_metal, return_indices=True)
        e_field[is_metal] = e_grid[nearest[0][is_metal], nearest[1][is_metal], nearest[2][is_metal]]
    return e_field


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
    # Latent-heat correction: conduction-only dT/dt is reduced by cp/cp_eff
    # in the mushy zone where the apparent heat capacity is boosted by L*dfs/dT.
    if alloy.latent_heat_j_kg > 0:
        cp0 = alloy.cp_j_kgk
        dT_mush = max(alloy.t_liquidus_c - alloy.t_solidus_c, 1.0)
        df_dT = _dscheil_dT(T, alloy.t_liquidus_c, alloy.t_solidus_c, alloy.partition_coefficient)
        cp_eff = cp0 + alloy.latent_heat_j_kg * np.clip(df_dT, 0.0, 1.0 / dT_mush)
        cp_eff = np.where(cp_eff > cp0, cp_eff, cp0)
        cooling_rate = np.where(is_metal, cooling_rate * (cp0 / cp_eff), cooling_rate)
    thermal_divergence = ndimage.laplace(T) / (dx * dx)
    return T, cooling_rate, solid_fraction, thermal_divergence


def compute_thermal_stress(
    temperature: np.ndarray,
    solid_fraction: np.ndarray,
    alloy: Alloy,
    is_metal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simplified thermomechanical stress and crack-risk maps.

    Assumes fully constrained shrinkage: thermal strain = alpha * (Ts - T).
    Stress is capped at the high-temperature yield stress.  Hot-tear risk is
    highest in the mushy zone where liquid films remain and accumulated strain
    exceeds the alloy's hot-tear threshold.  Cold-crack risk is evaluated near
    room temperature against the room-temperature yield stress.
    """
    E = float(getattr(alloy, "young_modulus_pa", 2.1e11))
    alpha = float(getattr(alloy, "thermal_expansion_cinv", 1.2e-5))
    yield_h = float(getattr(alloy, "yield_strength_pa", 2.5e8))
    yield_r = float(getattr(alloy, "room_temp_yield_pa", 4.0e8))
    tear_thr = float(getattr(alloy, "hot_tear_threshold_strain", 0.015))

    dT = np.maximum(alloy.t_solidus_c - temperature, 0.0)
    strain = alpha * dT
    stress = np.clip(E * strain, 0.0, yield_h)

    # Hot tear: strain demand in the mushy zone (fs ~ 0.3..0.95) relative to
    # the alloy's interdendritic ductility.
    mushy = is_metal & (solid_fraction > 0.3) & (solid_fraction < 0.95)
    with np.errstate(divide="ignore", invalid="ignore"):
        hot_tear = np.where(
            mushy, np.clip(strain / max(tear_thr, 1e-9), 0.0, 1.0), 0.0
        )

    # Cold crack: stress relative to room-temperature yield at T <= 100 °C.
    cold_region = is_metal & (temperature <= 100.0) & (solid_fraction >= 0.99)
    cold_crack = np.where(cold_region, np.clip(stress / max(yield_r, 1e-9), 0.0, 1.0), 0.0)

    return stress, hot_tear, cold_crack


def compute_chvorinov_t(M_field: np.ndarray, C: float) -> np.ndarray:
    """
    Chvorinov solidification time: t_s [s].
        t_s = C [dk/cm²] * (M [mm] / 10.0)^2 * 60.0
    C is the mould constant in minutes per square centimetre and M is the
    local casting modulus in millimetres.
    """
    M_cm = np.maximum(M_field, 0.0) / 10.0
    return C * M_cm ** 2 * 60.0


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
    local solidification cooling rate.  If a latent-heat-aware cooling_rate
    field is supplied, it is used directly; otherwise the Chvorinov-based
    effective cooling rate explicitly includes the latent-heat temperature
    equivalent.  The result is weighted by f = M_mod / sdf so that bulky,
    sphere-like regions (f < 1) report lower Niyama (higher shrinkage risk)
    than plates (f ≈ 1).
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
    # Use supplied latent-heat-aware cooling rate where available.
    if cooling_rate is not None and cooling_rate.size:
        R = np.where(
            np.isfinite(cooling_rate) & (cooling_rate > 1e-12),
            np.abs(cooling_rate),
            np.nan,
        )
    else:
        R = np.full_like(t_s, np.nan)
    # Fallback: effective temperature drop = sensible + latent-heat equivalent.
    nan_mask = ~np.isfinite(R)
    dT_eff = (alloy.t_liquidus_c - alloy.t_solidus_c) + alloy.latent_heat_j_kg / max(
        alloy.cp_j_kgk, 1e-9
    )
    R = np.where(nan_mask & (sdf > 0), dT_eff / t_s, R)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
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


def _carlson_gp_pct(ny_star: np.ndarray, b0: float, key: str = "WCB") -> np.ndarray:
    """Return shrinkage pore volume percentage from Carlson-Beckermann curve.

    Carlson & Beckermann, Metall. Mater. Trans. A 40A (2009) 163.
    The fits are evaluated as  -A log10(Ny*) + B  for the log branches and
    C Ny*^{-D} for the power-law branches; gp is capped at the alloy total
    solidification shrinkage b0 (percent).
    """
    safe = np.maximum(ny_star, 1e-12)
    if key == "A356":
        gp = np.where(
            safe <= 1.43,
            -2.068 * np.log10(safe) + 3.160,
            np.where(
                safe <= 18.0,
                4.024 * np.power(safe, -0.9786),
                7.771 * np.power(safe, -1.206),
            ),
        )
    elif key == "AZ91D":
        gp = np.where(
            safe <= 41.0,
            -1.671 * np.log10(safe) + 3.483,
            np.where(
                safe <= 45.2,
                -10.81 * np.log10(safe) + 18.23,
                73.01 * np.power(safe, -1.415),
            ),
        )
    else:  # WCB default
        gp = np.where(
            safe <= 28.2,
            -1.654 * np.log10(safe) + 3.052,
            43.05 * np.power(safe, -1.254),
        )
    b0_safe = np.maximum(b0, 1e-9) if isinstance(b0, np.ndarray) else max(b0, 1e-9)
    return np.clip(gp, 0.0, b0_safe)


def compute_pore_size(
    niyama: np.ndarray,
    M_mod: np.ndarray,
    feed_risk: np.ndarray,
    alloy: Alloy,
    part_mask: np.ndarray,
    t_s: Optional[np.ndarray] = None,
    feeder_mask: Optional[np.ndarray] = None,
    dx: float = 1.0,
    gravity_vector: Tuple[float, float, float] = (0.0, 0.0, -1.0),
    fill_time: Optional[np.ndarray] = None,
    darcy_factor: Optional[np.ndarray] = None,
    velocity_magnitude: Optional[np.ndarray] = None,
    solid_fraction: Optional[np.ndarray] = None,
    mold: Optional[MoldMaterial] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate pore size from the Carlson-Beckermann dimensionless Niyama model.

    The engine Niyama field is converted to the dimensionless Ny* via the alloy
    niyama_star_scale, then the published Carlson-Beckermann curve gives the
    shrinkage pore volume percentage gp.  gp is reduced by the directional
    feeding efficiency and amplified where the Darcy pressure head cannot
    overcome the mushy-zone resistance.

    The pore size is then computed physically from the volume fraction:
    shrinkage pore diameter scales with the cube-root of gp and a local
    characteristic length L = max(2*M_mod, SDAS).  This naturally separates
    micro-shrinkage (L ~ SDAS) from macro-shrinkage (L ~ section thickness).
    A separate gas/oxide micro-porosity contribution is driven by the local
    melt velocity: velocities above the alloy critical entrainment velocity
    (Campbell ~0.5 m/s for Al, higher for ferrous alloys) increase bifilm/gas
    pore size, so gate velocity and gate area have a direct effect on porosity.

    ``fill_time`` supplies per-voxel metal arrival time (s) so that a feeder reached
    late is penalised.  ``darcy_factor`` (>1 where pressure head cannot overcome the
    mushy-zone resistance) directly scales the predicted pore volume in regions
    where liquid metal cannot be supplied fast enough.

    ``solid_fraction`` and ``mold`` drive the fs-dependent graphite expansion
    model for cast irons.  ``mold.mold_rigidity_factor`` determines how much of
    the graphite expansion compensates shrinkage versus pushing the mold wall.

    Returns pore_size_um, pore_size_mm, macro_mask, micro_mask, fine_mask,
    shrinkage_pore_size_um, pore_volume_pct, mold_wall_movement_pct.
    """
    # Directional feeding: a feeder aligned with the solidification front and
    # located above the voxel (opposite to the user-defined gravity vector) is
    # much more effective, but never removes all risk.
    _feeder = feeder_mask if feeder_mask is not None else np.zeros_like(part_mask)
    feed_eff = directional_feed_efficiency(
        t_s, _feeder, part_mask, dx, gravity_vector=gravity_vector, fill_time=fill_time
    )

    # Optional C++ accelerated porosity map.  Call before any large Python
    # temporaries are built so Windows is not asked for a fresh 807 MiB block.
    if USE_CPP_POROSITY and JOSECAST_CORE is not None:
        def _to_3d(arr: Optional[np.ndarray]) -> np.ndarray:
            if arr is None:
                return np.empty((0, 0, 0), dtype=np.float64)
            if arr.ndim == 3:
                return arr.astype(np.float64, copy=False)
            if arr.size == 0:
                return np.empty((0, 0, 0), dtype=np.float64)
            return arr.astype(np.float64, copy=False)

        v_in = _to_3d(velocity_magnitude)
        d_in = _to_3d(darcy_factor)
        fs_in = (
            solid_fraction.astype(np.float64, copy=False)
            if solid_fraction is not None and solid_fraction.ndim == 3 and solid_fraction.shape == niyama.shape
            else np.empty((0, 0, 0), dtype=np.float64)
        )
        try:
            ps_um, ps_mm, macro, micro, fine, shrink, gp, mold_move = JOSECAST_CORE.compute_porosity(
                niyama.astype(np.float64, copy=False),
                M_mod.astype(np.float64, copy=False),
                feed_risk.astype(np.float64, copy=False),
                feed_eff.astype(np.float64, copy=False),
                part_mask.astype(np.uint8, copy=False),
                v_in,
                d_in,
                _alloy_to_dict(alloy),
                alloy.carlson_curve_key,
                alloy.material_family,
                fs_in,
                alloy.carbon_equivalent,
                float(mold.mold_rigidity_factor) if mold is not None else 1.0,
                alloy.graphite_expansion_fraction,
                alloy.inoculation_factor,
            )
            return (
                ps_um,
                ps_mm,
                macro.astype(bool),
                micro.astype(bool),
                fine.astype(bool),
                shrink,
                gp,
                mold_move,
            )
        except Exception as exc:
            print(
                f"[Porosity] C++ imza/argüman hatası, Python fallback kullanılıyor: {exc}",
                file=sys.stderr,
            )

    # Python fallback (C++ disabled or failed).
    valid = part_mask & np.isfinite(niyama) & (niyama > 0.0)

    # ---- fs-dependent shrinkage / graphite expansion (cast irons) ----
    def _graphite_cumulative(fs: np.ndarray) -> np.ndarray:
        """Cumulative graphite expansion fraction up to the current solid fraction.

        The eutectic tent (peak = 1 at fs_center) is integrated so the total
        expansion equals ``graphite_expansion_fraction`` once solidification has
        passed the CE-dependent eutectic band (fs >= fs_end).  Cells inside the
        band receive a partial, smooth expansion.
        """
        family = alloy.material_family
        if family not in ("gray_iron", "ductile_iron", "white_iron"):
            return np.zeros_like(fs)
        ce = max(alloy.carbon_equivalent, 0.0)
        if ce <= 0.0:
            return np.zeros_like(fs)
        fs_center = float(np.clip(0.7 - 0.12 * (ce - 4.3), 0.1, 0.9))
        fs_half_width = 0.12
        fs_start = fs_center - fs_half_width
        fs_end = fs_center + fs_half_width
        total_area = 0.5 * (fs_end - fs_start)
        if total_area <= 0.0:
            return np.zeros_like(fs)

        left = (fs > fs_start) & (fs < fs_center)
        right = (fs >= fs_center) & (fs < fs_end)
        full = fs >= fs_end
        denom_left = max(fs_center - fs_start, 1e-9)
        denom_right = max(fs_end - fs_center, 1e-9)

        left_area = np.where(left, 0.5 * (fs - fs_start) ** 2 / denom_left, 0.0)
        mid_area = 0.5 * (fs_center - fs_start)
        df = np.where(right, fs - fs_center, 0.0)
        right_area = np.where(right, mid_area + df - 0.5 * df ** 2 / denom_right, 0.0)
        area = left_area + right_area + np.where(full, total_area, 0.0)
        return np.clip(area / total_area, 0.0, 1.0)

    fs_input = (
        solid_fraction
        if solid_fraction is not None and solid_fraction.shape == niyama.shape
        else np.zeros_like(niyama)
    )
    graphite_frac = _graphite_cumulative(fs_input)
    rigidity = float(mold.mold_rigidity_factor) if mold is not None else 1.0
    rigidity = np.clip(rigidity, 0.0, 1.0)
    expansion = alloy.graphite_expansion_fraction * alloy.inoculation_factor * graphite_frac
    compensated = expansion * rigidity
    net_shrink = alloy.shrinkage_factor - compensated
    b0_eff = np.where(valid, np.clip(net_shrink * 100.0, 0.0, None), alloy.shrinkage_factor * 100.0)
    mold_wall_movement = np.where(
        valid, np.clip(expansion * (1.0 - rigidity) * 100.0, 0.0, None), 0.0
    )

    # (C++ porosity path is attempted before this fallback code.)

    feed_factor = np.power(np.clip(feed_risk, 0.0, 1.0), alloy.feed_risk_exponent) * feed_eff

    # Carlson-Beckermann dimensionless Niyama -> shrinkage pore volume %.
    ny_star = niyama * alloy.niyama_star_scale
    b0 = b0_eff  # per-voxel effective solidification shrinkage [%]
    gp_pct = _carlson_gp_pct(ny_star, b0, key=alloy.carlson_curve_key)
    # Apply feeding efficiency; keep zero for invalid (surface/boundary) voxels.
    # Where hydrostatic/Darcy balance says the pressure head cannot supply enough
    # liquid, the pore volume is amplified by the supplied darcy_factor.
    if darcy_factor is not None and darcy_factor.shape == gp_pct.shape:
        gp_pct = np.where(valid, gp_pct * feed_factor * darcy_factor, 0.0)
    else:
        gp_pct = np.where(valid, gp_pct * feed_factor, 0.0)
    # Total shrinkage porosity cannot exceed the alloy's total solidification
    # shrinkage, even where feeding is completely blocked.
    gp_pct = np.clip(gp_pct, 0.0, b0)

    # Convert predicted volume percentage to a physical pore size.
    # The alloy's `pore_size_um_per_porosity_pct` calibrates the representative
    # pore diameter directly from the predicted pore volume percentage; e.g.
    # 1 % porosity -> 1000 um for the default alloy calibration.  This avoids
    # the previous over-estimate caused by taking the whole section thickness
    # (2*M_mod) as the pore spacing, which produced pores as large as the wall.
    sdas_um = alloy.dendrite_spacing_mm * 1000.0
    max_d_um = np.maximum(2.0 * M_mod * 1000.0, sdas_um)
    d_shrinkage_um = np.where(
        valid,
        np.clip(
            gp_pct
            * alloy.pore_size_um_per_porosity_pct
            * alloy.pore_size_length_factor,
            0.0,
            max_d_um,
        ),
        0.0,
    )
    d_shrinkage_um = np.nan_to_num(d_shrinkage_um, nan=0.0, posinf=0.0, neginf=0.0)

    # Gas/oxide micro-porosity baseline: always present, larger in thicker /
    # lower-Niyama regions.  It is further amplified when the local melt velocity
    # exceeds the critical entrainment velocity (gate velocity -> bifilms).
    baseline_min_um = max(alloy.gas_pore_baseline_um, sdas_um * 0.02)
    raw_micro = np.clip(1.0 - niyama / max(alloy.niyama_shrinkage, 1e-9), 0.0, 1.0)
    m_max = float(np.max(M_mod[part_mask])) if np.any(part_mask) else 1.0
    m_rel = np.clip(M_mod / max(m_max, 1e-9), 0.0, 1.0)
    baseline_factor = (
        1.0
        + alloy.gas_pore_time_factor * m_rel
        + alloy.gas_pore_niyama_factor * raw_micro
    )
    baseline_um = baseline_min_um * np.clip(baseline_factor, 1.0, None)

    v_mag = (
        np.asarray(velocity_magnitude, dtype=np.float64)
        if velocity_magnitude is not None
        else np.zeros_like(part_mask, dtype=np.float64)
    )
    entrainment = np.zeros_like(part_mask, dtype=np.float64)
    v_crit = float(alloy.critical_entrainment_velocity_m_s)
    if v_crit > 1e-9:
        v_over = np.where(
            part_mask,
            np.maximum(v_mag - v_crit, 0.0) / v_crit,
            0.0,
        )
        entrainment = np.where(
            part_mask,
            np.power(v_over, alloy.pore_entrainment_exponent),
            0.0,
        )
    d_gas_um = baseline_um * (1.0 + alloy.pore_entrainment_factor * entrainment)

    shrinkage_pore_size_um = d_shrinkage_um
    pore_size_um = np.where(part_mask, np.maximum(d_shrinkage_um, d_gas_um), 0.0)
    pore_size_mm = pore_size_um / 1000.0

    pore_size_um = np.nan_to_num(pore_size_um, nan=0.0, posinf=0.0, neginf=0.0)
    pore_size_mm = np.nan_to_num(pore_size_mm, nan=0.0, posinf=0.0, neginf=0.0)

    macro_thr = alloy.macro_pore_limit_um
    micro_thr = alloy.micro_pore_limit_um

    # Local defect risk: shrinkage volume relative to the macro reference plus
    # the velocity-driven entrainment term.  This is used to keep the class masks
    # from being dominated by the ever-present gas baseline.
    gp_ref = alloy.macro_pore_limit_um / max(alloy.pore_size_um_per_porosity_pct, 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        defect_risk = np.clip(gp_pct / max(gp_ref, 1e-9) + entrainment, 0.0, 50.0)
    risk_local = 1.0 - np.exp(-defect_risk)

    macro_mask = (pore_size_um >= macro_thr) & part_mask & (risk_local > 0.01)
    micro_mask = (
        (pore_size_um >= micro_thr) & (pore_size_um < macro_thr) & part_mask & (risk_local > 0.01)
    )
    fine_mask = (
        (pore_size_um > 0.0) & (pore_size_um < micro_thr) & part_mask & (risk_local > 0.01)
    )

    return (
        pore_size_um,
        pore_size_mm,
        macro_mask,
        micro_mask,
        fine_mask,
        shrinkage_pore_size_um,
        gp_pct,
        mold_wall_movement,
    )


def compute_cold_shot_risk(
    part_mask: np.ndarray,
    fill_time: Optional[np.ndarray],
    velocity_magnitude: Optional[np.ndarray],
    temperature: np.ndarray,
    t_solid: np.ndarray,
    M_mod: np.ndarray,
    alloy: Alloy,
    t_pour_c: float,
    t_mold_c: float,
    dx: float,
    origin_mm: np.ndarray,
    t_liq: Optional[np.ndarray] = None,
    mold: Optional[Any] = None,
    feeder_mask: Optional[np.ndarray] = None,
    feed_risk: Optional[np.ndarray] = None,
    velocity_m_s: Optional[np.ndarray] = None,
    sdf: Optional[np.ndarray] = None,
    curvature_mean: Optional[np.ndarray] = None,
    curvature_gauss: Optional[np.ndarray] = None,
    C_field: Optional[np.ndarray] = None,
    e_field: Optional[np.ndarray] = None,
    H_field: Optional[np.ndarray] = None,
    dist_feed: Optional[np.ndarray] = None,
    grid: Optional[np.ndarray] = None,
    body_index: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], np.ndarray]:
    """Estimate per-voxel cold-shut (soğuk birleşme) risk and the last fill point.

    The model is built on three physically grounded observations:

    1. Fill-temperature drop is governed by the GLOBAL casting solidification
       time (Chvorinov), not by the local cell.  A late-filled cell receives
       freshly poured metal, so it does not cool from the start of the pour.
    2. Local solidification after filling depends on the local modulus
       (Chvorinov).  A cell with a very short liquidus window solidifies
       before later fronts can weld with it.
    3. Cold shuts form where two fronts meet at low kinetic energy.  Fast
       jets weld; slow / stagnant / multi-directional flow (confluence)
       produces cold shuts (Kashiwai et al., J. JFS 78/2006; Feng & Liao,
       China Foundry 2021).

    Risk is the product of:

        thermal_risk       : fill-temperature superheat (global Chvorinov
                             cooling) times the local liquidus-time window
        flow_risk          : low-velocity stagnation amplified by multi-axis
                             confluence from the 3-D velocity field
        thin_section       : small-modulus sections cool faster
        mold_chill_factor  : metal/ceramic moulds extract heat faster
                             (effusivity)
        feeder_factor      : feeders keep the surrounding metal hotter

    Returns ``(cold_shot_risk, lap_risk, cold_shot_saddles,
    last_fill_point_mm, cold_shot_risk_viz, lap_risk_viz, cold_shot_lines)``.
    ``cold_shot_saddles`` is a diagnostics dictionary and ``last_fill_point_mm``
    is an empty array when no valid fill data exists.

    All large per-voxel intermediates are allocated from a ``VoxelArena`` and
    computed in-place, so Windows does not need to hand out a fresh 526 MiB
    contiguous block for every ``np.where`` / ``np.clip`` call.
    """
    empty_lines: List[Dict[str, Any]] = []
    if fill_time is None or fill_time.size == 0:
        return (
            np.zeros(part_mask.shape, dtype=np.float64),
            np.zeros(part_mask.shape, dtype=np.float64),
            {},
            np.array([], dtype=np.float64),
            np.zeros(part_mask.shape, dtype=np.float64),
            np.zeros(part_mask.shape, dtype=np.float64),
            empty_lines,
        )

    shape = part_mask.shape
    n = part_mask.size

    # Convert boolean inputs once; these are 1 byte/voxel, not the memory hog.
    part_mask = part_mask.astype(bool, copy=False)
    ft = np.asarray(fill_time, dtype=np.float64)
    valid_fill = part_mask & (ft < 1.0e6) & np.isfinite(ft) & (ft >= 0.0)
    if not valid_fill.any():
        return (
            np.zeros(part_mask.shape, dtype=np.float64),
            np.zeros(part_mask.shape, dtype=np.float64),
            {},
            np.array([], dtype=np.float64),
            np.zeros(part_mask.shape, dtype=np.float64),
            np.zeros(part_mask.shape, dtype=np.float64),
            empty_lines,
        )

    t_max = float(np.max(ft[valid_fill]))

    # Reserve virtual address space for the scratch arrays used below.  The OS
    # only commits physical pages when the arrays are written, and every large
    # temporary is carved from this one contiguous block.
    reserve_options = [n * 80, n * 56, n * 40, n * 28]
    reserve_options = [max(int(r), 1 << 30) for r in reserve_options]
    arena = None
    last_err = None
    for reserve_bytes in reserve_options:
        try:
            arena = VoxelArena(reserve_bytes)
            break
        except MemoryError as exc:
            last_err = exc
            continue
    if arena is None:
        raise MemoryError(
            f"compute_cold_shot_risk: could not reserve any VoxelArena "
            f"(tried up to {reserve_options[0] / (1024 ** 3):.2f} GiB): {last_err}"
        )

    # Final float64 result in its own mmap so it survives arena cleanup.
    out_bytes = n * np.dtype(np.float64).itemsize
    out_mmap = mmap.mmap(-1, out_bytes, access=mmap.ACCESS_WRITE)
    out = np.frombuffer(out_mmap, dtype=np.float64, count=n).reshape(shape)
    out.fill(0.0)

    try:
        with arena:
            # Scratch float32 arrays; names map to the logical variables below.
            fill_delay = arena.alloc(shape, np.float32, name="fill_delay")
            v_local = arena.alloc(shape, np.float32, name="v_local")
            low_vel = arena.alloc(shape, np.float32, name="low_vel")
            T_meet = arena.alloc(shape, np.float32, name="T_meet")
            t_liq_surf = arena.alloc(shape, np.float32, name="t_liq_surf")
            t_sol_surf = arena.alloc(shape, np.float32, name="t_sol_surf")
            t_super = arena.alloc(shape, np.float32, name="t_super")
            tmp = arena.alloc(shape, np.float32, name="tmp")
            thin = arena.alloc(shape, np.float32, name="thin")
            dt_step = arena.alloc(shape, np.float32, name="dt_step")

            # ---------- thermal_risk : Chvorinov-based fill temperature + local solidification window ----------
            t_liq_c = float(alloy.t_liquidus_c)
            t_sol_c = float(alloy.t_solidus_c)

            C_ch = float(chvorinov_c_from_properties(alloy, mold)) if mold is not None else 1.0

            # Local time to reach the Kashiwai critical solid fraction (fs = 0.52)
            # from the moment the cell is filled, using the alloy's partition
            # coefficient in the Scheil equation.
            cp_m = float(alloy.cp_j_kgk)
            L_m = float(getattr(alloy, "latent_heat_j_kg", 0.0))
            k_partition = float(getattr(alloy, "partition_coefficient", 0.2))
            k = max(min(k_partition, 0.999), 1e-6)
            # fs = 1 - (1 - ratio)^(1/(1-k)); solve fs = 0.52 for ratio
            ratio_fs52 = 1.0 - 0.48 ** (1.0 - k)
            T_52 = t_liq_c - ratio_fs52 * (t_liq_c - t_sol_c)
            H_total = max(cp_m * (t_pour_c - t_sol_c) + L_m, 1e-9)
            H_to_fs52 = max(cp_m * (t_pour_c - T_52), 0.0) + 0.52 * L_m
            frac_to_fs52 = float(H_to_fs52 / H_total)
            ch_const = float(C_ch * 60.0 * frac_to_fs52)

            # t_liq_surf <- local time to reach fs = 0.52 [s]
            t_liq_surf.fill(np.float32(0.0))
            np.copyto(t_liq_surf, M_mod, where=part_mask, casting="unsafe")
            np.maximum(t_liq_surf, np.float32(1e-3), out=t_liq_surf)
            np.divide(t_liq_surf, np.float32(10.0), out=t_liq_surf)
            np.multiply(t_liq_surf, t_liq_surf, out=t_liq_surf)
            np.multiply(t_liq_surf, np.float32(ch_const), out=t_liq_surf)

            # t_super <- local Chvorinov solidification time per voxel [s].
            # M_mod is in mm, C_ch is in min/cm2, so M/10 converts mm to cm.
            t_super.fill(np.float32(0.0))
            np.copyto(t_super, M_mod, where=part_mask, casting="unsafe")
            np.maximum(t_super, np.float32(1e-3), out=t_super)
            np.divide(t_super, np.float32(10.0), out=t_super)
            np.multiply(t_super, t_super, out=t_super)
            np.multiply(t_super, np.float32(C_ch * 60.0), out=t_super)

            # Fill-temperature drop is governed by the LOCAL cooling time.
            # Thick sections keep their superheat; thin/late-filled cells arrive
            # near the liquidus.  T_meet = t_pour - (t_pour - t_liq)*min(ft/t_cool_local, 1).
            np.divide(np.asarray(ft, dtype=np.float32), t_super, out=tmp, casting="unsafe")
            np.clip(tmp, 0.0, 1.0, out=tmp)
            np.subtract(np.float32(t_pour_c), np.float32(t_liq_c), out=t_sol_surf)
            np.multiply(tmp, t_sol_surf, out=tmp)
            np.subtract(np.float32(t_pour_c), tmp, out=T_meet)
            np.clip(T_meet, np.float32(t_mold_c), np.float32(t_pour_c), out=T_meet)

            # Local metal speed (used for confluence and solid-time correction).
            v_local.fill(0.0)
            if velocity_m_s is not None and velocity_m_s.size == 3 * n:
                if velocity_m_s.ndim == 4 and velocity_m_s.shape[0] == 3:
                    vel = velocity_m_s
                elif velocity_m_s.ndim == 1 and velocity_m_s.size == 3 * n:
                    vel = velocity_m_s.reshape((3,) + shape)
                else:
                    vel = None
                if vel is not None:
                    # v_local = sqrt(vx^2 + vy^2 + vz^2)
                    np.copyto(dt_step, vel[0], casting="unsafe")
                    np.multiply(dt_step, dt_step, out=dt_step)
                    np.copyto(v_local, vel[1], casting="unsafe")
                    np.multiply(v_local, v_local, out=v_local)
                    np.add(v_local, dt_step, out=v_local)
                    np.copyto(dt_step, vel[2], casting="unsafe")
                    np.multiply(dt_step, dt_step, out=dt_step)
                    np.add(v_local, dt_step, out=v_local)
                    np.sqrt(v_local, out=v_local)
                    np.copyto(v_local, 0.0, where=~part_mask)
            elif velocity_magnitude is not None and velocity_magnitude.size == n:
                np.copyto(v_local, velocity_magnitude, casting="unsafe")

            v_crit_cold = float(getattr(alloy, "critical_entrainment_velocity_m_s", 0.5))
            # The cold-shut threshold is lower than the entrainment threshold:
            # metal can be slow and still avoid a cold shut, but truly stagnant or
            # converging fronts are dangerous.  Scale the material threshold down.
            v_crit_cold = max(np.float32(0.10), v_crit_cold * 0.45)

            # temperature_factor: 1 at/below liquidus, exp(-(T_meet - t_liq)/safe_super) above.
            superheat = max(t_pour_c - t_liq_c, 1.0)
            safe_super = max(superheat / 3.0, 10.0)
            np.subtract(T_meet, np.float32(t_liq_c), out=tmp)
            np.divide(tmp, np.float32(safe_super), out=tmp)
            np.negative(tmp, out=tmp)
            np.maximum(tmp, np.float32(0.0), out=tmp)
            np.exp(tmp, out=tmp)

            # solid_time_factor: t_liq_local / t_cool_local gives the fraction of
            # the local solidification interval needed to reach fs = 0.52.  Fast
            # local motion delays solidification, so multiply by (1 + (v/vcrit)^2).
            np.divide(t_liq_surf, t_super, out=t_super)
            np.maximum(t_super, np.float32(0.0), out=t_super)
            np.divide(v_local, np.float32(v_crit_cold), out=t_sol_surf)
            np.multiply(t_sol_surf, t_sol_surf, out=t_sol_surf)
            np.add(t_sol_surf, np.float32(1.0), out=t_sol_surf)
            np.multiply(t_super, t_sol_surf, out=t_super)
            np.negative(t_super, out=t_super)
            np.exp(t_super, out=t_super)

            # thermal_risk = temperature_factor * solid_time_factor
            np.multiply(tmp, t_super, out=T_meet)
            np.copyto(T_meet, np.float32(0.0), where=~part_mask)
            np.nan_to_num(T_meet, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            # Superheat gate: final temperature still above liquidus -> no cold shut.
            if temperature is not None and temperature.size == n:
                np.copyto(T_meet, np.float32(0.0), where=part_mask & (temperature > np.float32(t_liq_c)))

            # ---------- flow_risk : low velocity / stagnation + confluence ----------
            # Feng & Liao ridge filter: smooth fill_time (suppress voxel staircasing)
            # and compute gradient magnitude.  High gradient = a sharp fill front.
            finite_ft = ft[part_mask]
            ft_mean = float(finite_ft[np.isfinite(finite_ft)].mean()) if finite_ft.size > 0 else 0.0
            np.copyto(t_liq_surf, np.asarray(ft, dtype=np.float32), casting="unsafe")
            np.copyto(t_liq_surf, np.float32(ft_mean), where=~part_mask)
            ndimage.gaussian_filter(t_liq_surf, sigma=1.0, output=t_liq_surf)

            grad_x = np.gradient(t_liq_surf, axis=0)
            grad_y = np.gradient(t_liq_surf, axis=1)
            grad_z = np.gradient(t_liq_surf, axis=2)
            np.square(grad_x, out=tmp)
            np.square(grad_y, out=t_super)
            np.add(tmp, t_super, out=tmp)
            np.square(grad_z, out=t_super)
            np.add(tmp, t_super, out=tmp)
            np.sqrt(tmp, out=t_sol_surf)  # gradient magnitude |∇ft| in t_sol_surf

            # Front speed from the fill-time gradient: v_front = dx / |∇ft|.
            # This is far more stable than the final static velocity field, which
            # tends to zero after a cell has filled.  Cap at 10 m/s.
            np.add(t_sol_surf, np.float32(1e-9), out=t_liq_surf)
            np.divide(np.float32(dx / 1000.0), t_liq_surf, out=t_liq_surf)
            np.clip(t_liq_surf, np.float32(0.0), np.float32(10.0), out=t_liq_surf)

            finite_grad = t_sol_surf[part_mask]
            mean_grad = float(finite_grad[np.isfinite(finite_grad)].mean()) + 1e-12
            np.divide(t_sol_surf, np.float32(mean_grad), out=t_super)
            np.negative(t_super, out=t_super)
            np.exp(t_super, out=t_super)
            np.subtract(np.float32(1.0), t_super, out=t_super)  # ridge factor in t_super
            np.copyto(t_super, np.float32(0.0), where=~part_mask)

            # Stagnation factor: high when the local front speed is low.
            # Prefer the larger of the flow velocity and the fill-front speed so
            # filled-but-stagnant bulk metal does not get over-penalised.
            np.maximum(v_local, t_liq_surf, out=low_vel)
            np.divide(low_vel, np.float32(v_crit_cold), out=low_vel)
            np.multiply(low_vel, low_vel, out=low_vel)      # (v/vcrit)^2
            np.multiply(low_vel, low_vel, out=low_vel)      # (v/vcrit)^4
            np.negative(low_vel, out=low_vel)
            np.exp(low_vel, out=low_vel)

            # Confluence from 3-D velocity: opposing fronts are detected by the
            # normalised dot product between a voxel and each face neighbour.
            # Dot product < -0.5 means the two cells move in opposite directions.
            fill_delay.fill(0.0)
            if velocity_m_s is not None and velocity_m_s.size == 3 * n:
                if velocity_m_s.ndim == 4 and velocity_m_s.shape[0] == 3:
                    vel = velocity_m_s
                elif velocity_m_s.ndim == 1 and velocity_m_s.size == 3 * n:
                    vel = velocity_m_s.reshape((3,) + shape)
                else:
                    vel = None
                if vel is not None:
                    min_dot = t_liq_surf
                    min_dot.fill(1.0)
                    eps = np.float32(1e-6)

                    # x faces -> t_sol_surf[:-1]
                    np.multiply(vel[0][:-1, :, :], vel[0][1:, :, :], out=t_sol_surf[:-1, :, :])
                    np.multiply(vel[1][:-1, :, :], vel[1][1:, :, :], out=tmp[:-1, :, :])
                    np.add(t_sol_surf[:-1, :, :], tmp[:-1, :, :], out=t_sol_surf[:-1, :, :])
                    np.multiply(vel[2][:-1, :, :], vel[2][1:, :, :], out=tmp[:-1, :, :])
                    np.add(t_sol_surf[:-1, :, :], tmp[:-1, :, :], out=t_sol_surf[:-1, :, :])
                    np.add(v_local[:-1, :, :], eps, out=tmp[:-1, :, :])
                    np.add(v_local[1:, :, :], eps, out=dt_step[:-1, :, :])
                    np.multiply(tmp[:-1, :, :], dt_step[:-1, :, :], out=tmp[:-1, :, :])
                    np.divide(t_sol_surf[:-1, :, :], tmp[:-1, :, :], out=t_sol_surf[:-1, :, :])
                    mask_x = (v_local[:-1, :, :] > eps) & (v_local[1:, :, :] > eps)
                    np.putmask(t_sol_surf[:-1, :, :], ~mask_x, 1.0)
                    np.minimum(min_dot[:-1, :, :], t_sol_surf[:-1, :, :], out=min_dot[:-1, :, :])
                    np.minimum(min_dot[1:, :, :], t_sol_surf[:-1, :, :], out=min_dot[1:, :, :])

                    # y faces -> t_sol_surf[:, :-1]
                    np.multiply(vel[0][:, :-1, :], vel[0][:, 1:, :], out=t_sol_surf[:, :-1, :])
                    np.multiply(vel[1][:, :-1, :], vel[1][:, 1:, :], out=tmp[:, :-1, :])
                    np.add(t_sol_surf[:, :-1, :], tmp[:, :-1, :], out=t_sol_surf[:, :-1, :])
                    np.multiply(vel[2][:, :-1, :], vel[2][:, 1:, :], out=tmp[:, :-1, :])
                    np.add(t_sol_surf[:, :-1, :], tmp[:, :-1, :], out=t_sol_surf[:, :-1, :])
                    np.add(v_local[:, :-1, :], eps, out=tmp[:, :-1, :])
                    np.add(v_local[:, 1:, :], eps, out=dt_step[:, :-1, :])
                    np.multiply(tmp[:, :-1, :], dt_step[:, :-1, :], out=tmp[:, :-1, :])
                    np.divide(t_sol_surf[:, :-1, :], tmp[:, :-1, :], out=t_sol_surf[:, :-1, :])
                    mask_y = (v_local[:, :-1, :] > eps) & (v_local[:, 1:, :] > eps)
                    np.putmask(t_sol_surf[:, :-1, :], ~mask_y, 1.0)
                    np.minimum(min_dot[:, :-1, :], t_sol_surf[:, :-1, :], out=min_dot[:, :-1, :])
                    np.minimum(min_dot[:, 1:, :], t_sol_surf[:, :-1, :], out=min_dot[:, 1:, :])

                    # z faces -> t_sol_surf[:, :, :-1]
                    np.multiply(vel[0][:, :, :-1], vel[0][:, :, 1:], out=t_sol_surf[:, :, :-1])
                    np.multiply(vel[1][:, :, :-1], vel[1][:, :, 1:], out=tmp[:, :, :-1])
                    np.add(t_sol_surf[:, :, :-1], tmp[:, :, :-1], out=t_sol_surf[:, :, :-1])
                    np.multiply(vel[2][:, :, :-1], vel[2][:, :, 1:], out=tmp[:, :, :-1])
                    np.add(t_sol_surf[:, :, :-1], tmp[:, :, :-1], out=t_sol_surf[:, :, :-1])
                    np.add(v_local[:, :, :-1], eps, out=tmp[:, :, :-1])
                    np.add(v_local[:, :, 1:], eps, out=dt_step[:, :, :-1])
                    np.multiply(tmp[:, :, :-1], dt_step[:, :, :-1], out=tmp[:, :, :-1])
                    np.divide(t_sol_surf[:, :, :-1], tmp[:, :, :-1], out=t_sol_surf[:, :, :-1])
                    mask_z = (v_local[:, :, :-1] > eps) & (v_local[:, :, 1:] > eps)
                    np.putmask(t_sol_surf[:, :, :-1], ~mask_z, 1.0)
                    np.minimum(min_dot[:, :, :-1], t_sol_surf[:, :, :-1], out=min_dot[:, :, :-1])
                    np.minimum(min_dot[:, :, 1:], t_sol_surf[:, :, :-1], out=min_dot[:, :, 1:])

                    # confluence: dot < -0.5 -> max, dot >= -0.5 -> 0, linear in between
                    np.add(min_dot, np.float32(0.5), out=fill_delay)
                    np.negative(fill_delay, out=fill_delay)
                    np.clip(fill_delay, 0.0, 0.5, out=fill_delay)
                    np.multiply(fill_delay, np.float32(2.0), out=fill_delay)
                    np.nan_to_num(fill_delay, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            # Modulate confluence by the fill-time ridge factor (suppresses false
            # confluence in flat, uniform fill regions).
            np.multiply(fill_delay, t_super, out=fill_delay)
            np.copyto(fill_delay, np.float32(0.0), where=~part_mask)

            # flow_risk: confluence only matters when the metal is already slow.
            # Fast multi-directional jets tend to weld; slow converging fronts cold shut.
            np.add(fill_delay, np.float32(1.0), out=fill_delay)
            np.multiply(low_vel, fill_delay, out=fill_delay)
            np.clip(fill_delay, 0.0, 1.0, out=fill_delay)

            # ---------- thin_section_factor ----------
            thin.fill(np.float32(np.inf))
            np.copyto(thin, M_mod, where=part_mask, casting="unsafe")
            finite_m = thin[np.isfinite(thin) & (thin > 0.0)]
            m_ref = float(np.percentile(finite_m, 10)) if finite_m.size > 0 else float(dx)
            np.maximum(thin, np.float32(m_ref), out=thin)
            np.divide(np.float32(m_ref), thin, out=thin)
            np.clip(thin, 0.0, 1.0, out=thin)
            np.copyto(thin, 0.0, where=~part_mask)

            # ---------- material / feeder modifiers ----------
            cold_shot_gain = float(getattr(alloy, "cold_shot_gain", 1.25))
            mold_chill_factor = 1.0
            if mold is not None:
                k_m = float(getattr(mold, "k_w_mk", 0.0) or 0.0)
                rho_m = float(getattr(mold, "rho_kg_m3", 0.0) or 0.0)
                cp_m = float(getattr(mold, "cp_j_kgk", 0.0) or 0.0)
                if k_m > 0.0 and rho_m > 0.0 and cp_m > 0.0:
                    e_m = math.sqrt(k_m * rho_m * cp_m)
                    e_ref = math.sqrt(0.58 * 1600.0 * 1170.0)
                    ratio = max(e_m / e_ref, 0.25)
                    mold_chill_factor = 1.0 + 0.25 * math.log(ratio)
                else:
                    alpha = float(getattr(mold, "diffusivity_mm2_s", 0.0) or 0.0)
                    if alpha > 0.0:
                        mold_chill_factor = 1.0 + 0.25 * math.log1p(alpha / 0.31)
                mold_chill_factor = float(np.clip(mold_chill_factor, 0.7, 2.0))

            scale = cold_shot_gain * mold_chill_factor

            # Feeder factor
            t_liq_surf.fill(1.0)
            if feed_risk is not None and feed_risk.size == n:
                np.copyto(t_liq_surf, feed_risk, casting="unsafe")
                np.multiply(t_liq_surf, np.float32(0.8), out=t_liq_surf)
                np.add(t_liq_surf, np.float32(0.2), out=t_liq_surf)
                np.copyto(t_liq_surf, np.float32(1.0), where=~part_mask)
            elif feeder_mask is not None and feeder_mask.size == n and np.any(feeder_mask):
                dist_to_feeder = ndimage.distance_transform_edt(
                    ~feeder_mask, sampling=float(dx)
                )
                L_feed = np.full(shape, float(dx), dtype=np.float64)
                np.multiply(M_mod, 2.5, out=L_feed, where=part_mask)
                np.maximum(L_feed, float(dx), out=L_feed)
                with np.errstate(divide="ignore", invalid="ignore"):
                    np.divide(dist_to_feeder, L_feed, out=dist_to_feeder)
                    np.negative(dist_to_feeder, out=dist_to_feeder)
                    np.exp(dist_to_feeder, out=dist_to_feeder)
                np.multiply(dist_to_feeder, np.float64(-0.8), out=dist_to_feeder)
                np.add(dist_to_feeder, np.float64(1.0), out=dist_to_feeder)
                np.copyto(dist_to_feeder, np.float64(1.0), where=~part_mask)
                np.copyto(t_liq_surf, dist_to_feeder, casting="unsafe")

            # ---------- cold_shot_risk ----------
            np.multiply(T_meet, fill_delay, out=v_local)
            np.multiply(v_local, thin, out=v_local)
            if scale != 1.0:
                np.multiply(v_local, np.float32(scale), out=v_local)
            np.multiply(v_local, t_liq_surf, out=v_local)
            np.nan_to_num(v_local, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(v_local, 0.0, 1.0, out=v_local)

            np.copyto(out, v_local, casting="unsafe")
    except Exception:
        arena.close()
        raise

    # last fill point: find the flat index of a maximum fill_time inside valid_fill.
    t_max = float(np.max(ft[valid_fill]))
    argmax_mask = valid_fill & (ft >= t_max - 1e-12)
    flat_indices = np.flatnonzero(argmax_mask)
    if flat_indices.size > 0:
        flat_idx = int(flat_indices[0])
        idx = np.unravel_index(flat_idx, ft.shape)
        last_fill_point_mm = (
            np.asarray(origin_mm, dtype=np.float64)
            + np.asarray(idx, dtype=np.float64) * float(dx)
        )
    else:
        last_fill_point_mm = np.array([], dtype=np.float64)

    # ---------- V8 SFER (spherical front encounter rate) cold-shot / lap risk ----------
    lap_risk = np.zeros_like(out)
    cold_shot_saddles: Dict[str, Any] = {}
    cold_shot_risk_viz = np.zeros_like(out)
    lap_risk_viz = np.zeros_like(out)
    cold_shot_lines = empty_lines
    if (
        H_field is not None
        and H_field.shape == shape
        and C_field is not None
        and C_field.shape == shape
        and M_mod is not None
        and M_mod.shape == shape
    ):
        # distance to feeder for the feeder factor inside SFER
        if dist_feed is not None and dist_feed.shape == shape:
            dist_feeder = np.asarray(dist_feed, dtype=np.float64)
        elif feeder_mask is not None and feeder_mask.shape == shape:
            dist_feeder = ndimage.distance_transform_edt(~feeder_mask, sampling=float(dx))
            dist_feeder = dist_feeder.astype(np.float64)
        else:
            dist_feeder = np.full(shape, 1e6, dtype=np.float64)

        # front velocity from the fill-time gradient (m/s); mask outside valid fill.
        t_max = float(np.max(ft[valid_fill])) if valid_fill.any() else 1.0
        ft_for_v = np.where(valid_fill, ft, t_max + 1.0)
        v_front = peclet_front_velocity_m_s(ft_for_v, dx)

        # velocity vector field for the head-on closing speed
        if velocity_m_s is not None and velocity_m_s.size == 3 * n:
            if velocity_m_s.ndim == 4 and velocity_m_s.shape[0] == 3:
                velocity = velocity_m_s
            else:
                velocity = velocity_m_s.reshape((3,) + shape)
        else:
            velocity = np.zeros((3,) + shape, dtype=np.float64)

        ft_range = float(np.max(ft[valid_fill])) - float(np.min(ft[valid_fill]))
        base_persistence = 0.1 if (alloy.material_family or "steel").lower() in ("al", "mg") else 0.3
        saddles = np.empty((0, 3), dtype=np.int64)

        # 1) Gate-body seeded watershed (multiple ingates give distinct fronts).
        saddle_pers = np.empty(0, dtype=np.float64)
        if grid is not None and body_index is not None and grid.shape == shape and body_index.shape == shape:
            saddles, _, saddle_pers = find_saddles_gate_watershed(
                ft,
                part_mask,
                grid,
                body_index,
                persistence_thresh_s=base_persistence,
                max_saddles=1000,
                sigma=1.0,
            )

        # 2) Sublevel-set persistence (merge-tree saddles) on the fill-time field.
        if saddles.shape[0] < 10:
            saddles, _, saddle_pers = find_saddles_sublevel(
                ft, part_mask, persistence_thresh_s=base_persistence, max_saddles=1000
            )
            if saddles.shape[0] < 10 and ft_range > 0.0:
                persistence = max(0.001, 0.001 * ft_range)
                saddles, _, saddle_pers = find_saddles_sublevel(
                    ft, part_mask, persistence_thresh_s=persistence, max_saddles=1000
                )

        # 3) Persistent local maxima of fill time are closure/confluence points.
        if saddles.shape[0] < 10:
            saddles, _, saddle_pers = find_local_maxima(
                ft, part_mask, h_relative=0.05, sigma=2.0, max_candidates=1000
            )

        # 4) Medial-axis / skeleton ridges capture confluence lines even when
        #    the fill-time landscape has only one broad closure region.
        if saddles.shape[0] < 10:
            saddles, _, saddle_pers = find_saddles_skeleton(
                ft, part_mask, persistence_thresh_s=base_persistence, ft_percentile=80.0, max_saddles=1000
            )

        if saddles.shape[0] > 0:
            dirs, adj = get_sphere_64_6()
            lut = build_H_T_fs_LUT(alloy, n=1000)
            risk_cs_sfer, risk_lap_sfer, diagnostics = compute_sfer_risk(
                saddles,
                ft,
                H_field,
                M_mod,
                sdf if sdf is not None else np.full(shape, 1.0, dtype=np.float64),
                C_field,
                v_front,
                velocity,
                dist_feeder,
                part_mask,
                lut,
                dirs,
                adj,
                dx,
                alloy,
                saddle_persistence=saddle_pers,
            )
            np.maximum(out, risk_cs_sfer, out=out)
            np.copyto(lap_risk, risk_lap_sfer)
            cold_shot_saddles = {"count": len(diagnostics), "saddles": diagnostics}

            # V8 cold-shot visualisation: splat the saddle-source risk into a
            # small thickness-aware sphere so the whole part is not painted.
            splat_saddle_risks(saddles, out, sdf, dx, 0.3, cold_shot_risk_viz)
            splat_saddle_risks(saddles, risk_lap_sfer, sdf, dx, 0.05, lap_risk_viz)

            cold_shot_lines = extract_confluence_lines(
                ft,
                part_mask,
                out,
                diagnostics,
                origin_mm,
                dx,
                sigma=1.0,
                risk_threshold=0.1,
                min_length_mm=max(3.0, 1.5 * dx),
                max_lines=8,
            )

    return out, lap_risk, cold_shot_saddles, last_fill_point_mm, cold_shot_risk_viz, lap_risk_viz, cold_shot_lines


def compute_erosion_risk(
    velocity_magnitude: Optional[np.ndarray],
    is_metal: np.ndarray,
    alloy,
    mold,
) -> np.ndarray:
    """Per-voxel mold-sand erosion risk driven by local metal velocity.

    Erosion becomes significant when the local metal speed exceeds the
    material-specific threshold (based on Campbell's critical entrainment
    velocity) and is amplified for low-rigidity green-sand molds.  Risk is
    clipped to [0, 1] and zero outside the metal domain.
    """
    if velocity_magnitude is None or velocity_magnitude.size == 0:
        return np.zeros_like(is_metal, dtype=np.float64)
    v = np.asarray(velocity_magnitude, dtype=np.float64)
    v_thresh = float(getattr(alloy, "critical_entrainment_velocity_m_s", 0.5))
    # Low-rigidity molds (green sand) erode at lower velocities.
    rigidity = float(getattr(mold, "mold_rigidity_factor", 1.0))
    v_thresh = v_thresh * max(0.3, rigidity)
    v_max = v_thresh * 3.0
    with np.errstate(divide="ignore", invalid="ignore"):
        risk = (v - v_thresh) / max(v_max - v_thresh, 1e-9)
    risk = np.clip(np.nan_to_num(risk, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    risk = np.where(is_metal, risk, 0.0)
    return risk


def directional_feed_efficiency(
    t_s: np.ndarray,
    feeder_mask: np.ndarray,
    part_mask: np.ndarray,
    dx: float,
    min_eff: float = 0.05,
    max_reduction: float = 0.95,
    gravity_vector: Tuple[float, float, float] = (0.0, 0.0, -1.0),
    fill_time: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a per-voxel feeding-efficiency factor in [min_eff, 1.0].

    The local solidification time gradient points toward the last-freezing
    region.  A riser that lies in that direction can feed the voxel well, so
    shrinkage risk is reduced.  The user-defined gravity vector is also used:
    a feeder located above the voxel (opposite to gravity) feeds more
    effectively because shrinkage voids migrate upward.  Where no gradient
    exists (uniform) the factor is neutral (0.5).  Far from any feeder the
    factor remains ~1.0.

    The feeder solidification time is taken from the hottest voxel inside a
    small 3x3x3 neighbourhood around the nearest feeder voxel, because the
    riser core (not the thin neck/contact face) is the actual liquid reservoir
    that feeds shrinkage.  A well-aligned, liquid-rich feeder can therefore
    reduce the remaining shrinkage risk by up to ``max_reduction`` (default
    95 %, leaving the usual 5 % micro/gas residual).

    If ``fill_time`` (per-voxel metal arrival time, s) is provided, the metal
    travel time from the nearest feeder is subtracted from the feeder's available
    solidification time.  A riser that is reached late, or after the voxel has
    already started to solidify, cannot feed effectively.

    This implementation uses a single anonymous ``mmap`` arena: all large
    float32 intermediates are carved from a contiguous block of *virtual*
    address space; the OS only commits physical pages when the arrays are
    actually written.  The final float64 result is placed in a separate mmap so
    that the returned array remains valid after the arena is released.
    """
    if t_s is None or not feeder_mask.any():
        out = np.ones(part_mask.shape, dtype=np.float64)
        return out

    n = part_mask.size
    shape = part_mask.shape
    nx, ny, nz = shape

    # Reserve 3x the expected peak working set in virtual address space.
    # If Windows cannot back that much page-file space, fall back to smaller
    # multiples until the smallest feasible arena is found.
    reserve_options = [n * 180, n * 128, n * 96, n * 64]
    reserve_options = [max(int(r), 1 << 30) for r in reserve_options]
    arena = None
    last_err = None
    for reserve_bytes in reserve_options:
        try:
            arena = VoxelArena(reserve_bytes)
            break
        except MemoryError as exc:
            last_err = exc
            continue
    if arena is None:
        raise MemoryError(
            f"directional_feed_efficiency: could not reserve any VoxelArena "
            f"(tried up to {reserve_options[0] / (1024 ** 3):.2f} GiB): {last_err}"
        )

    # Final float64 output lives in its own mmap so it survives arena cleanup.
    out_bytes = n * np.dtype(np.float64).itemsize
    out_mmap = mmap.mmap(-1, out_bytes, access=mmap.ACCESS_WRITE)
    out = np.frombuffer(out_mmap, dtype=np.float64, count=n).reshape(shape)
    out.fill(1.0)

    try:
        with arena:
            dx_f = np.float32(dx)

            # t_safe
            t_safe = arena.alloc(shape, np.float32, name="t_safe")
            np.copyto(t_safe, t_s, casting="unsafe")
            finite_vals = t_safe[np.isfinite(t_safe)]
            t_fill = np.float32(finite_vals.max() if finite_vals.size > 0 else 0.0)
            np.nan_to_num(t_safe, nan=t_fill, posinf=0.0, neginf=0.0, copy=False)

            # grad and |grad| in the arena; compute gradient in-place.
            grad = arena.alloc(shape + (3,), np.float32, name="grad")
            _in_place_gradient_3d(t_safe, dx_f, grad)

            grad_mag = arena.alloc(shape, np.float32, name="grad_mag")
            tmp = arena.alloc(shape, np.float32, name="tmp")
            np.multiply(grad[..., 0], grad[..., 0], out=grad_mag)
            np.multiply(grad[..., 1], grad[..., 1], out=tmp)
            np.add(grad_mag, tmp, out=grad_mag)
            np.multiply(grad[..., 2], grad[..., 2], out=tmp)
            np.add(grad_mag, tmp, out=grad_mag)
            np.sqrt(grad_mag, out=grad_mag)

            mask_grad = arena.alloc(shape, bool, name="mask_grad")
            np.greater(grad_mag, 1e-12, out=mask_grad)
            np.divide(grad, grad_mag[..., None], out=grad, where=mask_grad[..., None])

            # Nearest feeder voxel indices (3, nx, ny, nz).
            nearest = arena.alloc((3,) + shape, np.int32, name="nearest")
            ndimage.distance_transform_edt(
                ~feeder_mask, return_distances=False, return_indices=True, indices=nearest
            )

            # diff = (nearest - index_grid) * dx.
            diff = arena.alloc((3,) + shape, np.float32, name="diff")
            x = np.arange(nx, dtype=np.float32)
            y = np.arange(ny, dtype=np.float32)
            z = np.arange(nz, dtype=np.float32)
            np.subtract(nearest[0], x[:, None, None], out=diff[0])
            np.subtract(nearest[1], y[None, :, None], out=diff[1])
            np.subtract(nearest[2], z[None, None, :], out=diff[2])
            np.multiply(diff, dx_f, out=diff)

            diff_norm = arena.alloc(shape, np.float32, name="diff_norm")
            np.multiply(diff[0], diff[0], out=diff_norm)
            np.multiply(diff[1], diff[1], out=tmp)
            np.add(diff_norm, tmp, out=diff_norm)
            np.multiply(diff[2], diff[2], out=tmp)
            np.add(diff_norm, tmp, out=diff_norm)
            np.sqrt(diff_norm, out=diff_norm)

            mask_diff = arena.alloc(shape, bool, name="mask_diff")
            np.greater(diff_norm, 1e-9, out=mask_diff)
            np.divide(diff, diff_norm[None, ...], out=diff, where=mask_diff[None, ...])
            diff_dir = np.moveaxis(diff, 0, -1)

            # Thermal alignment.
            thermal_alignment = arena.alloc(shape, np.float32, name="thermal_alignment")
            np.einsum("...i,...i->...", diff_dir, grad, out=thermal_alignment)
            np.multiply(thermal_alignment, mask_grad, out=thermal_alignment)
            arena.free("grad")

            # Gravity alignment.
            gravity_alignment = arena.alloc(shape, np.float32, name="gravity_alignment")
            g = np.asarray(gravity_vector, dtype=np.float32)
            g_norm = np.linalg.norm(g) + np.float32(1e-12)
            g = g / g_norm
            np.einsum("...i,i->...", diff_dir, -g, out=gravity_alignment)
            arena.free("diff")
            arena.free("diff_norm")
            arena.free("mask_diff")
            arena.free("mask_grad")
            arena.free("tmp")

            # Combine and form reduction factor.
            np.multiply(thermal_alignment, np.float32(0.7), out=thermal_alignment)
            np.multiply(gravity_alignment, np.float32(0.3), out=gravity_alignment)
            np.add(thermal_alignment, gravity_alignment, out=thermal_alignment)
            arena.free("gravity_alignment")
            alignment = thermal_alignment

            np.clip(alignment, np.float32(0.0), np.float32(1.0), out=alignment)
            np.multiply(alignment, np.float32(max_reduction), out=alignment)
            reduction = alignment

            # Flat index of nearest voxel for np.take (avoids advanced-indexing copy).
            flat_idx = arena.alloc(shape, np.int64, name="flat_idx")
            np.multiply(nearest[2], ny, out=flat_idx)
            np.add(flat_idx, nearest[1], out=flat_idx)
            np.multiply(flat_idx, nx, out=flat_idx)
            np.add(flat_idx, nearest[0], out=flat_idx)
            arena.free("nearest")

            # t_feeder: use the hottest voxel in a 3x3x3 neighbourhood around
            # the nearest feeder voxel.  The riser core stays liquid longest; the
            # thin neck or contact face cools faster and would underestimate the
            # available feeding time if sampled directly.
            t_feeder_core = arena.alloc(shape, np.float32, name="t_feeder_core")
            t_feeder_core.fill(np.float32(-np.inf))
            np.copyto(t_feeder_core, t_safe, where=feeder_mask)
            ndimage.maximum_filter(t_feeder_core, size=3, mode="nearest", output=t_feeder_core)

            t_feeder = arena.alloc(shape, np.float32, name="t_feeder")
            np.take(t_feeder_core, flat_idx, out=t_feeder)
            np.nan_to_num(t_feeder, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            arena.free("t_feeder_core")

            if fill_time is not None and fill_time.size == n:
                fill_time_f = arena.alloc(shape, np.float32, name="fill_time_f")
                np.copyto(fill_time_f, fill_time, casting="unsafe")
                np.nan_to_num(fill_time_f, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

                fill_feeder = arena.alloc(shape, np.float32, name="fill_feeder")
                np.take(fill_time_f, flat_idx, out=fill_feeder)
                np.nan_to_num(fill_feeder, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

                # Travel time = fill_time - fill_feeder, clamped to >= 0.
                np.subtract(fill_time_f, fill_feeder, out=fill_time_f)
                np.maximum(fill_time_f, np.float32(0.0), out=fill_time_f)
                np.subtract(t_feeder, fill_time_f, out=t_feeder)
                np.maximum(t_feeder, np.float32(0.0), out=t_feeder)
                arena.free("fill_time_f")
                arena.free("fill_feeder")

            arena.free("flat_idx")

            # time_factor = clamp(t_feeder / max(t_safe, 1e-9), 0, 1).
            np.maximum(t_safe, np.float32(1e-9), out=t_safe)
            np.divide(t_feeder, t_safe, out=t_feeder)
            np.clip(t_feeder, np.float32(0.0), np.float32(1.0), out=t_feeder)
            time_factor = t_feeder
            arena.free("t_safe")

            # efficiency = clamp(1 - reduction * time_factor, min_eff, 1).
            np.multiply(reduction, time_factor, out=reduction)
            np.subtract(np.float32(1.0), reduction, out=reduction)
            np.clip(reduction, np.float32(min_eff), np.float32(1.0), out=reduction)
            efficiency = reduction

            np.copyto(out, efficiency, where=part_mask)
    except Exception:
        arena.close()
        raise

    return out


def _in_place_gradient_3d(t_safe: np.ndarray, dx: float, grad: np.ndarray) -> None:
    """Compute a first-order gradient into the pre-allocated ``grad`` array.

    ``grad[..., i]`` receives the derivative along axis ``i`` using central
    differences in the interior and one-sided differences at the boundaries.
    """
    nx, ny, nz = t_safe.shape
    rdx = 1.0 / float(dx)
    half_rdx = 0.5 * rdx

    if nx > 1:
        # interior
        if nx > 2:
            np.subtract(t_safe[2:], t_safe[:-2], out=grad[1:-1, :, :, 0])
            np.multiply(grad[1:-1, :, :, 0], half_rdx, out=grad[1:-1, :, :, 0])
        # first and last
        np.subtract(t_safe[1:2], t_safe[0:1], out=grad[0:1, :, :, 0])
        np.multiply(grad[0:1, :, :, 0], rdx, out=grad[0:1, :, :, 0])
        np.subtract(t_safe[-1:], t_safe[-2:-1], out=grad[-1:, :, :, 0])
        np.multiply(grad[-1:, :, :, 0], rdx, out=grad[-1:, :, :, 0])

    if ny > 1:
        if ny > 2:
            np.subtract(t_safe[:, 2:], t_safe[:, :-2], out=grad[:, 1:-1, :, 1])
            np.multiply(grad[:, 1:-1, :, 1], half_rdx, out=grad[:, 1:-1, :, 1])
        np.subtract(t_safe[:, 1:2], t_safe[:, 0:1], out=grad[:, 0:1, :, 1])
        np.multiply(grad[:, 0:1, :, 1], rdx, out=grad[:, 0:1, :, 1])
        np.subtract(t_safe[:, -1:], t_safe[:, -2:-1], out=grad[:, -1:, :, 1])
        np.multiply(grad[:, -1:, :, 1], rdx, out=grad[:, -1:, :, 1])

    if nz > 1:
        if nz > 2:
            np.subtract(t_safe[:, :, 2:], t_safe[:, :, :-2], out=grad[:, :, 1:-1, 2])
            np.multiply(grad[:, :, 1:-1, 2], half_rdx, out=grad[:, :, 1:-1, 2])
        np.subtract(t_safe[:, :, 1:2], t_safe[:, :, 0:1], out=grad[:, :, 0:1, 2])
        np.multiply(grad[:, :, 0:1, 2], rdx, out=grad[:, :, 0:1, 2])
        np.subtract(t_safe[:, :, -1:], t_safe[:, :, -2:-1], out=grad[:, :, -1:, 2])
        np.multiply(grad[:, :, -1:, 2], rdx, out=grad[:, :, -1:, 2])


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


def _hotspot_cluster_threshold(
    dominant_m_mm: float,
    bbox_size_mm: np.ndarray,
    dx: float,
) -> float:
    """Return a scale-aware clustering distance for hot spots.

    The threshold is anchored to the dominant local modulus (a hot-spot cloud
    is typically a few moduli wide) and to a small percentage of the bounding
    box so that very long/thin parts are not over-clustered.  It is bounded
    below by a few voxels and above by 15 % of the largest box dimension.
    """
    bbox_max = float(np.max(bbox_size_mm))
    cluster_mm = max(
        2.5 * float(dominant_m_mm),
        0.05 * bbox_max,
        5.0 * float(dx),
    )
    return float(min(cluster_mm, 0.15 * bbox_max))


def _sample_field_at_position(
    position_mm: np.ndarray,
    field: np.ndarray,
    origin_mm: np.ndarray,
    dx: float,
    order: int = 1,
    default: float = 0.0,
) -> float:
    """Interpolate a scalar field at an arbitrary physical position.

    Replaces coarse-voxel rounding / _snap_to_part for scalar reads so that
    adjacent hot spots no longer inherit the same voxel value.
    """
    vox = (np.asarray(position_mm, dtype=np.float64) - np.asarray(origin_mm, dtype=np.float64)) / float(dx)
    coords = vox.reshape(3, 1)
    sampled = ndimage.map_coordinates(
        field,
        coords,
        order=order,
        mode="constant",
        cval=default,
        prefilter=False,
    )
    # map_coordinates with a (3, 1) coordinate array returns a 1-element
    # ndarray; .flat[0] works for both 0-d and 1-d returns.
    return float(np.asarray(sampled).flat[0])


def find_hotspots(
    sdf: np.ndarray,
    part_mask: np.ndarray,
    dx: float,
    origin_mm: np.ndarray,
    curvature: Optional[np.ndarray] = None,
    gaussian_curvature: Optional[np.ndarray] = None,
    use_skeleton: bool = True,
    min_size_mm: float = 2.0,
    cluster_eps_mm: float = 10.0,
    riser_mask: Optional[np.ndarray] = None,
    is_metal: Optional[np.ndarray] = None,
    feeder_mask: Optional[np.ndarray] = None,
    chvorinov_c: Optional[float] = None,
    n_time_steps: int = 40,
    niyama: Optional[np.ndarray] = None,
    feeder_time_factor: float = 1.0,
    merge_clusters: bool = True,
) -> List[HotSpot]:
    """Detect hot spots by pseudo-thermal solidification + CCL (Method 2).

    The metal is solidified in Chvorinov time layers.  At each layer, the
    remaining liquid metal is labelled with 26-connectivity.  Liquid pockets
    that are not connected to a feeder (riser / gating) are isolated; the last
    points to become isolated are the true hot spots.  A feeder/riser neck
    naturally solidifies earlier and breaks the connection, so the region under
    a riser is not reported as a part hot spot.
    """
    # Shape-corrected modulus (Steiner: M = SDF / (1 - 2H*SDF + K*SDF^2))
    if curvature is not None:
        gauss = gaussian_curvature
        if gauss is None:
            _, gauss = compute_curvature(sdf, dx)
        M_mod = compute_steiner_modulus(sdf, curvature, gauss, clip_min=0.5)
    else:
        M_mod = sdf.copy()

    if is_metal is None:
        is_metal = part_mask
    if feeder_mask is None:
        feeder_mask = riser_mask if riser_mask is not None else np.zeros_like(is_metal)
    if chvorinov_c is None or chvorinov_c <= 0:
        raise ValueError(
            "find_hotspots requires a positive chvorinov_c. "
            "Compute it with chvorinov_c_from_properties(alloy, mold)."
        )

    # Solidification time from shape-corrected modulus (Chvorinov)
    t_solid = compute_chvorinov_t(M_mod, chvorinov_c)
    t_solid = np.nan_to_num(t_solid, nan=0.0, posinf=0.0, neginf=0.0)

    # Riser thermal attraction: a feeder is a heat reservoir, so it solidifies
    # later and pulls the last-freezing region toward itself.  The feeder's own
    # solidification time is boosted by the sleeve/insulation factor and a
    # Gaussian potential is added to neighbouring metal so the hot spot physically
    # migrates toward the riser.
    if feeder_mask is not None and feeder_mask.any() and is_metal is not None:
        feeder_cells = feeder_mask & is_metal
        if feeder_cells.any():
            t_solid[feeder_cells] *= max(1.0, float(feeder_time_factor))
            dist_to_feeder = ndimage.distance_transform_edt(is_metal & ~feeder_cells) * dx
            r_feeder = max(float(np.mean(M_mod[feeder_cells])), 2.0 * dx)
            feeder_late = float(np.percentile(t_solid[feeder_cells], 99.0))
            with np.errstate(over="ignore", under="ignore"):
                attraction = feeder_late * np.exp(-(dist_to_feeder / max(r_feeder, 1e-6)) ** 2)
            t_solid = t_solid + attraction

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
        # fallback: pick the most critical voxel within the candidate mask
        # (highest isolation time, then highest modulus).  Avoids selecting a
        # voxel outside the intended region when no regional maximum is found.
        cand = np.argwhere(candidate_mask)
        if len(cand) == 0:
            return []
        vals = isolation_time[candidate_mask]
        m_vals = M_mod[candidate_mask]
        best_idx = int(np.argmax(vals * 1000.0 + m_vals))
        pos_vox = cand[best_idx]
        m_value = float(M_mod[pos_vox[0], pos_vox[1], pos_vox[2]])
        return [
            HotSpot(
                position_mm=origin_mm + pos_vox * dx,
                m_value_mm=m_value,
                dist_to_riser_mm=np.inf,
                feed_ok=False,
                max_feeding_distance_mm=0.0,
                niyama_ensemble=float(niyama[pos_vox[0], pos_vox[1], pos_vox[2]]) if niyama is not None else 0.0,
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
        # Use raw SDF (local half wall thickness) for modulus scoring and the
        # displayed hotspot modulus. M_mod can be inflated by the Steiner clip.
        m_vals = sdf[cand[:, 0], cand[:, 1], cand[:, 2]]
        # Pick the most critical voxel: late isolating, high modulus, low Niyama.
        iso_score = vals / max(max_iso, 1e-9)
        m_score = m_vals / max(float(m_vals.max()), 1e-9)
        if niyama is not None:
            n_vals = niyama[cand[:, 0], cand[:, 1], cand[:, 2]]
            n_max = max(float(n_vals.max()), 1e-9)
            n_score = n_vals / n_max
        else:
            n_score = np.ones_like(iso_score)
        best_idx = int(np.argmax(iso_score + 0.5 * m_score - 0.5 * n_score))
        pos_vox = cand[best_idx]
        m_value = float(sdf[pos_vox[0], pos_vox[1], pos_vox[2]])
        position_mm = origin_mm + pos_vox * dx
        hotspots.append(
            HotSpot(
                position_mm=position_mm,
                m_value_mm=m_value,
                dist_to_riser_mm=np.inf,
                feed_ok=False,
                max_feeding_distance_mm=0.0,
                niyama_ensemble=float(niyama[pos_vox[0], pos_vox[1], pos_vox[2]]) if niyama is not None else 0.0,
            )
        )

    # Final cleanup: merge hot spots that are close enough to be fed by one riser.
    if merge_clusters and len(hotspots) > 1:
        hotspots = _merge_hotspots(hotspots, cluster_eps_mm, prefer_unresolved=False)

    return hotspots


def _merge_hotspots(
    hotspots: List[HotSpot],
    cluster_eps_mm: float,
    prefer_unresolved: bool = True,
) -> List[HotSpot]:
    """Merge close hot spots and keep the most critical representative.

    When ``prefer_unresolved`` is True and a cluster contains at least one
    unsolved hot spot, the representative is chosen from the unsolved subset so
    that a dangerous (unfed) hot spot is not hidden behind a solved neighbour.
    The representative is then the one with the largest modulus and the lowest
    Niyama (highest shrinkage risk).
    """
    if len(hotspots) <= 1:
        return hotspots
    positions = np.array([hs.position_mm for hs in hotspots], dtype=np.float64)
    clustering = DBSCAN(eps=cluster_eps_mm, min_samples=1, metric="euclidean").fit(
        positions
    )
    merged: List[HotSpot] = []
    for lbl in set(clustering.labels_):
        group = [hs for i, hs in enumerate(hotspots) if clustering.labels_[i] == lbl]
        has_unresolved = any(not h.solved for h in group)

        def _priority(h: HotSpot) -> Tuple[float, float, float]:
            # Prefer unresolved when requested; then larger modulus; then lower Niyama.
            solved_penalty = float(h.solved)
            if not prefer_unresolved or not has_unresolved:
                solved_penalty = 0.0
            return (solved_penalty, -float(h.m_value_mm), float(h.niyama_ensemble))

        group.sort(key=_priority)
        merged.append(group[0])
    return merged


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

    metal_vox = np.argwhere(is_metal)
    n = int(metal_vox.shape[0])
    idx_dtype = np.int32 if n <= np.iinfo(np.int32).max else np.int64
    idx = np.full(is_metal.shape, -1, dtype=idx_dtype)
    idx[tuple(metal_vox.T)] = np.arange(n, dtype=idx_dtype)

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
        rows.append(source_idx[valid].astype(idx_dtype, copy=False))
        cols.append(neighbor_idx[valid].astype(idx_dtype, copy=False))
        vals.append(step.astype(np.float32))

    riser_flat = np.where(riser_mask[tuple(metal_vox.T)])[0]
    if len(riser_flat) == 0:
        return dist
    rows.append(np.full(len(riser_flat), n, dtype=idx_dtype))
    cols.append(riser_flat.astype(idx_dtype, copy=False))
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
    # Predecessor values are flat voxel indices; int32 is enough for grids up to
    # 2^31 voxels and halves memory on large industrial meshes.
    flat_idx_dtype = np.int32 if is_metal.size <= np.iinfo(np.int32).max else np.int64
    pred = np.full(is_metal.shape, -1, dtype=flat_idx_dtype)
    if not (is_metal & riser_mask).any():
        return cost, pred, np.empty((0, 3), dtype=np.int64)

    metal_vox = np.argwhere(is_metal)
    n = int(metal_vox.shape[0])
    idx_dtype = np.int32 if n <= np.iinfo(np.int32).max else np.int64
    idx = np.full(is_metal.shape, -1, dtype=idx_dtype)
    idx[tuple(metal_vox.T)] = np.arange(n, dtype=idx_dtype)

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
        rows.append(source_idx[valid].astype(idx_dtype, copy=False))
        cols.append(neighbor_idx[valid].astype(idx_dtype, copy=False))
        vals.append(step.astype(np.float32))

    riser_flat = np.where(riser_mask[tuple(metal_vox.T)])[0]
    if len(riser_flat) == 0:
        return cost, pred, metal_vox
    rows.append(np.full(len(riser_flat), n, dtype=idx_dtype))
    cols.append(riser_flat.astype(idx_dtype, copy=False))
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
    fp = flat_pred[:n].astype(idx_dtype, copy=False)
    pred_values = np.full(n, -1, dtype=flat_idx_dtype)
    valid = (fp >= 0) & (fp < n)
    if valid.any():
        pv = metal_vox[fp[valid], 0] * is_metal.shape[1] * is_metal.shape[2]
        pv += metal_vox[fp[valid], 1] * is_metal.shape[2]
        pv += metal_vox[fp[valid], 2]
        pred_values[valid] = pv.astype(flat_idx_dtype, copy=False)
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
        for di, dj, dk in NEIGH_26:
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
    gravity_vector: Tuple[float, float, float] = (0.0, 0.0, -1.0),
    bodies: Optional[List[Body]] = None,
    body_index: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, bool, bool, float, bool, float]:
    """
    Walk the lowest-resistance feeding path from start_vox to a riser and compute:
      * Darcy pressure drop through the mushy zone (Kozeny-Carman)
      * minimum neck modulus along the part path
      * t_s at the hot spot
      * directional solidification flag (using actual 3-D solidification times)
      * Heuver's circle flag
      * total feeding cost
      * darcy_ok flag
      * feedable_fraction of the shrinkage demand that the hydrostatic head can drive

    When ``bodies`` and ``body_index`` are supplied, the nearest riser body is
    identified and its ``feeder_type`` is used to cap the available hydrostatic
    head (blind feeders have no open top; side feeders have reduced head).
    """
    path = _trace_cost_path(start_vox, cost_pred, sdf.shape)
    if not path:
        return 0.0, 0.0, 0.0, True, True, 0.0, True, 1.0

    m_hot = float(M_mod[start_vox[0], start_vox[1], start_vox[2]])
    # Use the actual 3-D transient solidification time if available
    if t_sol is not None and np.isfinite(t_sol[start_vox[0], start_vox[1], start_vox[2]]):
        t_s_hot = float(t_sol[start_vox[0], start_vox[1], start_vox[2]])
    else:
        C = chvorinov_c_from_properties(alloy, mold)
        t_s_hot = float(compute_chvorinov_t(np.array([m_hot], dtype=float), C)[0])

    # Hydrostatic head from feeders above the hot spot (opposite to gravity).
    # Use the feeder voxel centroid instead of the highest voxel so a long/thin
    # riser does not artificially inflate the available metal head.
    P_head = 0.0
    if feeder_voxels is not None and len(feeder_voxels) > 0:
        g = np.asarray(gravity_vector, dtype=np.float64)
        g_norm = float(np.linalg.norm(g)) + 1e-12
        g = g / g_norm
        feeder_centroid_mm = np.mean(feeder_voxels, axis=0) * dx
        hot_pos_mm = np.asarray(start_vox, dtype=np.float64) * dx
        diff = feeder_centroid_mm - hot_pos_mm
        # Positive projection means the feeder centroid is in the -g direction
        # (above) relative to the hot spot.
        head_mm = float(np.dot(diff, -g))
        if head_mm > 0.0:
            P_head = alloy.rho_kg_m3 * 9.81 * (head_mm / 1000.0)

        # v10.1: account for feeder geometry/type. Blind feeders have no open
        # top, so the available head is capped to the metal height inside the
        # feeder multiplied by a type-specific efficiency. Side feeders are
        # horizontal and get a reduced head.
        head_efficiency = 1.0
        max_head_m = float("inf")
        if (
            body_index is not None
            and body_index.shape == sdf.shape
            and bodies is not None
            and len(feeder_voxels) > 0
        ):
            # Identify the riser body nearest to the hot spot.
            distances = np.linalg.norm(feeder_voxels - start_vox, axis=1)
            nearest_idx = int(np.argmin(distances))
            nearest_vox = tuple(int(v) for v in feeder_voxels[nearest_idx])
            bidx = int(body_index[nearest_vox])
            if 0 <= bidx < len(bodies):
                feeder_body = bodies[bidx]
                ftype = (feeder_body.feeder_type or "conventional").lower().strip()
                if not ftype:
                    ftype = "conventional"
                if ftype == "blind":
                    head_efficiency = 0.3
                    # The metal column in a blind riser cannot exceed the
                    # feeder height projected onto the gravity-opposite axis.
                    bounds = feeder_body.mesh.bounds  # (2, 3) in mm
                    extent_mm = bounds[1] - bounds[0]
                    feeder_height_mm = float(np.dot(extent_mm, -g))
                    max_head_m = max(feeder_height_mm, 0.0) / 1000.0
                elif ftype == "side":
                    head_efficiency = 0.7
                elif ftype in ("exothermic", "insulated", "sleeve"):
                    head_efficiency = 1.0
                else:
                    head_efficiency = 1.0
        P_head = P_head * head_efficiency
        if max_head_m < float("inf"):
            P_head = min(P_head, max(alloy.rho_kg_m3 * 9.81 * max_head_m, 0.0))

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
    # Fraction of the required shrinkage flow that the available hydrostatic head
    # can actually drive through the mushy-zone resistance.  <1 means the rest becomes
    # a macro shrinkage cavity.
    if feed_stopped or P_head <= 0.0:
        feedable_fraction = 0.0
    else:
        feedable_fraction = float(min(1.0, P_head / max(darcy, 1e-9)))

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
    # Non-finite values (not reached in the thermal horizon) are treated as
    # very late so they do not produce artificial drops.
    directional_ok = True
    if has_thermal and len(part_path) > 4:
        t_path = np.array([float(t_sol[v[0], v[1], v[2]]) for v in part_path])
        t_path = np.nan_to_num(t_path, nan=1e6, posinf=1e6, neginf=0.0)
        tol = max(0.05 * t_s_hot, 1.0)
        if np.any(np.diff(t_path[1:]) < -tol):
            directional_ok = False

    feeding_cost = float(cost_grid[start_vox[0], start_vox[1], start_vox[2]])
    return darcy, min_neck_m, t_s_hot, directional_ok, heuvers_ok, feeding_cost, darcy_ok, feedable_fraction


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
    """Average SDF (modulus) of part voxels 26-neighbouring an ingate."""
    ingate = grid == BodyType.INGATE
    if not ingate.any():
        return 0.0
    # 26-neighbour dilation marks every cavity cell that touches an ingate voxel
    # through a face, edge or corner.  border_value=0 prevents wrap-around.
    dilated = ndimage.binary_dilation(
        ingate,
        structure=np.ones((3, 3, 3), dtype=bool),
        border_value=False,
    )
    touch = dilated & part_mask
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
    # Local refinement is intentionally small; do not auto-refine past the
    # caller's local memory budget.
    grid, _, origin, dx, _ = build_voxel_grid(
        cropped_bodies,
        target_dim=target_dim,
        progress_callback=progress_callback,
        conservative=False,
        max_dim=max_local_dim,
        auto_refine=False,
    )
    is_metal = np.isin(grid, BODY_METAL_TYPES)
    sdf = compute_sdf(is_metal, dx)
    C = chvorinov_c_from_properties(alloy, mold)
    mean_curv, gauss_curv = compute_curvature(sdf, dx)
    M_mod = compute_steiner_modulus(sdf, mean_curv, gauss_curv, clip_min=0.5)
    t_s = compute_chvorinov_t(M_mod, C)
    T, R, fs, _ = compute_thermal_field(
        grid, is_metal, alloy, mold, dx, sdf=sdf, M_mod=M_mod
    )
    thermal_stress, hot_tear_risk, cold_crack_risk = compute_thermal_stress(
        T, fs, alloy, is_metal
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
    alloy: Alloy,
    mold: MoldMaterial,
    cluster_eps_mm: Optional[float] = None,
    min_size_mm: Optional[float] = None,
    progress_callback: Optional[callable] = None,
) -> Optional[List[HotSpot]]:
    """Build a high-resolution grid containing the PART and connected casting-metal
    bodies (gating/riser) and detect hot spots.

    Including gating/riser geometry in the local high-resolution grid lets the
    pseudo-thermal CCL see the real metal connectivity, so gate/riser-fed part
    pockets are not reported as isolated hot spots.
    """
    try:
        part_grid, _, part_origin, part_dx, _ = build_part_grid(
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
    mean_curv, gauss_curv = compute_curvature(part_sdf, part_dx)
    part_M_mod = compute_steiner_modulus(part_sdf, mean_curv, gauss_curv, clip_min=0.5)

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
    hotspot_min_size_mm = (
        min_size_mm
        if min_size_mm is not None and min_size_mm > 0.0
        else min(2.0, max(0.5, 0.5 * max_part_sdf))
    )
    # Use the scale-aware caller-supplied cluster distance if available.
    if cluster_eps_mm is None:
        cluster_eps_mm = max(12.0, 2.0 * part_dx)

    feeder_time_factor = _weighted_feeder_time_factor(bodies)

    if progress_callback:
        progress_callback(83)

    # High-resolution Niyama so the representative point reflects the real field.
    _, _, part_niyama = compute_niyama(
        part_sdf, part_M_mod, alloy, mold, part_dx, is_metal=part_is_metal
    )

    part_hotspots = find_hotspots(
        part_sdf,
        part_mask,
        part_dx,
        part_origin,
        curvature=mean_curv,
        use_skeleton=True,
        min_size_mm=hotspot_min_size_mm,
        cluster_eps_mm=cluster_eps_mm,
        is_metal=part_is_metal,
        feeder_mask=part_feeder_mask,
        chvorinov_c=chvorinov_c,
        niyama=part_niyama,
        feeder_time_factor=feeder_time_factor,
        merge_clusters=False,
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
        "side": 1.0,
        "blind": 1.0,
    }.get(ftype, 1.0)


def _apply_feeder_sleeve_time_factor(
    grid: np.ndarray,
    t_liq: np.ndarray,
    t_sol: np.ndarray,
    bodies: List[Body],
    body_index: Optional[np.ndarray] = None,
) -> None:
    """Scale solidification times of RISER voxels by the per-body feeder type.

    If ``body_index`` is supplied, each RISER voxel is matched to its owning
    body and multiplied by that body's sleeve factor.  This prevents an
    exothermic and a blind feeder in the same model from being merged into a
    single volume-weighted average.
    """
    riser_bodies = [b for b in bodies if b.body_type == BodyType.RISER]
    if not riser_bodies:
        return
    riser_mask = grid == BodyType.RISER
    if not riser_mask.any():
        return

    if body_index is None or body_index.shape != grid.shape:
        # Fallback to the old volume-weighted average.
        total_volume = sum(max(b.volume_cm3, 1e-9) for b in riser_bodies)
        if total_volume <= 0.0:
            return
        factor = sum(
            _feeder_time_factor(b) * max(b.volume_cm3, 1e-9) / total_volume
            for b in riser_bodies
        )
        finite_liq = riser_mask & np.isfinite(t_liq)
        finite_sol = riser_mask & np.isfinite(t_sol)
        t_liq[finite_liq] = t_liq[finite_liq] * factor
        t_sol[finite_sol] = t_sol[finite_sol] * factor
        return

    # Body-index aware: apply the exact factor for each riser body's voxels.
    for b in riser_bodies:
        factor = _feeder_time_factor(b)
        if factor == 1.0:
            continue
        bmask = riser_mask & (body_index == b.index)
        if not bmask.any():
            continue
        finite_liq = bmask & np.isfinite(t_liq)
        finite_sol = bmask & np.isfinite(t_sol)
        if finite_liq.any():
            t_liq[finite_liq] = t_liq[finite_liq] * factor
        if finite_sol.any():
            t_sol[finite_sol] = t_sol[finite_sol] * factor


def _weighted_feeder_time_factor(bodies: List[Body]) -> float:
    """Volume-weighted solidification-time multiplier for all riser bodies."""
    riser_bodies = [b for b in bodies if b.body_type == BodyType.RISER]
    if not riser_bodies:
        return 1.0
    total_volume = sum(max(b.volume_cm3, 1e-9) for b in riser_bodies)
    if total_volume <= 0.0:
        return 1.0
    return sum(
        _feeder_time_factor(b) * max(b.volume_cm3, 1e-9) / total_volume
        for b in riser_bodies
    )


def _run_filling_flow(
    gate,
    casting_params: Optional[CastingParameters],
    user_section_areas_cm2: Optional[Dict[str, float]],
    grid: np.ndarray,
    origin_mm: np.ndarray,
    dx_mm: float,
    alloy,
    mold,
    bodies: List[Body],
    body_index: Optional[np.ndarray],
):
    """Run the 3-D Darcy filling-flow solver using the gate design/user inputs."""
    from core.filling_solver import solve_filling_flow, cad_source_area_m2, GatingVelocityError

    design_section_key = (
        getattr(casting_params, "velocity_section_key", None) or "SPRUE_THROAT"
    )
    user_v = float(getattr(casting_params, "ingate_velocity_m_s", 0.0) or 0.0)
    design_v = user_v if user_v > 0.0 else float(
        getattr(gate, "design_choke_velocity_m_s", 0.0) or 0.0
    )
    user_area_cm2 = (
        user_section_areas_cm2.get(design_section_key, 0.0)
        if user_section_areas_cm2
        else 0.0
    )
    if user_area_cm2 > 0.0:
        design_area_cm2 = float(user_area_cm2)
    else:
        g_vec = getattr(casting_params, "gravity_vector", (0.0, 0.0, -1.0))
        cad_area_m2 = cad_source_area_m2(bodies, design_section_key, g_vec)
        if cad_area_m2 > 1e-12:
            design_area_cm2 = float(cad_area_m2 * 1e4)
        else:
            raise GatingVelocityError(
                f"{design_section_key} kesit alanı CAD geometrisinden hesaplanamadı. "
                f"Lütfen elle kesit alanı girin veya geometriyi kontrol edin."
            )
    sprue_throat_cm2 = float(gate.sprue_throat_area_cm2) if gate.sprue_throat_area_cm2 else 0.0
    sprue_base_cm2 = float(gate.sprue_base_area_cm2) if gate.sprue_base_area_cm2 else 0.0
    if sprue_throat_cm2 <= 0.0 and "SPRUE_THROAT" in (gate.section_flows or {}):
        sprue_throat_cm2 = float(gate.section_flows["SPRUE_THROAT"].area_cm2)
    if sprue_base_cm2 <= 0.0 and "SPRUE_BASE" in (gate.section_flows or {}):
        sprue_base_cm2 = float(gate.section_flows["SPRUE_BASE"].area_cm2)
    section_areas_m2 = {
        "SPRUE_THROAT": sprue_throat_cm2 * 1e-4,
        "SPRUE_BASE": sprue_base_cm2 * 1e-4,
        "RUNNER": float(gate.runner_min_area_cm2) * 1e-4 if gate.runner_min_area_cm2 else 0.0,
        "INGATE": float(gate.total_ingate_contact_area_cm2) * 1e-4 if gate.total_ingate_contact_area_cm2 else 0.0,
        "DISTRIBUTOR": float(gate.distributor_area_cm2) * 1e-4 if gate.distributor_area_cm2 else 0.0,
        "CURUFLUK": float(gate.curufluk_area_cm2) * 1e-4 if gate.curufluk_area_cm2 else 0.0,
    }
    # Ensure the selected section area in the flow solver matches the area used
    # to compute Q_user; otherwise the first node velocity will not equal the
    # user-entered sprue velocity.
    if design_area_cm2 > 0.0:
        section_areas_m2[design_section_key] = float(design_area_cm2) * 1e-4
    # Allow the user to override any measured section area (cm2 -> m2).
    if user_section_areas_cm2:
        for key, val in user_section_areas_cm2.items():
            if val and val > 0.0:
                section_areas_m2[key.upper()] = float(val) * 1e-4
    fast_hydraulic = bool(getattr(casting_params, "fast_flow", False))
    return solve_filling_flow(
        grid,
        origin_mm,
        dx_mm,
        casting_params,
        alloy,
        bodies=bodies,
        body_index=body_index,
        progress_callback=None,
        design_velocity_m_s=design_v,
        design_section_key=design_section_key,
        design_area_m2=design_area_cm2 * 1e-4,
        section_areas_m2=section_areas_m2,
        mold=mold,
        fast_hydraulic=fast_hydraulic,
    )


def analyze(
    bodies: List[Body],
    grid: np.ndarray,
    body_index: Optional[np.ndarray],
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
    user_section_areas_cm2: Optional[Dict[str, float]] = None,
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
        mold = make_effective_mold(mold, casting_params=casting_params)
        mold = replace(mold, t0_c=casting_params.t_mold_c)
    chvorinov_c = chvorinov_c_from_properties(alloy, mold)
    gravity_vector = (
        getattr(casting_params, "gravity_vector", None)
        or getattr(casting_params, "gravity_direction", None)
        or (0.0, 0.0, -1.0)
        if casting_params is not None
        else (0.0, 0.0, -1.0)
    )
    g_unit = np.asarray(gravity_vector, dtype=np.float64)
    g_norm = float(np.linalg.norm(g_unit))
    if g_norm > 1e-12:
        g_unit = g_unit / g_norm
    else:
        g_unit = np.array([0.0, 0.0, -1.0])
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
    chill_mask = np.isin(grid, CHILL_BODY_TYPES)

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
    # Steiner shape-corrected modulus: M = SDF / (1 - 2H*SDF + K*SDF^2).
    # The SDF Laplacian is 2*H and the Hessian determinant is K, so the
    # formula reduces to 1 - mean_curv*SDF + gauss_curv*SDF^2.
    M_mod = compute_steiner_modulus(sdf, mean_curv, gauss_curv, clip_min=0.5)
    # Porozite/Niyama hesapları için eski davranışı koruyan ayrı modül.
    # Hotspot/t_section/besleme M_mod'u (clip 0.5) ile karışmaması gerekir.
    M_mod_porosity = compute_steiner_modulus(sdf, mean_curv, gauss_curv, clip_min=0.1)
    # Debug: voxels where the Steiner shape factor is very low indicate
    # discretised/non-manifold geometry (sharp edges, thin triangles) rather
    # than a real thick section. Log a few coordinates for inspection.
    _sf_check = 1.0 - mean_curv * sdf + gauss_curv * (sdf ** 2)
    _bad_sf = (_sf_check < 0.5) & part_mask
    if _bad_sf.any():
        _bad_count = int(_bad_sf.sum())
        _bad_min = float(_sf_check[_bad_sf].min())
        _bad_vox = np.argwhere(_bad_sf)
        _n_sample = min(3, _bad_vox.shape[0])
        _bad_mm = origin_mm + _bad_vox[:_n_sample] * dx
        print(
            f"[ANALYZE] UYARI: {_bad_count} vokselde shape_factor < 0.5 "
            f"(min {_bad_min:.3f}); örnek koordinatlar (mm): {_bad_mm.tolist()}",
            flush=True,
        )
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

    # AŞAMA 2.5: Gating / 3-D Darcy flow (needed for fill_time before thermal).
    gate_result_for_flow = None
    flow_result_for_thermal = None
    print(f"[ANALYZE] 28% -> gating analysis start ({time.strftime('%H:%M:%S')})", flush=True)
    try:
        from core.gating import analyze_gating

        tmp_result = SimpleNamespace(
            grid=grid,
            origin_mm=origin_mm,
            dx_mm=dx,
            is_metal=is_metal,
            sdf=sdf,
            subvoxel_sdf=sdf,
            part_volume_mm3=part_volume_mm3,
            part_surface_area_mm2=part_surface_area_mm2,
            wall_thickness_mm=wall_thickness,
            dominant_m_mm=dominant_m,
            bbox_size_mm=bbox_size,
            alloy_key=alloy_key,
            mold_key=mold_key,
            hotspots=[],
            riser_results=[],
            risk=np.zeros_like(grid, dtype=float),
            recommendations=[],
        )
        gate_result_for_flow = analyze_gating(
            tmp_result,
            casting_params=casting_params,
            bodies=bodies,
            user_section_areas_cm2=user_section_areas_cm2,
        )
        print(f"[ANALYZE] gating done ({time.strftime('%H:%M:%S')})", flush=True)
    except Exception as exc:
        if isinstance(exc, GatingVelocityError):
            raise
        print(f"[ANALYZE] gating failed: {exc}", flush=True)
        pass

    if gate_result_for_flow is not None:
        print(f"[ANALYZE] filling flow start ({time.strftime('%H:%M:%S')})", flush=True)
        try:
            flow_result_for_thermal = _run_filling_flow(
                gate_result_for_flow,
                casting_params,
                user_section_areas_cm2,
                grid,
                origin_mm,
                dx,
                alloy,
                mold,
                bodies,
                body_index,
            )
        except Exception as exc:
            if isinstance(exc, GatingVelocityError):
                raise
            import traceback
            print("[Darcy exception]", exc)
            traceback.print_exc()
            pass
        else:
            print(f"[ANALYZE] filling flow done ({time.strftime('%H:%M:%S')})", flush=True)

    # AŞAMA 3: Full 3-D transient enthalpy thermal solver (downsampled for speed)
    if progress_callback:
        progress_callback(30)
        progress_callback(31)
    fill_time_s = (
        flow_result_for_thermal.fill_time
        if flow_result_for_thermal is not None and flow_result_for_thermal.fill_time is not None
        else None
    )
    velocity_m_s = (
        flow_result_for_thermal.velocity
        if flow_result_for_thermal is not None and flow_result_for_thermal.velocity is not None
        else None
    )
    # v10.6: air entrapment from LBM/VOF free-surface tracking.
    air_entrapment_field = np.zeros_like(grid, dtype=np.float64)
    air_pressure_pa_field = np.zeros_like(grid, dtype=np.float64)
    air_density_kg_m3_field = np.zeros_like(grid, dtype=np.float64)
    trapped_air_volume_m3 = 0.0
    air_entrapment_centroid_mm = np.array([], dtype=np.float64)
    if (
        flow_result_for_thermal is not None
        and flow_result_for_thermal.air_entrapment is not None
        and flow_result_for_thermal.air_entrapment.size == grid.size
    ):
        air_entrapment_field = np.asarray(flow_result_for_thermal.air_entrapment, dtype=np.float64)
        trapped_air_volume_m3 = float(getattr(flow_result_for_thermal, "trapped_air_volume_m3", 0.0))
        air_entrapment_centroid_mm = np.asarray(
            getattr(flow_result_for_thermal, "air_entrapment_centroid_mm", np.array([])),
            dtype=np.float64,
        )
        if getattr(flow_result_for_thermal, "air_pressure_pa", None) is not None:
            air_pressure_pa_field = np.asarray(
                flow_result_for_thermal.air_pressure_pa, dtype=np.float64
            ).reshape(grid.shape)
        if getattr(flow_result_for_thermal, "air_density_kg_m3", None) is not None:
            air_density_kg_m3_field = np.asarray(
                flow_result_for_thermal.air_density_kg_m3, dtype=np.float64
            ).reshape(grid.shape)

    # Geometric trapped-air fallback / complement: runs even when the 3-D LBM
    # solver is disabled or misses closed pockets under overhangs.
    use_geometric = (
        flow_result_for_thermal is None
        or flow_result_for_thermal.air_entrapment is None
        or air_entrapment_field.size != grid.size
        or float(air_entrapment_field.max()) < 0.05
    )
    if use_geometric:
        from core.filling_solver import compute_air_entrapment_geofc

        gating_nodes = getattr(flow_result_for_thermal, "gating_nodes", None)
        geo_fill_time_s = float(getattr(flow_result_for_thermal, "fill_time_s", 0.0) or 0.0)
        Q_m3_s = float(getattr(flow_result_for_thermal, "Q_m3_s", 0.0) or 0.0)

        geo_risk, geo_vol, geo_cent = compute_air_entrapment_geofc(
            grid,
            origin_mm,
            dx,
            gravity_vector=tuple(g_unit),
            mold=mold,
            casting_params=casting_params,
            bodies=bodies,
            body_index=body_index,
            gating_nodes=gating_nodes,
            fill_time_s=geo_fill_time_s,
            Q_m3_s=Q_m3_s,
            alloy=alloy,
            max_cells=150_000,
        )
        if geo_risk.size == grid.size:
            if air_entrapment_field.size != grid.size:
                air_entrapment_field = np.asarray(geo_risk, dtype=np.float64)
            else:
                air_entrapment_field = np.maximum(air_entrapment_field, geo_risk)
            trapped_air_volume_m3 = max(trapped_air_volume_m3, float(geo_vol))
            if geo_cent.size == 3:
                air_entrapment_centroid_mm = geo_cent

    temperature, solid_fraction, t_liq, t_s, G, cooling_rate, niyama = solve_3d_thermal(
        grid, alloy, mold, dx,
        max_time_s=thermal_max_time_s,
        downsample=thermal_downsample,
        progress_callback=progress_callback,
        fill_time_s=fill_time_s,
        velocity_m_s=velocity_m_s,
        gravity_vector=tuple(
            getattr(casting_params, "gravity_vector", None)
            or getattr(casting_params, "gravity_direction", None)
            or (0.0, 0.0, -1.0)
        )
        if casting_params is not None
        else (0.0, 0.0, -1.0),
    )
    thermal_stress, hot_tear_risk, cold_crack_risk = compute_thermal_stress(
        temperature, solid_fraction, alloy, is_metal
    )
    # v9.3: account for feeder sleeves/exothermic/chilled type by scaling the
    # solidification time of RISER voxels.  With the body-index grid, each
    # riser body receives its own sleeve/exothermic/chilled factor instead of
    # a single volume-weighted average.
    _apply_feeder_sleeve_time_factor(grid, t_liq, t_s, bodies, body_index)

    # Fallback for thick regions that did not reach solidus within max_time_s:
    # use the analytical Chvorinov/Stefan Niyama so hot spots are not reported as 0.
    G_ana, R_ana, niyama_ana = compute_niyama(
        sdf, M_mod_porosity, alloy, mold, dx, is_metal=is_metal
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
    user_min_size = (
        casting_params.hotspot_min_size_mm
        if casting_params is not None and casting_params.hotspot_min_size_mm > 0.0
        else 0.0
    )
    user_cluster = (
        casting_params.hotspot_cluster_eps_mm
        if casting_params is not None and casting_params.hotspot_cluster_eps_mm > 0.0
        else 0.0
    )
    hotspot_min_size_mm = (
        user_min_size
        if user_min_size > 0.0
        else min(2.0, max(0.5, 0.5 * max_part_sdf))
    )
    hotspot_cluster_mm = (
        user_cluster
        if user_cluster > 0.0
        else _hotspot_cluster_threshold(dominant_m, bbox_size, dx)
    )
    feeder_time_factor = _weighted_feeder_time_factor(bodies)
    hotspots = find_hotspots(
        sdf, part_mask, dx, origin_mm, curvature=mean_curv, gaussian_curvature=gauss_curv,
        use_skeleton=True,
        min_size_mm=hotspot_min_size_mm,
        cluster_eps_mm=hotspot_cluster_mm,
        is_metal=is_metal,
        feeder_mask=feeder_mask,
        chvorinov_c=chvorinov_c,
        merge_clusters=False,
        niyama=niyama,
        feeder_time_factor=feeder_time_factor,
    )
    if progress_callback:
        progress_callback(75)

    # AŞAMA 6: 26-neighbor Dijkstra feeding distance and lowest-resistance cost path
    dist_feed = feeding_distance_dijkstra(is_metal, feeder_mask, dx, gravity_vector=gravity_vector)
    cost_feed, cost_pred, _ = feeding_cost_dijkstra(
        is_metal, feeder_mask, M_mod, dx, gravity_vector=gravity_vector
    )
    # Distance to nearest chill insert for P3 filtering.
    if chill_mask.any():
        dist_chill_vox = ndimage.distance_transform_edt(~chill_mask)
    else:
        dist_chill_vox = None
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
            alloy,
            mold,
            cluster_eps_mm=hotspot_cluster_mm,
            min_size_mm=hotspot_min_size_mm,
            progress_callback=progress_callback,
        )
        if part_hotspots is not None and part_hotspots:
            hotspots = part_hotspots

    # v9.4: detect hot spots inside risers/feeders separately.  They are not part
    # defects, but they are useful for riser sizing and verification.
    feeder_hotspots: List[HotSpot] = []
    if riser_mask is not None and riser_mask.any():
        # Disable feeder-touch suppression so pockets inside feeders are reported.
        no_suppress = np.zeros_like(riser_mask)
        feeder_hotspots = find_hotspots(
            sdf,
            riser_mask,
            dx,
            origin_mm,
            curvature=mean_curv,
            gaussian_curvature=gauss_curv,
            use_skeleton=True,
            min_size_mm=hotspot_min_size_mm,
            cluster_eps_mm=hotspot_cluster_mm,
            is_metal=is_metal,
            feeder_mask=no_suppress,
            chvorinov_c=chvorinov_c,
            niyama=niyama,
        )
        for fhs in feeder_hotspots:
            fhs.feed_ok = True
            fhs.directional_ok = True
            fhs.heuvers_ok = True
            fhs.darcy_ok = True
            fhs.dist_to_riser_mm = 0.0

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

    # AŞAMA 7: Hot-spot physics
    feeder_voxels = np.argwhere(feeder_mask)
    # Hydraulic feeding factor: 1.0 everywhere, locally amplified where the
    # hydrostatic head cannot overcome the Darcy resistance along the feeding path.
    darcy_factor = np.ones(grid.shape, dtype=np.float64)

    for hs in hotspots:
        vox_raw = np.round((hs.position_mm - origin_mm) / dx).astype(int)
        vox = _snap_to_part(vox_raw)
        if 0 <= vox[0] < grid.shape[0] and 0 <= vox[1] < grid.shape[1] and 0 <= vox[2] < grid.shape[2]:
            # Scalar fields are sampled at the hot-spot's exact physical position
            # instead of snapping to the nearest coarse voxel, eliminating the
            # duplicated-value problem for adjacent hot spots.
            hs.dist_to_riser_mm = _sample_field_at_position(
                hs.position_mm, dist_feed, origin_mm, dx, order=1, default=np.inf
            )
            hs.niyama_min = _sample_field_at_position(
                hs.position_mm, niyama, origin_mm, dx, order=1, default=0.0
            )
            hs.niyama_variants = {
                k: _sample_field_at_position(
                    hs.position_mm, v, origin_mm, dx, order=1, default=0.0
                )
                for k, v in niyama_variants.items()
            }
            hs.niyama_ensemble = _sample_field_at_position(
                hs.position_mm, niyama, origin_mm, dx, order=1, default=0.0
            )
            hs.local_sdf_max = _sample_field_at_position(
                hs.position_mm, sdf, origin_mm, dx, order=1, default=0.0
            )
            hs.m_uncertainty_mm = dx / 2.0
            hs.feeding_cost = _sample_field_at_position(
                hs.position_mm, cost_feed, origin_mm, dx, order=1, default=0.0
            )

            darcy, min_neck_m, t_hs, directional_ok, heuvers_ok, feeding_cost, darcy_ok, feedable_fraction = _path_darcy_and_directional(
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
                gravity_vector=gravity_vector,
                bodies=bodies,
                body_index=body_index,
            )
            hs.darcy_resistance = darcy
            hs.feedable_fraction = feedable_fraction
            hs.min_neck_m_mm = min_neck_m
            hs.directional_ok = directional_ok
            hs.heuvers_ok = heuvers_ok
            hs.darcy_ok = darcy_ok

            # Where the pressure head cannot drive the shrinkage demand through the
            # mushy-zone resistance, the pore volume grows.  Amplify darcy_factor
            # around this hot spot with a Gaussian falloff so the amplification is
            # strongest at the hot-spot centre and vanishes within about one local
            # modulus.  A high local Niyama value suppresses the amplification, because
            # a strong thermal gradient can feed shrinkage even when the pressure
            # head is marginal.
            if feedable_fraction < 1.0:
                local_factor = 1.0 / max(float(feedable_fraction), 0.1)
                # Shrink affected radius to one local modulus; decay so 90% of the
                # extra amplification is within ~0.4 M.
                radius_vox = max(1.0 * hs.m_value_mm / dx, 3.0)
                centre = np.array(vox, dtype=np.float64)
                zz, yy, xx = np.indices(grid.shape, dtype=np.float64)
                dist2 = (
                    (zz - centre[0]) ** 2
                    + (yy - centre[1]) ** 2
                    + (xx - centre[2]) ** 2
                )
                falloff = np.exp(-(24.0 * dist2) / (radius_vox ** 2 + 1e-9))

                # Niyama damping: at N >= niyama_macro the feeding gradient alone
                # is sufficient, so Darcy amplification is zero.  At lower N it ramps
                # up linearly.
                with np.errstate(divide="ignore", invalid="ignore"):
                    niyama_ratio = niyama / max(float(alloy.niyama_macro), 1e-9)
                niyama_ratio = np.nan_to_num(niyama_ratio, nan=0.0, posinf=0.0, neginf=0.0)
                niyama_penalty = np.clip(1.0 - niyama_ratio, 0.0, 1.0)
                # Only act on metal/part voxels; valid mask in compute_pore_size
                # will ignore the rest, but keep the array clean anyway.
                niyama_penalty = np.where(part_mask, niyama_penalty, 0.0)

                darcy_factor = np.maximum(
                    darcy_factor,
                    1.0 + (local_factor - 1.0) * falloff * niyama_penalty,
                )

            hs.curvature_mean = _sample_field_at_position(
                hs.position_mm, mean_curv, origin_mm, dx, order=1, default=0.0
            )
            hs.curvature_gaussian = _sample_field_at_position(
                hs.position_mm, gauss_curv, origin_mm, dx, order=1, default=0.0
            )

            # Section thickness = 2 * local modulus (equivalent wall thickness)
            hs.t_section_mm = 2.0 * hs.m_value_mm
            hs.width_mm = 2.0 * hs.local_sdf_max
            hs.shape_factor = _shape_factor(
                _sphere_mask(part_mask.shape, vox, max(3.0 * hs.m_value_mm / dx, 5.0)) & part_mask,
                dx,
            )

            if len(feeder_voxels) > 0:
                feeder_positions_mm = feeder_voxels * dx + origin_mm
                diff_to_feeder = feeder_positions_mm - hs.position_mm
                # Project the feeder-to-hot-spot vector onto the gravity-opposite
                # direction.  A feeder above the hot spot increases effective
                # feeding distance; a feeder at the same level or below gives no
                # bonus (max(0, ...)).
                gravity_proj = np.dot(diff_to_feeder, -g_unit)
                closest_feeder_idx = int(
                    np.argmin(np.linalg.norm(diff_to_feeder, axis=1))
                )
                proj_closest = gravity_proj[closest_feeder_idx]
                hs.gravity_factor = 1.0 + 0.3 * max(
                    0.0, proj_closest / max(hs.dist_to_riser_mm, 1.0)
                )
            else:
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
            # A chill solves the hot spot if it is close enough to the local modulus.
            if dist_chill_vox is not None:
                d_chill = _sample_field_at_position(
                    hs.position_mm, dist_chill_vox, origin_mm, dx, order=1, default=np.inf
                ) * dx
                hs.chill_ok = d_chill <= 1.5 * hs.m_value_mm
            else:
                hs.chill_ok = False
        else:
            hs.feed_ok = False
            hs.chill_ok = False

    # Re-merge hot spots now that feed_ok / chill_ok are known so an unresolved
    # (dangerous) hot spot is never hidden behind a solved neighbour in the UI.
    if hotspots:
        hotspots = _merge_hotspots(hotspots, hotspot_cluster_mm, prefer_unresolved=True)

    if progress_callback:
        progress_callback(88)

    # AŞAMA 8: Riser sufficiency with resistance-corrected modulus transfer
    riser_results: List[RiserResult] = []
    labeled, num = ndimage.label(riser_mask)
    # v9.1: per-voxel riser-size factor for effective feeding distance.
    # ID 0 means "no riser" and keeps the default factor 1.0.
    riser_factor_map = np.zeros(grid.shape, dtype=np.int32)
    factor_by_id = [1.0]
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

        # Prefer the watertight mesh volume/area when available, but fall back
        # to exact exposed-voxel face counting for non-watertight solids.
        if body.volume_cm3 > 0.0:
            volume_mm3 = float(body.volume_cm3) * 1000.0
        else:
            volume_mm3 = voxel_count * (dx ** 3)
        volume_cm3 = volume_mm3 / 1000.0

        surface_mm2 = _exposed_surface_area_mm2(component_mask, grid, dx)
        if surface_mm2 <= 0.0 and body.surface_area_cm2 > 0.0:
            surface_mm2 = float(body.surface_area_cm2) * 100.0
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
            "side": 1.0,
            "blind": 1.0,
        }.get(feeder_type, 1.0)
        # Volume yield: exothermic mini-risers supply the same modulus with less metal.
        feeder_volume_yield = {
            "conventional": 1.0,
            "exothermic": getattr(alloy, "exothermic_volume_yield", 0.45),
            "insulated": 0.8,
            "sleeve": 0.9,
            "chilled": 1.2,
            "side": 1.0,
            "blind": 1.0,
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
        # gravity_proj > 0  -> riser centroid is ABOVE the hotspot (g points down)
        # gravity_proj < 0  -> riser centroid is BELOW the hotspot.
        # Above => smaller required modulus (gravity assists feeding);
        # below => larger required modulus.  Coefficient 0.002 / mm means
        # ~20% modulus change per 100 mm vertical offset, capped between 0.6 and
        # 3.0 to keep engineering results sane while still penalising an
        # upside-down feeder much harder than the old 0.85 floor.
        riser_centroid_mm = riser_centroid_vox * dx + origin_mm
        if nearest_hs is not None:
            gravity_proj = float(np.dot(nearest_pos - riser_centroid_mm, g_unit))
        else:
            gravity_proj = 0.0
        gravity = float(np.clip(1.0 - 0.002 * gravity_proj, 0.6, 3.0))
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

    # Feeding risk: 0 at the feeder, -> 1 far beyond the effective feeding distance.
    with np.errstate(divide="ignore", invalid="ignore"):
        FD_field = alloy.feed_k1 * (2.0 * M_mod) * riser_factor_field
        feed_risk = dist_feed / (dist_feed + np.maximum(FD_field, 1.0))
        feed_risk = np.clip(np.nan_to_num(feed_risk, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)

    # v8.8: estimate pore size from the Carlson-Beckermann dimensionless Niyama model.
    velocity_magnitude = (
        flow_result_for_thermal.velocity_magnitude
        if flow_result_for_thermal is not None and flow_result_for_thermal.velocity_magnitude is not None
        else None
    )
    (
        pore_size_um,
        pore_size_mm,
        pore_macro_mask,
        pore_micro_mask,
        pore_fine_mask,
        pore_shrinkage_um,
        pore_volume_pct,
        mold_wall_movement,
    ) = compute_pore_size(
        niyama,
        M_mod_porosity,
        feed_risk,
        alloy,
        part_mask,
        t_s=t_s,
        feeder_mask=feeder_mask,
        dx=dx,
        gravity_vector=gravity_vector,
        fill_time=fill_time_s,
        darcy_factor=darcy_factor,
        velocity_magnitude=velocity_magnitude,
        solid_fraction=solid_fraction,
        mold=mold,
    )

    # v10.4: per-voxel cold-shut (soğuk birleşme) risk and the last fill point.
    # V8: local mould-material Chvorinov and effusivity fields.
    C_field = build_local_chvorinov_c_field(grid, body_index, bodies, mold, alloy)
    e_field = build_effusivity_field(grid, body_index, bodies, mold)

    # V8: meeting enthalpy field from Chvorinov T(t) + Scheil fs (direct, no LUT).
    if fill_time_s is not None and fill_time_s.size:
        H_field, T_meet, fs_meet = compute_H_field(
            M_mod, C_field, fill_time_s, alloy, t_pour_c=alloy.t_pour_c, t_mold_c=mold.t0_c
        )
    else:
        H_field = np.zeros_like(M_mod)
        T_meet = np.full_like(M_mod, alloy.t_pour_c)
        fs_meet = np.zeros_like(M_mod)

    (
        cold_shot_risk,
        lap_risk,
        cold_shot_saddles,
        last_fill_point_mm,
        cold_shot_risk_viz,
        lap_risk_viz,
        cold_shot_lines,
    ) = compute_cold_shot_risk(
        part_mask,
        fill_time_s,
        velocity_magnitude,
        temperature,
        t_s,
        M_mod,
        alloy,
        t_pour_c=alloy.t_pour_c,
        t_mold_c=mold.t0_c,
        dx=dx,
        origin_mm=origin_mm,
        t_liq=t_liq,
        mold=mold,
        feeder_mask=feeder_mask,
        feed_risk=feed_risk,
        velocity_m_s=velocity_m_s,
        sdf=sdf,
        curvature_mean=mean_curv,
        curvature_gauss=gauss_curv,
        C_field=C_field,
        e_field=e_field,
        H_field=H_field,
        dist_feed=dist_feed,
        grid=grid,
        body_index=body_index,
    )

    # v10.5: per-voxel mold-sand erosion risk from local metal velocity.
    erosion_risk = compute_erosion_risk(
        velocity_magnitude,
        is_metal,
        alloy,
        mold,
    )

    # AŞAMA 9: Risk map aligned with the Carlson-Beckermann porosity volume.
    # The predicted pore volume percentage is already reduced by feeding
    # efficiency; convert it to a 0-1 risk field using the macro class limit
    # (gp at macro limit -> ~63 % risk).
    gp_ref = alloy.macro_pore_limit_um / max(alloy.pore_size_um_per_porosity_pct, 1e-9)
    risk = 1.0 - np.exp(-np.clip(pore_volume_pct / max(gp_ref, 1e-9), 0.0, 50.0))
    # v8.6: risk belongs to the part only; risers/gating/chills are not part porosity.
    risk = np.where(part_mask, risk, 0.0)
    risk = np.nan_to_num(risk, nan=0.0, posinf=0.0, neginf=0.0)
    risk_norm = risk
    # Class thresholds are taken directly from the alloy's physical micron
    # limits; no empirical top-percent filters are used.
    pore_macro_threshold_um = float(alloy.macro_pore_limit_um)
    pore_micro_threshold_um = float(alloy.micro_pore_limit_um)
    pore_fine_threshold_um = 0.0
    pore_threshold_um = pore_macro_threshold_um
    pore_macro_percent = 0.0
    pore_micro_percent = 0.0
    pore_fine_percent = 0.0

    # Assign pore-size estimate to each hot spot and re-evaluate feed_ok with
    # the riser-size-modulated feeding distance.
    for hs in hotspots:
        # Sample pore-size and nearest-riser factor at the exact hot-spot
        # location instead of the coarse voxel centre.
        ps_um = _sample_field_at_position(
            hs.position_mm, pore_size_um, origin_mm, dx, order=1, default=0.0
        )
        hs.pore_size_um = ps_um
        hs.pore_size_mm = ps_um / 1000.0
        hs.pore_size_class = _pore_size_class(
            ps_um,
            macro_threshold_um=alloy.macro_pore_limit_um,
            micro_threshold_um=alloy.micro_pore_limit_um,
        )
        # v9.1: effective feeding distance depends on the nearest riser size.
        factor = _sample_field_at_position(
            hs.position_mm, riser_factor_field, origin_mm, dx, order=0, default=1.0
        )
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
        feeder_hotspots=feeder_hotspots,
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
        pore_size_shrinkage_um=pore_shrinkage_um,
        pore_size_shrinkage_mm=pore_shrinkage_um / 1000.0,
        pore_size_macro_mask=pore_macro_mask,
        pore_size_micro_mask=pore_micro_mask,
        pore_size_fine_mask=pore_fine_mask,
        mold_wall_movement=mold_wall_movement,
        cold_shot_risk=cold_shot_risk,
        lap_risk=lap_risk,
        cold_shot_risk_viz=cold_shot_risk_viz,
        lap_risk_viz=lap_risk_viz,
        cold_shot_saddles=cold_shot_saddles,
        cold_shot_lines=cold_shot_lines,
        last_fill_point_mm=last_fill_point_mm,
        H_field=H_field,
        T_meet=T_meet,
        fs_meet=fs_meet,
        erosion_risk=erosion_risk,
        air_entrapment=air_entrapment_field,
        trapped_air_volume_m3=trapped_air_volume_m3,
        air_entrapment_centroid_mm=air_entrapment_centroid_mm,
        air_pressure_pa=air_pressure_pa_field,
        air_density_kg_m3=air_density_kg_m3_field,
        pore_size_noise_percent=pore_macro_percent,
        pore_size_threshold_um=pore_macro_threshold_um,
        pore_size_macro_percent=pore_macro_percent,
        pore_size_macro_threshold_um=pore_macro_threshold_um,
        pore_size_micro_percent=pore_micro_percent,
        pore_size_micro_threshold_um=pore_micro_threshold_um,
        pore_size_fine_percent=pore_fine_percent,
        pore_size_fine_threshold_um=pore_fine_threshold_um,
        thermal_stress_pa=thermal_stress,
        hot_tear_risk=hot_tear_risk,
        cold_crack_risk=cold_crack_risk,
        fill_time_s=fill_time_s,
    )

    result.riser_proposals = propose_risers(
        result, alloy, existing_riser_count=len(riser_results), gravity_vector=gravity_vector
    )

    result.flow_result = flow_result_for_thermal

    # AŞAMA 11: Gating result (flow already solved in stage 2.5 for fill_time).
    try:
        from core.gating import analyze_gating
        result.gate_result = analyze_gating(
            result,
            casting_params=casting_params,
            bodies=bodies,
            user_section_areas_cm2=user_section_areas_cm2,
        )
    except Exception as exc:
        if isinstance(exc, GatingVelocityError):
            raise
        result.gate_result = gate_result_for_flow
        result.recommendations.append(f"Gating analizi atlandı: {exc}")

    if result.gate_result and result.flow_result:
        result.gate_result.flow_result = result.flow_result

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

    M_cm = result.dominant_m_mm / 10.0
    t_solid_s = (
        result.chvorinov_c * (M_cm ** 2) * 60.0
        if result.chvorinov_c and result.dominant_m_mm > 0.0
        else 0.0
    )
    M_geo_cm = result.geometric_m_cm
    t_solid_geo_s = (
        result.chvorinov_c * (M_geo_cm ** 2) * 60.0
        if result.chvorinov_c and M_geo_cm > 0.0
        else 0.0
    )
    recs.append(
        f"Malzeme: {alloy.name} | Kalıp: {mold.name} | Chvorinov C = {result.chvorinov_c:.4f} dk/cm² | "
        f"Baskın M = {M_cm:.2f} cm (t_s ≈ {t_solid_s:.1f} s / {t_solid_s/60.0:.2f} dk) | "
        f"Geometrik M (V/A) = {M_geo_cm:.2f} cm (t_s ≈ {t_solid_geo_s:.1f} s / {t_solid_geo_s/60.0:.2f} dk) | "
        f"Duvar kalınlığı t_wall ≈ {result.wall_thickness_mm:.2f} mm | "
        f"Şekil faktörü SF = {result.shape_factor_global:.6f}"
    )
    recs.append(
        f"Modül istatistikleri: ortalama M = {result.m_mean_mm/10.0:.2f} cm, std = {result.m_std_mm/10.0:.2f} cm, "
        f"çarpıklık = {result.m_skewness:.2f}. "
        + ("Parça duvar kalınlığı dengesiz." if abs(result.m_skewness) > 1.0 else "Kalınlık dağılımı nispeten dengeli.")
    )

    if not result.hotspots:
        recs.append(
            "Kritik sıcak nokta (hot spot) tespit edilmedi. Model çok ince veya geometri düzgün okunamamış olabilir."
        )
        return recs

    for idx, hs in enumerate(result.hotspots, 1):
        pos = ",".join(f"{v:.1f}" for v in hs.position_mm)
        issues: List[str] = []
        if hs.dist_to_riser_mm > hs.max_feeding_distance_mm:
            issues.append("besleme mesafesi yetersiz")
        if not hs.directional_ok:
            issues.append("yönlü katılaşma bozuk")
        if not hs.heuvers_ok:
            issues.append("Heuver ihlali")
        if not hs.darcy_ok:
            if hs.darcy_resistance < 0.01:
                issues.append("Darcy: katılaşmış yol tıkalı")
            else:
                issues.append(f"Darcy basınç kaybı ({hs.darcy_resistance:.2f} Pa)")
        niy = hs.niyama_ensemble
        if niy < alloy.niyama_macro:
            issues.append(f"Niyama {niy:.2f} < {alloy.niyama_macro} (makro shrinkage riski)")
        elif niy < alloy.niyama_shrinkage:
            issues.append(f"Niyama {niy:.2f} < {alloy.niyama_shrinkage} (mikro gözenek riski)")
        status = "; ".join(issues) if issues else "riskli ama beslenebilir"

        if hs.chill_ok:
            suggestion = "çıkıcı (chill) konumlandırın"
        elif hs.feed_ok and hs.darcy_ok:
            suggestion = "besleyici menzili içinde; besleyici boyun/hacim kontrolü yapın"
        else:
            suggestion = "mini ekzotermik besleyici veya çıkıcı (chill) kullanın; yolu kısaltın, kesiti büyütün veya geçiş yarıçapını büyütün"

        pore_extra = ""
        if hs.pore_size_class and hs.pore_size_um > 0:
            pore_extra = f" | Gözenek tahmini: {hs.pore_size_um:.1f} µm ({hs.pore_size_class})"

        recs.append(
            f"Hata Bölgesi #{idx} ({pos} mm): Kritik Hotspot. "
            f"M={hs.m_value_mm/10.0:.2f} cm, Niyama={niy:.2f}. "
            f"Durum: {status}. "
            f"Öneri: {suggestion}.{pore_extra}"
        )

    if result.feeder_hotspots:
        recs.append("Besleyici/Riser içi sıcak noktalar (parça hatası değil, referans için):")
        for idx, fhs in enumerate(result.feeder_hotspots, 1):
            pos = ",".join(f"{v:.1f}" for v in fhs.position_mm)
            recs.append(
                f"  Riser Bölgesi #{idx} ({pos} mm): M={fhs.m_value_mm/10.0:.2f} cm, "
                f"Niyama={fhs.niyama_ensemble:.2f}. Besleyici içindedir."
            )

    for rr in result.riser_results:
        eff_m = max(rr.effective_m_value_mm, rr.m_value_mm)
        type_text = f" ({rr.feeder_type})" if rr.feeder_type else ""
        if not rr.large_enough:
            increase = (
                (rr.effective_m_required / max(eff_m, 1e-6) - 1.0) * 100.0
            )
            recs.append(
                f"{rr.name}{type_text}: M_besleyici={eff_m / 10.0:.2f} cm (gerçek {rr.m_value_mm / 10.0:.2f} cm) < gerekli {rr.effective_m_required / 10.0:.2f} cm. "
                f"Besleyici modülünü %{int(increase)} büyütün."
            )
        if not rr.volume_ratio_ok:
            short = rr.required_volume_cm3 - rr.volume_cm3
            recs.append(
                f"{rr.name}{type_text}: hacim yetersiz (V={rr.volume_cm3:.2f} cm³, gerekli {rr.required_volume_cm3:.2f} cm³). "
                f"En az {short:.2f} cm³ daha hacim ekleyin."
            )

    for idx, proposal in enumerate(result.riser_proposals):
        pos = f"({proposal.placement_mm[0] / 10.0:.1f}, {proposal.placement_mm[1] / 10.0:.1f}, {proposal.placement_mm[2] / 10.0:.1f})"
        if proposal.infeasible:
            recs.append(
                f"UYARI {idx + 1}: Hotspot #{proposal.target_hotspot_index + 1} için önerilen "
                f"besleyici/çıkıcı parça geometrisine sığmıyor. "
                f"Mini exotermik besleyici veya çıkıcı (chill) önerilir; konum {pos} cm. "
                f"{proposal.warning if proposal.warning else 'Çözüm kullanıcı kararıdır.'}"
            )
        elif proposal.shape == "chill":
            recs.append(
                f"ÖNERİ {idx + 1}: çıkıcı (chill) ekle -> "
                f"çap={proposal.diameter_mm / 10.0:.1f} cm, yükseklik={proposal.height_mm / 10.0:.1f} cm, "
                f"V={proposal.volume_cm3:.2f} cm³. "
                f"Konum {pos} cm. Neden: {proposal.reason}."
            )
        elif proposal.exothermic:
            recs.append(
                f"ÖNERİ {idx + 1}: ekzotermik mini besleyici ekle -> "
                f"çap={proposal.diameter_mm / 10.0:.1f} cm, yükseklik={proposal.height_mm / 10.0:.1f} cm, "
                f"V={proposal.volume_cm3:.2f} cm³. "
                f"Konum {pos} cm. Neden: {proposal.reason}."
            )
        else:
            recs.append(
                f"ÖNERİ {idx + 1}: konvansiyonel silindirik besleyici ekle -> "
                f"çap={proposal.diameter_mm / 10.0:.1f} cm, yükseklik={proposal.height_mm / 10.0:.1f} cm, "
                f"V={proposal.volume_cm3:.2f} cm³, M={proposal.m_required_mm / 10.0:.2f} cm. "
                f"Konum {pos} cm. Neden: {proposal.reason}."
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
