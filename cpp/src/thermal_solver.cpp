/* Enthalpy-based 3-D solidification solver + Carlson-Beckermann porosity.
 *
 * Solves  rho * cp_eff * dT/dt = div(k grad T) - rho*cp*(u . grad T)
 * with cp_eff boosted by the latent-heat Scheil derivative in the mushy zone.
 * Liquidus/solidus crossing times are recorded and the Niyama criterion
 * (G / sqrt(R)) is returned for porosity post-processing.
 */

#include "josecast/thermal_solver.h"
#include "josecast/arena.hpp"

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <memory_resource>
#include <queue>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nb = nanobind;

namespace josecast {

namespace {

inline size_t cidx(int x, int y, int z, int ny, int nz) {
    return static_cast<size_t>((x * ny + y) * nz + z);
}

double getd(const std::map<std::string, double> &m, const std::string &key, double def = 0.0) {
    auto it = m.find(key);
    return (it == m.end()) ? def : it->second;
}

// Try a 3x over-allocation arena first; if the OS cannot back that much
// virtual address / page-file space, fall back to smaller multipliers.  This
// keeps the solver from failing on 100M+ voxel grids when 3x would exceed the
// process commit limit, while still preferring the requested headroom.
std::unique_ptr<VirtualArena> make_arena_fallback(size_t n, const std::vector<std::size_t>& multipliers) {
    for (std::size_t m : multipliers) {
        try {
            return std::make_unique<VirtualArena>(VirtualArena::recommended(n, m));
        } catch (const std::bad_alloc&) {
            continue;
        }
    }
    throw std::bad_alloc();
}

// Scheil solid fraction [0..1]
double scheil_fs(double T, double Tl, double Ts, double k) {
    if (T >= Tl) return 0.0;
    if (T <= Ts) return 1.0;
    double denom = Tl - Ts;
    if (denom <= 0.0) return 0.0;
    double v = (T - Ts) / denom;
    v = std::max(0.0, std::min(1.0, v));
    double p = 1.0 / std::max(1e-6, 1.0 - k);
    return 1.0 - std::pow(v, p);
}

// d(fs)/dT for apparent heat capacity boost
double dscheil_dT(double T, double Tl, double Ts, double k) {
    if (T <= Ts || T >= Tl) return 0.0;
    double denom = Tl - Ts;
    if (denom <= 0.0) return 0.0;
    double v = (T - Ts) / denom;
    v = std::max(1e-9, std::min(1.0 - 1e-9, v));
    double p = 1.0 / std::max(1e-6, 1.0 - k);
    return p * std::pow(v, p - 1.0) / denom;
}

double carlson_gp(double ny_star, double b0, const std::string &key) {
    double safe = std::max(ny_star, 1e-12);
    double gp = 0.0;
    if (key == "A356") {
        if (safe <= 1.43)
            gp = -2.068 * std::log10(safe) + 3.160;
        else if (safe <= 18.0)
            gp = 4.024 * std::pow(safe, -0.9786);
        else
            gp = 7.771 * std::pow(safe, -1.206);
    } else if (key == "AZ91D") {
        if (safe <= 41.0)
            gp = -1.671 * std::log10(safe) + 3.483;
        else if (safe <= 45.2)
            gp = -10.81 * std::log10(safe) + 18.23;
        else
            gp = 73.01 * std::pow(safe, -1.415);
    } else { // WCB default
        if (safe <= 28.2)
            gp = -1.654 * std::log10(safe) + 3.052;
        else
            gp = 43.05 * std::pow(safe, -1.254);
    }
    return std::max(0.0, std::min(b0, gp));
}

} // namespace

nb::tuple solve_thermal(
    nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, -1>> is_metal,
    nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, -1>> is_gating,
    nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, -1>> is_chill,
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> fill_time,
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1, -1>> velocity,
    double dx_mm,
    double max_time_s,
    int n_steps,
    std::map<std::string, double> alloy,
    std::map<std::string, double> mold,
    double feed_velocity_m_s,
    std::array<double, 3> gravity_vector)
{
    int nx = static_cast<int>(is_metal.shape(0));
    int ny = static_cast<int>(is_metal.shape(1));
    int nz = static_cast<int>(is_metal.shape(2));
    size_t n = static_cast<size_t>(nx) * ny * nz;
    const double dx_m = dx_mm / 1000.0;

    // One virtual-address arena for all large per-voxel working arrays.
    // 250 bytes/voxel covers T, k, rho, cp0, T_new/T_adv/T_tmp, t_liq, t_sol,
    // G/R, fs_final, cp_eff, metal/gating/chill/mold_layer masks, full_to_int,
    // and boundary/C/b buffers.  Reserve 3x to leave headroom for temporaries.
    auto arena_ptr = make_arena_fallback(n, {80, 60, 45, 30, 20});
    VirtualArena* ar = arena_ptr.get();

    // ---- alloy / mould properties ----
    const double Tl = getd(alloy, "t_liquidus_c", 1500.0);
    const double Ts = getd(alloy, "t_solidus_c", 1400.0);
    const double Tp = getd(alloy, "t_pour_c", 1600.0);
    const double dT_mush = Tl - Ts;
    const double L = getd(alloy, "latent_heat_j_kg", 0.0);
    const double k_part = getd(alloy, "partition_coefficient", 0.5);

    const double metal_k = getd(alloy, "k_w_mk", 45.0);
    const double metal_cp = getd(alloy, "cp_j_kgk", 460.0);
    const double metal_rho = getd(alloy, "rho_kg_m3", 7850.0);

    const double mold_k = getd(mold, "k_w_mk", 0.58);
    const double mold_cp = getd(mold, "cp_j_kgk", 1170.0);
    const double mold_rho = getd(mold, "rho_kg_m3", 1600.0);
    const double T0 = getd(mold, "t0_c", 25.0);

    // Green-sand / chemically bonded sand property modifiers.
    const double afs = getd(mold, "afs_grain_size", 50.0);
    const double moisture = getd(mold, "moisture_percent", 4.0);
    const double binder = getd(mold, "binder_percent", 2.0);
    const double compactability = getd(mold, "compactability_percent", 45.0);
    const bool is_sand_mold = (moisture > 0.0 || afs > 0.0 || getd(mold, "is_sand", 0.0) > 0.5);

    // Empirical corrections: finer sand and more moisture/binder reduce k;
    // moisture/binder increase effective heat capacity.
    double k_bulk = mold_k;
    double cp_bulk = mold_cp;
    if (is_sand_mold) {
        k_bulk *= std::max(0.3, 1.0 - 0.0015 * (afs - 50.0))
                 * std::max(0.5, 1.0 - 0.015 * moisture)
                 * std::max(0.5, 1.0 - 0.01 * binder)
                 * std::max(0.5, 1.0 + 0.003 * (compactability - 45.0));
        cp_bulk *= 1.0 + 0.025 * moisture + 0.005 * binder
                   - 0.0002 * std::max(0.0, afs - 50.0)
                   + 0.0003 * std::max(0.0, compactability - 45.0);
    }
    // Extra heat capacity in the first 3 mould layers due to water vapourisation.
    double cp_surface = cp_bulk * (1.0 + 0.04 * std::max(0.0, moisture));
    double k_surface = k_bulk * std::max(0.5, 1.0 - 0.005 * moisture);

    // Chill properties: explicit keys allow a different material (e.g. copper
    // chill) from the bulk mould.  If no explicit keys are supplied, fall back
    // to the mould conductivity when it looks like a metal chill (k > 100),
    // otherwise use a steel chill default.
    const double chill_k = getd(mold, "chill_k_w_mk",
                                getd(mold, "chill_k",
                                      (mold_k > 100.0 ? mold_k : 45.0)));
    const double chill_cp = getd(mold, "chill_cp_j_kgk",
                                 getd(mold, "chill_cp",
                                       (mold_k > 100.0 ? mold_cp : 460.0)));
    const double chill_rho = getd(mold, "chill_rho_kg_m3",
                                  getd(mold, "chill_rho",
                                        (mold_k > 100.0 ? mold_rho : 7850.0)));

    // ---- normalized gravity ----
    double gx = gravity_vector[0];
    double gy = gravity_vector[1];
    double gz = gravity_vector[2];
    double gnorm = std::sqrt(gx * gx + gy * gy + gz * gz);
    if (gnorm < 1e-9) { gx = 0.0; gy = 0.0; gz = -1.0; gnorm = 1.0; }
    gx /= gnorm; gy /= gnorm; gz /= gnorm;

    // ---- masks and material fields ----
    const uint8_t *metal_ptr = is_metal.data();
    const uint8_t *gating_ptr = is_gating.data();
    const uint8_t *chill_ptr = is_chill.data();

    std::pmr::vector<double> T(n, T0, ar), k(n, k_bulk, ar), rho(n, mold_rho, ar), cp0(n, cp_bulk, ar);
    std::pmr::vector<uint8_t> metal(n, 0, ar), gating(n, 0, ar), chill(n, 0, ar);
    std::pmr::vector<int> mold_layer(n, -1, ar);
    std::queue<size_t> layer_q;
    for (size_t i = 0; i < n; ++i) {
        metal[i] = metal_ptr[i];
        gating[i] = gating_ptr[i];
        chill[i] = chill_ptr[i];
        double cp_i = cp_bulk, k_i = k_bulk, r_i = mold_rho;
        if (metal[i]) { cp_i = metal_cp; k_i = metal_k; r_i = metal_rho; }
        if (chill[i]) { cp_i = chill_cp; k_i = chill_k; r_i = chill_rho; }
        T[i] = metal[i] ? Tp : T0;
        k[i] = k_i;
        rho[i] = r_i;
        cp0[i] = cp_i;
        if (metal[i] || chill[i]) {
            mold_layer[i] = 0;
            layer_q.push(i);
        }
    }

    // 6-neighbour BFS to identify the first 3 mould layers next to metal/chill.
    if (is_sand_mold) {
        auto linear_to_ijk = [&](size_t idx, int& x, int& y, int& z) {
            z = static_cast<int>(idx % nz);
            size_t r = idx / nz;
            y = static_cast<int>(r % ny);
            x = static_cast<int>(r / ny);
        };
        while (!layer_q.empty()) {
            size_t i = layer_q.front(); layer_q.pop();
            int x, y, z;
            linear_to_ijk(i, x, y, z);
            int d = mold_layer[i];
            if (d >= 3) continue;
            const int dxs[6] = {1, -1, 0, 0, 0, 0};
            const int dys[6] = {0, 0, 1, -1, 0, 0};
            const int dzs[6] = {0, 0, 0, 0, 1, -1};
            for (int dir = 0; dir < 6; ++dir) {
                int nx2 = x + dxs[dir], ny2 = y + dys[dir], nz2 = z + dzs[dir];
                if (nx2 < 0 || nx2 >= nx || ny2 < 0 || ny2 >= ny || nz2 < 0 || nz2 >= nz) continue;
                size_t j = cidx(nx2, ny2, nz2, ny, nz);
                if (mold_layer[j] >= 0 || metal[j] || chill[j]) continue;
                mold_layer[j] = d + 1;
                layer_q.push(j);
            }
        }
        for (size_t i = 0; i < n; ++i) {
            if (mold_layer[i] > 0 && mold_layer[i] <= 3 && !metal[i] && !chill[i]) {
                k[i] = k_surface;
                cp0[i] = cp_surface;
            }
        }
    }

    // ---- fill time and velocity ----
    bool has_fill = (fill_time.size() == n);
    const double *fill_ptr = has_fill ? fill_time.data() : nullptr;
    double fill_end = 0.0;
    if (has_fill) {
        for (size_t i = 0; i < n; ++i) {
            if (std::isfinite(fill_ptr[i]) && fill_ptr[i] > fill_end)
                fill_end = fill_ptr[i];
        }
    }
    if (fill_end > 0.0) max_time_s += fill_end;

    bool has_velocity = false;
    const double *vel_ptr = nullptr;
    if (velocity.size() > 0 && velocity.ndim() == 4) {
        if (static_cast<int>(velocity.shape(0)) == 3 &&
            static_cast<int>(velocity.shape(1)) == nx &&
            static_cast<int>(velocity.shape(2)) == ny &&
            static_cast<int>(velocity.shape(3)) == nz) {
            has_velocity = true;
            vel_ptr = velocity.data();
        }
    }

    double vmax = 0.0;
    if (has_velocity) {
        for (size_t i = 0; i < n; ++i) {
            double vx = vel_ptr[i];
            double vy = vel_ptr[n + i];
            double vz = vel_ptr[2 * n + i];
            double vm = std::sqrt(vx * vx + vy * vy + vz * vz);
            if (vm > vmax) vmax = vm;
        }
    }
    const double dt_adv = (vmax > 1e-6) ? (0.3 * dx_m / vmax)
                                        : std::numeric_limits<double>::infinity();
    const double dt_feed = (feed_velocity_m_s > 1e-9) ? (0.3 * dx_m / feed_velocity_m_s)
                                                        : std::numeric_limits<double>::infinity();

    // ---- time step ----
    if (n_steps <= 0) {
        n_steps = std::max(20, std::min(100, static_cast<int>(max_time_s / 3.0)));
        // Alloys with a very narrow mushy zone (< 10C) need a smaller
        // diffusion time-step to avoid cp_eff spikes in the latent-heat region.
        if (dT_mush < 10.0) n_steps *= 2;
    }
    double dt_diff = max_time_s / std::max(1, n_steps);
    const int max_adv_subcycles = 200;

    // ---- precompute diffusion operator on the interior ----
    const int stride_x = ny * nz;
    const int stride_y = nz;
    std::pmr::vector<int> full_to_int(n, -1, ar);
    std::pmr::vector<int> int_to_full(ar);
    int n_int = 0;
    for (int x = 1; x < nx - 1; ++x)
        for (int y = 1; y < ny - 1; ++y)
            for (int z = 1; z < nz - 1; ++z) {
                size_t fi = cidx(x, y, z, ny, nz);
                full_to_int[fi] = n_int;
                int_to_full.push_back(static_cast<int>(fi));
                ++n_int;
            }

    std::pmr::vector<Eigen::Triplet<double>> trips_A(ar);
    trips_A.reserve(static_cast<size_t>(n_int) * 7);
    std::pmr::vector<double> boundary_sum(n_int, 0.0, ar);
    const double inv_dx2 = 1.0 / (dx_m * dx_m);

    for (int x = 1; x < nx - 1; ++x) {
        for (int y = 1; y < ny - 1; ++y) {
            for (int z = 1; z < nz - 1; ++z) {
                size_t fi = cidx(x, y, z, ny, nz);
                int row = full_to_int[fi];
                double diag = 0.0;
                auto add_face = [&](size_t nb) {
                    double kface = 2.0 * k[fi] * k[nb] / (k[fi] + k[nb] + 1e-12);
                    double coeff = kface * inv_dx2;
                    int col = full_to_int[nb];
                    if (col >= 0)
                        trips_A.emplace_back(row, col, coeff);
                    else
                        boundary_sum[row] += coeff;
                    diag -= coeff;
                };
                add_face(cidx(x - 1, y, z, ny, nz));
                add_face(cidx(x + 1, y, z, ny, nz));
                add_face(cidx(x, y - 1, z, ny, nz));
                add_face(cidx(x, y + 1, z, ny, nz));
                add_face(cidx(x, y, z - 1, ny, nz));
                add_face(cidx(x, y, z + 1, ny, nz));
                trips_A.emplace_back(row, row, diag);
            }
        }
    }

    // ---- solver state ----
    std::pmr::vector<double> T_new(n, 0.0, ar), T_adv(n, 0.0, ar);
    std::pmr::vector<double> t_liq(n, std::numeric_limits<double>::infinity(), ar);
    std::pmr::vector<double> t_sol(n, std::numeric_limits<double>::infinity(), ar);
    std::pmr::vector<double> G_at_ts(n, 0.0, ar), R_at_ts(n, 0.0, ar);
    std::pmr::vector<double> fs_final(n, 0.0, ar);
    std::pmr::vector<double> cp_eff(n, 0.0, ar);
    std::pmr::vector<double> T_tmp(n, 0.0, ar);
    std::pmr::vector<double> C_int(n_int, 0.0, ar), b(n_int, 0.0, ar);
    std::pmr::vector<Eigen::Triplet<double>> trips_M(ar);
    trips_M.reserve(trips_A.size());

    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper,
                              Eigen::DiagonalPreconditioner<double>> solver;
    solver.setTolerance(1e-5);
    solver.setMaxIterations(300);

    double t = 0.0;
    int step = 0;
    while (t < max_time_s) {
        double dt = dt_diff;
        if (has_velocity && t < fill_end)
            dt = std::min(dt_diff, max_adv_subcycles * dt_adv);
        if (t + dt > max_time_s) dt = max_time_s - t;
        if (dt <= 0.0) break;

        // ---- advection step (operator split) ----
        std::copy(T.begin(), T.end(), T_adv.begin());
        bool do_advection = (has_velocity && t < fill_end) ||
                            (feed_velocity_m_s > 1e-9 && t >= fill_end);
        if (do_advection) {
            bool use_vel = has_velocity && t < fill_end;
            double adv_dt = use_vel ? std::min(dt, fill_end - t) : dt;
            double dt_local = use_vel ? dt_adv : dt_feed;
            int n_sub = std::max(1, static_cast<int>(std::ceil(adv_dt / dt_local)));
            double sub_dt = adv_dt / n_sub;

            for (int s = 0; s < n_sub; ++s) {
                double sub_t = t + (s + 0.5) * sub_dt;
                #ifdef _OPENMP
                #pragma omp parallel for schedule(static)
                #endif
                for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
                    if (!metal[i] || T_adv[i] <= Ts) { T_tmp[i] = T_adv[i]; continue; }
                    if (fill_ptr && sub_t < fill_ptr[i]) { T_tmp[i] = T_adv[i]; continue; }

                    int z = static_cast<int>(i % nz);
                    size_t r = i / nz;
                    int y = static_cast<int>(r % ny);
                    int x = static_cast<int>(r / ny);

                    double vx = use_vel ? vel_ptr[i] : gx * feed_velocity_m_s;
                    double vy = use_vel ? vel_ptr[n + i] : gy * feed_velocity_m_s;
                    double vz = use_vel ? vel_ptr[2 * n + i] : gz * feed_velocity_m_s;

                    double dTdx = 0.0, dTdy = 0.0, dTdz = 0.0;
                    if (vx >= 0) {
                        dTdx = (x > 0) ? (T_adv[i] - T_adv[i - stride_x]) / dx_m : 0.0;
                    } else {
                        dTdx = (x < nx - 1) ? (T_adv[i + stride_x] - T_adv[i]) / dx_m : 0.0;
                    }
                    if (vy >= 0) {
                        dTdy = (y > 0) ? (T_adv[i] - T_adv[i - stride_y]) / dx_m : 0.0;
                    } else {
                        dTdy = (y < ny - 1) ? (T_adv[i + stride_y] - T_adv[i]) / dx_m : 0.0;
                    }
                    if (vz >= 0) {
                        dTdz = (z > 0) ? (T_adv[i] - T_adv[i - 1]) / dx_m : 0.0;
                    } else {
                        dTdz = (z < nz - 1) ? (T_adv[i + 1] - T_adv[i]) / dx_m : 0.0;
                    }
                    double adv = vx * dTdx + vy * dTdy + vz * dTdz;
                    T_tmp[i] = T_adv[i] - sub_dt * adv;
                }
                #ifdef _OPENMP
                #pragma omp parallel for schedule(static)
                #endif
                for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i)
                    T_adv[i] = std::max(T0, std::min(Tp, T_tmp[i]));
            }
        }

        // ---- implicit diffusion ----
        const double dT_mush = std::max(Tl - Ts, 1.0);
        const double df_cap = 1.0 / dT_mush;
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
            double cp = cp0[i];
            if (metal[i] && L > 0.0) {
                double df = dscheil_dT(T_adv[i], Tl, Ts, k_part);
                cp += L * std::max(0.0, std::min(df_cap, df));
            }
            cp_eff[i] = cp;
        }

        for (int r = 0; r < n_int; ++r) {
            int fi = int_to_full[r];
            C_int[r] = rho[fi] * cp_eff[fi];
            b[r] = C_int[r] * T_adv[fi] + dt * T0 * boundary_sum[r];
        }

        trips_M.clear();
        for (const auto &tp : trips_A) {
            double aval = tp.value();
            double mval;
            if (tp.row() == tp.col())
                mval = -dt * aval + C_int[tp.row()];
            else
                mval = -dt * aval;
            trips_M.emplace_back(tp.row(), tp.col(), mval);
        }

        Eigen::SparseMatrix<double> M(n_int, n_int);
        M.setFromTriplets(trips_M.begin(), trips_M.end());

        Eigen::VectorXd rhs(n_int), x(n_int);
        for (int r = 0; r < n_int; ++r) rhs[r] = b[r];
        for (int r = 0; r < n_int; ++r) x[r] = T_adv[int_to_full[r]];

        solver.compute(M);
        if (solver.info() == Eigen::Success) {
            x = solver.solveWithGuess(rhs, x);
        } else {
            for (int r = 0; r < n_int; ++r) x[r] = T_adv[int_to_full[r]];
        }

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) T_new[i] = T0; // boundary default
        for (int r = 0; r < n_int; ++r) {
            int fi = int_to_full[r];
            T_new[fi] = x[r];
        }

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
            if (!std::isfinite(T_new[i])) T_new[i] = T0;
            T_new[i] = std::max(T0, std::min(Tp, T_new[i]));
        }

        // keep gating liquid while the mould is still being filled
        if (t + dt <= fill_end) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
                if (gating[i] && metal[i] && T_new[i] < Tp)
                    T_new[i] = Tp;
            }
        }

        // ---- record solidification crossings ----
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
            if (!metal[i]) continue;
            double T_old_i = T[i];
            double T_new_i = T_new[i];
            if (T_old_i >= Tl && T_new_i < Tl && std::isinf(t_liq[i])) {
                t_liq[i] = t + dt * (Tl - T_old_i) / (T_new_i - T_old_i + 1e-12);
            }
            if (T_old_i >= Ts && T_new_i < Ts && std::isinf(t_sol[i])) {
                int z = static_cast<int>(i % nz);
                size_t r = i / nz;
                int y = static_cast<int>(r % ny);
                int x = static_cast<int>(r / ny);
                double dTdx = 0.0, dTdy = 0.0, dTdz = 0.0;
                if (x > 0 && x < nx - 1)
                    dTdx = (T_new[i + stride_x] - T_new[i - stride_x]) / (2.0 * dx_m);
                else if (x == 0)
                    dTdx = (T_new[i + stride_x] - T_new_i) / dx_m;
                else
                    dTdx = (T_new_i - T_new[i - stride_x]) / dx_m;
                if (y > 0 && y < ny - 1)
                    dTdy = (T_new[i + stride_y] - T_new[i - stride_y]) / (2.0 * dx_m);
                else if (y == 0)
                    dTdy = (T_new[i + stride_y] - T_new_i) / dx_m;
                else
                    dTdy = (T_new_i - T_new[i - stride_y]) / dx_m;
                if (z > 0 && z < nz - 1)
                    dTdz = (T_new[i + 1] - T_new[i - 1]) / (2.0 * dx_m);
                else if (z == 0)
                    dTdz = (T_new[i + 1] - T_new_i) / dx_m;
                else
                    dTdz = (T_new_i - T_new[i - 1]) / dx_m;
                double G_new = std::sqrt(dTdx * dTdx + dTdy * dTdy + dTdz * dTdz) / 1000.0;
                // Simple time-averaging when a cell crosses Ts over more than one step.
                if (G_at_ts[i] > 0.0)
                    G_at_ts[i] = 0.5 * (G_at_ts[i] + G_new);
                else
                    G_at_ts[i] = G_new;

                double fs_old = scheil_fs(T_old_i, Tl, Ts, k_part);
                double fs_new = scheil_fs(T_new_i, Tl, Ts, k_part);
                double dH = cp_eff[i] * (T_new_i - T_old_i) + L * (fs_new - fs_old);
                double R_new = std::fabs(dH) / (dt * std::max(cp_eff[i], 1e-9));
                if (R_at_ts[i] > 0.0)
                    R_at_ts[i] = 0.5 * (R_at_ts[i] + R_new);
                else
                    R_at_ts[i] = R_new;

                t_sol[i] = t + dt * (Ts - T_old_i) / (T_new_i - T_old_i + 1e-12);
            }
        }

        T.swap(T_new);
        t += dt;
        ++step;

        // early stop once all metal has solidified
        if (n_int > 0) {
            int unsolved = 0;
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) reduction(+:unsolved)
            #endif
            for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
                if (metal[i] && std::isinf(t_sol[i])) { ++unsolved; }
            }
            if (unsolved == 0) break;
        }
    }

    // ---- final Niyama and solid fraction ----
    std::pmr::vector<double> niyama(n, 0.0, ar);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
        fs_final[i] = scheil_fs(T[i], Tl, Ts, k_part);
        if (metal[i] && R_at_ts[i] > 1e-12 && std::isfinite(G_at_ts[i])) {
            niyama[i] = G_at_ts[i] / std::sqrt(R_at_ts[i]);
            if (!std::isfinite(niyama[i])) niyama[i] = 0.0;
        }
    }

    // Copy the final fields to heap-owned std::vectors so they survive the
    // function scope and can be returned to Python.
    auto *T_out = new std::vector<double>(T.begin(), T.end());
    auto *fs_out = new std::vector<double>(fs_final.begin(), fs_final.end());
    auto *tliq_out = new std::vector<double>(t_liq.begin(), t_liq.end());
    auto *tsol_out = new std::vector<double>(t_sol.begin(), t_sol.end());
    auto *G_out = new std::vector<double>(G_at_ts.begin(), G_at_ts.end());
    auto *R_out = new std::vector<double>(R_at_ts.begin(), R_at_ts.end());
    auto *N_out = new std::vector<double>(niyama.begin(), niyama.end());

    std::initializer_list<size_t> shape{static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz)};

    auto mk = [&](std::vector<double> *v) {
        return nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>>(
            v->data(), shape,
            nb::capsule(v, [](void *p) noexcept { delete static_cast<std::vector<double> *>(p); }));
    };

    return nb::make_tuple(mk(T_out), mk(fs_out), mk(tliq_out), mk(tsol_out), mk(G_out), mk(R_out), mk(N_out));
}

nb::tuple compute_porosity(
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> niyama,
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> M_mod,
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> feed_risk,
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> feed_eff,
    nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, -1>> part_mask,
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> velocity_magnitude,
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> darcy_factor,
    std::map<std::string, double> alloy,
    std::string carlson_curve_key,
    std::string material_family,
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> solid_fraction,
    double carbon_equivalent,
    double mold_rigidity_factor,
    double graphite_expansion_fraction,
    double inoculation_factor)
{
    int nx = static_cast<int>(niyama.shape(0));
    int ny = static_cast<int>(niyama.shape(1));
    int nz = static_cast<int>(niyama.shape(2));
    size_t n = static_cast<size_t>(nx) * ny * nz;

    // Virtual-address arena for all porosity working arrays.
    auto arena_ptr = make_arena_fallback(n, {50, 40, 30, 25, 15});
    VirtualArena* ar = arena_ptr.get();

    const double shrinkage_factor = getd(alloy, "shrinkage_factor", 0.03);
    const double dendrite_spacing_mm = getd(alloy, "dendrite_spacing_mm", 0.12);
    const double micro_pore_limit_um = getd(alloy, "micro_pore_limit_um", 50.0);
    const double macro_pore_limit_um = getd(alloy, "macro_pore_limit_um", 500.0);
    const double gas_pore_baseline_um = getd(alloy, "gas_pore_baseline_um", 0.5);
    const double pore_niyama_exponent = getd(alloy, "pore_niyama_exponent", 1.5);
    const double feed_risk_exponent = getd(alloy, "feed_risk_exponent", 1.2);
    const double gas_pore_time_factor = getd(alloy, "gas_pore_time_factor", 2.0);
    const double gas_pore_niyama_factor = getd(alloy, "gas_pore_niyama_factor", 0.5);
    const double critical_entrainment_velocity_m_s = getd(alloy, "critical_entrainment_velocity_m_s", 0.5);
    const double pore_entrainment_exponent = getd(alloy, "pore_entrainment_exponent", 1.5);
    const double pore_entrainment_factor = getd(alloy, "pore_entrainment_factor", 1.0);
    const double niyama_shrinkage = getd(alloy, "niyama_shrinkage", 1.5);
    const double niyama_star_scale = getd(alloy, "niyama_star_scale", 1.0);
    const double pore_size_um_per_porosity_pct = getd(alloy, "pore_size_um_per_porosity_pct", 400.0);
    const double pore_size_length_factor = getd(alloy, "pore_size_length_factor", 1.0);

    // Cast-iron graphite expansion parameters (may be overridden by caller).
    const double ce = (carbon_equivalent >= 0.0) ? carbon_equivalent : getd(alloy, "carbon_equivalent", 0.0);
    const double rigidity = (mold_rigidity_factor >= 0.0) ? mold_rigidity_factor : getd(alloy, "mold_rigidity_factor", 1.0);
    const double graphite_frac = (graphite_expansion_fraction >= 0.0) ? graphite_expansion_fraction : getd(alloy, "graphite_expansion_fraction", 0.0);
    const double inoc = (inoculation_factor >= 0.0) ? inoculation_factor : getd(alloy, "inoculation_factor", 1.0);

    const bool use_graphite = (material_family == "gray_iron" ||
                               material_family == "ductile_iron" ||
                               material_family == "white_iron") && ce > 0.0;

    // Eutectic solid fraction band derived from carbon equivalent.  CE ≈ 4.3 is
    // the eutectic composition; hypoeutectic iron reacts later (higher fs),
    // hypereutectic earlier (lower fs).
    const double fs_center = use_graphite ? std::max(0.1, std::min(0.9, 0.7 - 0.12 * (ce - 4.3))) : 0.0;
    const double fs_half_width = 0.12;
    const double fs_start = fs_center - fs_half_width;
    const double fs_end = fs_center + fs_half_width;

    // Cumulative graphite expansion fraction up to the current solid fraction.
    // The eutectic tent (peak = 1 at fs_center) is integrated so that the total
    // expansion equals graphite_expansion_fraction once solidification has
    // passed the eutectic band (fs >= fs_end).  Cells inside the band receive a
    // partial, smooth expansion.
    const double total_area = 0.5 * (fs_end - fs_start);  // area under the unit tent
    auto graphite_cumulative = [&](double fs) -> double {
        if (!use_graphite || total_area <= 0.0) return 0.0;
        if (fs <= fs_start) return 0.0;
        if (fs >= fs_end) return 1.0;
        double area = 0.0;
        if (fs < fs_center) {
            double dh = std::max(fs_center - fs_start, 1e-9);
            area = 0.5 * (fs - fs_start) * (fs - fs_start) / dh;
        } else {
            double left_area = 0.5 * (fs_center - fs_start);
            double dh = std::max(fs_end - fs_center, 1e-9);
            double df = fs - fs_center;
            area = left_area + df - 0.5 * df * df / dh;
        }
        return std::max(0.0, std::min(1.0, area / total_area));
    };

    auto effective_b0 = [&](double fs) -> double {
        double b = shrinkage_factor * 100.0;
        if (use_graphite) {
            double expansion = graphite_frac * inoc * graphite_cumulative(fs);
            double compensated = expansion * std::max(0.0, std::min(1.0, rigidity));
            b = std::max(0.0, (shrinkage_factor - compensated) * 100.0);
        }
        return b;
    };

    auto mold_movement = [&](double fs) -> double {
        if (!use_graphite) return 0.0;
        double expansion = graphite_frac * inoc * graphite_cumulative(fs);
        return std::max(0.0, expansion * (1.0 - std::max(0.0, std::min(1.0, rigidity))) * 100.0);
    };

    const double sdas_um = dendrite_spacing_mm * 1000.0;
    const double gp_ref = macro_pore_limit_um / std::max(pore_size_um_per_porosity_pct, 1e-9);

    bool has_feed_eff = (feed_eff.size() == n);
    bool has_vel = (velocity_magnitude.size() == n);
    bool has_darcy = (darcy_factor.size() == n);
    bool has_fs = (solid_fraction.size() == n);

    const double *ny_ptr = niyama.data();
    const double *M_ptr = M_mod.data();
    const double *fr_ptr = feed_risk.data();
    const double *fe_ptr = has_feed_eff ? feed_eff.data() : nullptr;
    const uint8_t *part_ptr = part_mask.data();
    const double *vel_ptr = has_vel ? velocity_magnitude.data() : nullptr;
    const double *darcy_ptr = has_darcy ? darcy_factor.data() : nullptr;
    const double *fs_ptr = has_fs ? solid_fraction.data() : nullptr;

    // find max modulus over the part for m_rel (sequential; avoids OpenMP 2.0 max reduction).
    double m_max = 1.0;
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
        if (part_ptr[i] && std::isfinite(M_ptr[i]) && M_ptr[i] > m_max) m_max = M_ptr[i];
    }

    std::pmr::vector<double> pore_size_um(n, 0.0, ar), pore_size_mm(n, 0.0, ar), shrinkage_um(n, 0.0, ar), gp_pct(n, 0.0, ar), mold_movement_um(n, 0.0, ar);
    std::pmr::vector<uint8_t> macro_mask(n, 0, ar), micro_mask(n, 0, ar), fine_mask(n, 0, ar);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
        bool part = part_ptr[i];
        double ny = ny_ptr[i];
        bool valid = part && std::isfinite(ny) && ny > 0.0;

        double feed_r = std::max(0.0, std::min(1.0, fr_ptr[i]));
        double feed_e = 1.0;
        if (has_feed_eff) feed_e = std::max(0.05, std::min(1.0, fe_ptr[i]));
        double feed_factor = std::pow(feed_r, feed_risk_exponent) * feed_e;

        double fs = (has_fs && std::isfinite(fs_ptr[i])) ? fs_ptr[i] : 0.0;
        double b0_eff = effective_b0(fs);
        double movement = mold_movement(fs);
        mold_movement_um[i] = valid ? movement : 0.0;

        double ny_star = ny * niyama_star_scale;
        double gp = valid ? carlson_gp(ny_star, b0_eff, carlson_curve_key) : 0.0;

        double dfac = 1.0;
        if (has_darcy && darcy_ptr[i] > 1.0) dfac = darcy_ptr[i];
        double gpv = valid ? gp * feed_factor * dfac : 0.0;
        gpv = std::max(0.0, std::min(b0_eff, gpv));
        gp_pct[i] = gpv;

        double max_d_um = std::max(2.0 * M_ptr[i] * 1000.0, sdas_um);
        double d_shrink = valid ? std::max(0.0, std::min(max_d_um, gpv * pore_size_um_per_porosity_pct * pore_size_length_factor)) : 0.0;
        shrinkage_um[i] = d_shrink;

        double baseline_min = std::max(gas_pore_baseline_um, sdas_um * 0.02);
        double raw_micro = valid ? std::max(0.0, std::min(1.0, 1.0 - ny / std::max(niyama_shrinkage, 1e-9))) : 0.0;
        double m_rel = std::max(0.0, std::min(1.0, M_ptr[i] / m_max));
        double baseline_factor = 1.0 + gas_pore_time_factor * m_rel + gas_pore_niyama_factor * raw_micro;
        double baseline_um = baseline_min * std::max(1.0, baseline_factor);

        double entrainment = 0.0;
        if (has_vel && part) {
            double v = vel_ptr[i];
            if (v > critical_entrainment_velocity_m_s && critical_entrainment_velocity_m_s > 1e-9) {
                double v_over = (v - critical_entrainment_velocity_m_s) / critical_entrainment_velocity_m_s;
                entrainment = std::pow(v_over, pore_entrainment_exponent);
            }
        }
        double d_gas = baseline_um * (1.0 + pore_entrainment_factor * entrainment);

        double psize = part ? std::max(d_shrink, d_gas) : 0.0;
        pore_size_um[i] = psize;
        pore_size_mm[i] = psize / 1000.0;

        double defect_risk = (gp_ref > 1e-12) ? (gpv / gp_ref + entrainment) : entrainment;
        defect_risk = std::max(0.0, std::min(50.0, defect_risk));
        double risk_local = 1.0 - std::exp(-defect_risk);

        macro_mask[i] = part && (psize >= macro_pore_limit_um) && (risk_local > 0.01) ? 1 : 0;
        micro_mask[i] = part && (psize >= micro_pore_limit_um) && (psize < macro_pore_limit_um) && (risk_local > 0.01) ? 1 : 0;
        fine_mask[i] = part && (psize > 0.0) && (psize < micro_pore_limit_um) && (risk_local > 0.01) ? 1 : 0;
    }

    // Copy result fields to heap-owned vectors for the Python capsules.
    auto *ps_out = new std::vector<double>(pore_size_um.begin(), pore_size_um.end());
    auto *psmm_out = new std::vector<double>(pore_size_mm.begin(), pore_size_mm.end());
    auto *macro_out = new std::vector<uint8_t>(macro_mask.begin(), macro_mask.end());
    auto *micro_out = new std::vector<uint8_t>(micro_mask.begin(), micro_mask.end());
    auto *fine_out = new std::vector<uint8_t>(fine_mask.begin(), fine_mask.end());
    auto *shrink_out = new std::vector<double>(shrinkage_um.begin(), shrinkage_um.end());
    auto *gp_out = new std::vector<double>(gp_pct.begin(), gp_pct.end());
    auto *mold_move_out = new std::vector<double>(mold_movement_um.begin(), mold_movement_um.end());

    std::initializer_list<size_t> shape{static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz)};

    auto mkd = [&](std::vector<double> *v) {
        return nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>>(
            v->data(), shape,
            nb::capsule(v, [](void *p) noexcept { delete static_cast<std::vector<double> *>(p); }));
    };
    auto mku8 = [&](std::vector<uint8_t> *v) {
        return nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, -1>>(
            v->data(), shape,
            nb::capsule(v, [](void *p) noexcept { delete static_cast<std::vector<uint8_t> *>(p); }));
    };

    return nb::make_tuple(mkd(ps_out), mkd(psmm_out), mku8(macro_out), mku8(micro_out), mku8(fine_out), mkd(shrink_out), mkd(gp_out), mkd(mold_move_out));
}

} // namespace josecast
