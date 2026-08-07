#include "josecast/lbm_solver.h"

#include <cstddef>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nb = nanobind;

namespace josecast {

namespace {

// D3Q19 lattice constants.  Layout: c[19][3], weights[19].
// Speeds: 0=rest, 1-6=axis, 7-18=edge diagonals.
constexpr int Q = 19;
constexpr std::array<std::array<int, 3>, Q> C = {{
    {{ 0,  0,  0}},
    {{ 1,  0,  0}}, {{-1,  0,  0}}, {{ 0,  1,  0}}, {{ 0, -1,  0}}, {{ 0,  0,  1}}, {{ 0,  0, -1}},
    {{ 1,  1,  0}}, {{-1, -1,  0}}, {{ 1, -1,  0}}, {{-1,  1,  0}},
    {{ 1,  0,  1}}, {{-1,  0, -1}}, {{ 1,  0, -1}}, {{-1,  0,  1}},
    {{ 0,  1,  1}}, {{ 0, -1, -1}}, {{ 0,  1, -1}}, {{ 0, -1,  1}}
}};

constexpr std::array<double, Q> W = {{
    1.0 / 3.0,
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
}};

// Opposite direction for each velocity vector.
constexpr std::array<int, Q> OPP = {{
    0,
    2, 1, 4, 3, 6, 5,
    8, 7, 10, 9,
    12, 11, 14, 13,
    16, 15, 18, 17
}};

inline size_t cidx(int x, int y, int z, int ny, int nz) {
    return static_cast<size_t>((x * ny + y) * nz + z);
}

inline bool in_cell(int x, int y, int z, int nx, int ny, int nz) {
    return x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz;
}

inline double feq(int i, double rho, double ux, double uy, double uz) {
    double cu = static_cast<double>(C[i][0]) * ux
              + static_cast<double>(C[i][1]) * uy
              + static_cast<double>(C[i][2]) * uz;
    double u2 = ux * ux + uy * uy + uz * uz;
    return W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
}

class LBMFilling {
public:
    LBMFilling(int nx, int ny, int nz,
               const uint8_t* grid,
               const uint8_t* inlet_mask,
               const uint8_t* outlet_mask,
               double dx,
               std::array<double, 3> g,
               double rho,
               double nu,
               double inflow_velocity,
               double t_max,
               int max_steps,
               double cfl_target,
               double smagorinsky,
               const double* target_velocity = nullptr,
               const double* inlet_distance = nullptr)
        : nx_(nx), ny_(ny), nz_(nz), dx_(dx), rho0_(rho),
          nu_phys_(nu), inflow_velocity_(inflow_velocity),
          t_max_(t_max), max_steps_(max_steps),
          cfl_target_(cfl_target), smag_const_(smagorinsky)
    {
        n_ = static_cast<size_t>(nx_) * ny_ * nz_;

        // Preserve the physical gravity magnitude; only normalize the direction.
        double gnorm = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
        if (gnorm < 1e-12) gnorm = 9.81;
        g_mag_ = gnorm;
        gx_ = g[0] / gnorm;
        gy_ = g[1] / gnorm;
        gz_ = g[2] / gnorm;

        // Time step: keep the lattice velocity below cfl_target_ for stability.
        double v = std::max(inflow_velocity_, 1e-6);
        dt_ = cfl_target_ * dx_ / v;
        // Viscous diffusion limit: tau must stay close to the BGK stability
        // window.  This is especially important for high-viscosity alloys.
        if (nu_phys_ > 1e-12) {
            double dt_visc = dx_ * dx_ / (6.0 * nu_phys_) * 0.2;
            if (dt_visc < dt_) dt_ = dt_visc;
        }
        int nsteps = static_cast<int>(std::ceil(t_max_ / dt_));
        if (nsteps > max_steps_ && max_steps_ > 0) {
            nsteps = max_steps_;
            // Avoid exceeding the requested simulation time, but do not let the
            // lattice velocity grow by shrinking dt.
            t_max_ = nsteps * dt_;
        }
        if (dt_ <= 0.0) dt_ = 1e-6;
        target_scale_ = dt_ / dx_;
        // Take ownership of externally supplied target/distance arrays, or
        // compute them internally from the gating geometry when omitted.
        if (target_velocity != nullptr) {
            target_velocity_owned_.assign(target_velocity, target_velocity + 3 * n_);
            target_velocity_ = target_velocity_owned_.data();
            has_target_ = true;
        }
        if (inlet_distance != nullptr) {
            inlet_distance_owned_.assign(inlet_distance, inlet_distance + n_);
            inlet_distance_ = inlet_distance_owned_.data();
            has_inlet_dist_ = true;
        }

        // Lattice kinematic viscosity.  Use a minimum tau to avoid BGK instability.
        double nu_lb = nu_phys_ * dt_ / (dx_ * dx_);
        tau0_ = 3.0 * nu_lb + 0.5;
        // A larger minimum relaxation time stabilises the pressure boundaries
        // and complex geometry.  Smagorinsky adds the turbulent viscosity in
        // high-shear regions, so the bulk is not over-damped.
        tau_min_ = 0.55;
        if (tau0_ < tau_min_) tau0_ = tau_min_;
        if (tau0_ > 2.0) {
            std::cerr << "[josecast_core] LBM warning: tau0=" << tau0_
                      << " is very large; simulation is over-damped.\n";
        }

        flags_.assign(n_, 0);
        for (size_t i = 0; i < n_; ++i) {
            uint8_t val = grid[i];
            // grid values: 0 = solid mold/outside, 9 = sand core (solid obstacle),
            // everything else (1..N, excluding 9) is the casting cavity.
            if (val == 0 || val == 9) {
                flags_[i] = 1; // solid
            } else {
                flags_[i] = 0; // fluid/gas cavity
            }
        }
        // Inlet and outlet masks may overlap (e.g. the open top of the sprue
        // throat is both where metal enters and where the open-surface detector
        // places a vent).  The inlet condition must win: a source cell cannot
        // also be an outlet, otherwise the pour is never initialised and
        // filled_frac stays at 0.
        for (size_t i = 0; i < n_; ++i) {
            if (inlet_mask[i]) {
                flags_[i] = 2;
            } else if (outlet_mask[i]) {
                flags_[i] = 3;
            }
        }

        // If the caller did not supply a target velocity and/or inlet-distance
        // field, build them here from a 26-neighbour geodesic distance field
        // inside the cavity.  This lets Python call the solver with 12
        // positional arguments while the LBM still receives a proper initial
        // front and a directional body force.
        if (!has_target_ || !has_inlet_dist_) {
            build_geodesic_target();
        }

        // Compute a local outward normal for each outlet cell from its solid/boundary neighbours.
        outlet_normal_.assign(n_, {0.0, 0.0, 0.0});
        for (int x = 0; x < nx_; ++x) {
            for (int y = 0; y < ny_; ++y) {
                for (int z = 0; z < nz_; ++z) {
                    size_t i = cidx(x, y, z, ny_, nz_);
                    if (flags_[i] != 3) continue;
                    double best_abs = -1.0;
                    std::array<double, 3> best = {0.0, 0.0, 0.0};
                    for (int q = 1; q <= 6; ++q) {
                        int nx2 = x + C[q][0];
                        int ny2 = y + C[q][1];
                        int nz2 = z + C[q][2];
                        bool outside = !in_cell(nx2, ny2, nz2, nx_, ny_, nz_);
                        size_t j = 0;
                        bool solid = false;
                        if (!outside) {
                            j = cidx(nx2, ny2, nz2, ny_, nz_);
                            solid = (flags_[j] == 1);
                        }
                        if (outside || solid) {
                            double cx = static_cast<double>(C[q][0]);
                            double cy = static_cast<double>(C[q][1]);
                            double cz = static_cast<double>(C[q][2]);
                            double dot = cx * gx_ + cy * gy_ + cz * gz_;
                            if (std::abs(dot) > best_abs) {
                                best_abs = std::abs(dot);
                                best = {cx, cy, cz};
                            }
                        }
                    }
                    if (best_abs > 0.0) {
                        outlet_normal_[i] = best;
                    } else {
                        // No solid neighbour: fall back to the anti-gravity direction.
                        outlet_normal_[i] = {-gx_, -gy_, -gz_};
                    }
                }
            }
        }

        f_.assign(n_ * Q, 0.0);
        f_new_.assign(n_ * Q, 0.0);
        rho_.assign(n_, 1.0);
        ux_.assign(n_, 0.0);
        uy_.assign(n_, 0.0);
        uz_.assign(n_, 0.0);
        phi_.assign(n_, 0.0);
        phi_new_.assign(n_, 0.0);
        fill_time_.assign(n_, std::numeric_limits<double>::infinity());
        trapped_time_.assign(n_, std::numeric_limits<double>::infinity());

        #ifdef _OPENMP
        omp_set_num_threads(omp_get_max_threads());
        #endif
        // Inflow lattice velocity and a slight density head to drive the flow.
        double u_in_lb = inflow_velocity_ * dt_ / dx_;
        double u2_in = u_in_lb * u_in_lb;
        if (u2_in > 0.2) {
            double s = 0.2 / std::sqrt(u2_in);
            u_in_lb *= s;
            u2_in = u_in_lb * u_in_lb;
        }
        // Keep the LBM reference density at 1.0.  The metal volume is tracked by
        // phi and the flow is driven by gravity and the inflow velocity boundary.
        rho_in_ = 1.0;
        u_in_target_ = u_in_lb;
        double ux_in = gx_ * u_in_lb;
        double uy_in = gy_ * u_in_lb;
        double uz_in = gz_ * u_in_lb;
        for (size_t i = 0; i < n_; ++i) {
            if (flags_[i] == 2) {
                phi_[i] = 1.0;
                fill_time_[i] = 0.0;
                for (int q = 0; q < Q; ++q) {
                    f_[i * Q + q] = feq(q, rho_in_, ux_in, uy_in, uz_in);
                }
            } else if (flags_[i] == 1) {
                for (int q = 0; q < Q; ++q) f_[i * Q + q] = feq(q, 1.0, 0.0, 0.0, 0.0);
            } else {
                for (int q = 0; q < Q; ++q) f_[i * Q + q] = feq(q, 1.0, 0.0, 0.0, 0.0);
            }
        }

        cavity_cells_ = 0;
        fluid_list_.clear();
        for (size_t i = 0; i < n_; ++i) {
            if (flags_[i] != 1) {
                // Outlets/vents (flags == 3) remain air and should not count
                // toward the fill-fraction denominator.
                if (flags_[i] != 3) ++cavity_cells_;
                fluid_list_.push_back(i);
            }
        }
    }

    void run() {
        if (cavity_cells_ == 0) return;

        // Entrapment detection is O(n α(n)); run it rarely to keep the solver fast.
        int entrapment_interval = std::max(1, max_steps_ / 20);
        int fill_check_interval = 10;

        for (int step = 0; step < max_steps_; ++step) {
            if (t_ >= t_max_) break;

            current_step_ = step;
            compute_macroscopic();
            advect_phi();
            update_fill_times();
            if (step % entrapment_interval == 0) detect_entrapment();

            if (step % fill_check_interval == 0 && filled_fraction() >= 0.9999) {
                t_ += dt_;
                ++steps_;
                break;
            }

            collide_and_stream();

            t_ += dt_;
            ++steps_;
        }

        compute_macroscopic();
        detect_entrapment();
    }

    double filled_fraction() const {
        if (cavity_cells_ == 0) return 0.0;
        ptrdiff_t filled = 0;
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(+:filled)
        #endif
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n_); ++i) {
            // Inlet/source (2) and cavity (0) cells must fill; vents (3) stay empty.
            if (flags_[i] != 1 && flags_[i] != 3 && phi_[i] >= 0.5) ++filled;
        }
        return static_cast<double>(filled) / static_cast<double>(cavity_cells_);
    }

    double current_time() const { return t_; }
    int steps() const { return steps_; }

    void get_velocity_magnitude(std::vector<double>* out) const {
        out->resize(n_);
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n_); ++i) {
            double u = ux_[i], v = uy_[i], w = uz_[i];
            (*out)[i] = std::sqrt(u * u + v * v + w * w);
        }
    }

    void get_velocity(std::vector<double>* out) const {
        out->resize(3 * n_);
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n_); ++i) {
            (*out)[0 * n_ + i] = ux_[i];
            (*out)[1 * n_ + i] = uy_[i];
            (*out)[2 * n_ + i] = uz_[i];
        }
    }

    void get_phi(std::vector<double>* out) const { *out = phi_; }

    void get_fill_time(std::vector<double>* out) const { *out = fill_time_; }

    void get_entrapment(std::vector<double>* out) const {
        out->resize(n_);
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n_); ++i) {
            (*out)[i] = (trapped_time_[i] < std::numeric_limits<double>::infinity() / 2.0) ? 1.0 : 0.0;
        }
    }

    double total_entrapped_volume() const {
        double vol = 0.0;
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(+:vol)
        #endif
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n_); ++i) {
            if (trapped_time_[i] < std::numeric_limits<double>::infinity() / 2.0) {
                vol += (1.0 - phi_[i]);
            }
        }
        return vol * dx_ * dx_ * dx_;
    }

private:
    int nx_, ny_, nz_;
    size_t n_ = 0;
    size_t cavity_cells_ = 0;
    double dx_, dt_ = 0.0, t_ = 0.0;
    double rho0_, nu_phys_, inflow_velocity_;
    double t_max_;
    int max_steps_, steps_ = 0;
    double cfl_target_, smag_const_;
    double tau0_, tau_min_;
    double gx_, gy_, gz_;
    double g_mag_ = 9.81;
    double rho_in_ = 1.0;
    double u_in_target_ = 0.0;
    const double* target_velocity_ = nullptr;
    bool has_target_ = false;
    double target_scale_ = 0.0;  // (dt/dx) converts physical velocity to lattice velocity.

    std::vector<uint8_t> flags_;
    std::vector<std::array<double, 3>> outlet_normal_;
    std::vector<size_t> fluid_list_;
    std::vector<double> f_;
    std::vector<double> f_new_;
    std::vector<double> rho_, ux_, uy_, uz_;
    std::vector<double> phi_, phi_new_;
    std::vector<double> fill_time_, trapped_time_;
    const double* inlet_distance_ = nullptr;
    bool has_inlet_dist_ = false;
    int current_step_ = 0;

    std::vector<double> target_velocity_owned_;
    std::vector<double> inlet_distance_owned_;

    void build_geodesic_target() {
        // 26-neighbour Dijkstra from inlet cells within the cavity.
        std::vector<double> dist(n_, std::numeric_limits<double>::infinity());
        using PQItem = std::pair<double, size_t>;
        std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> pq;

        auto linear_to_ijk = [&](size_t idx, int& x, int& y, int& z) {
            x = static_cast<int>(idx / (ny_ * nz_));
            size_t rem = idx % (ny_ * nz_);
            y = static_cast<int>(rem / nz_);
            z = static_cast<int>(rem % nz_);
        };

        for (size_t i = 0; i < n_; ++i) {
            if (flags_[i] == 2) {
                dist[i] = 0.0;
                pq.emplace(0.0, i);
            }
        }

        while (!pq.empty()) {
            auto [d, i] = pq.top();
            pq.pop();
            if (d > dist[i] + 1e-12) continue;
            int x, y, z;
            linear_to_ijk(i, x, y, z);
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        int nx2 = x + dx, ny2 = y + dy, nz2 = z + dz;
                        if (!in_cell(nx2, ny2, nz2, nx_, ny_, nz_)) continue;
                        size_t j = cidx(nx2, ny2, nz2, ny_, nz_);
                        if (flags_[j] == 1) continue; // solid
                        double w = std::sqrt(static_cast<double>(dx * dx + dy * dy + dz * dz));
                        double nd = d + w;
                        if (nd + 1e-12 < dist[j]) {
                            dist[j] = nd;
                            pq.emplace(nd, j);
                        }
                    }
                }
            }
        }

        inlet_distance_owned_ = std::move(dist);
        inlet_distance_ = inlet_distance_owned_.data();
        has_inlet_dist_ = true;

        // Derive a target velocity field from the geodesic-distance gradient.
        // The gradient points away from the inlet (flow direction); normalise it
        // and scale by the inflow velocity.
        target_velocity_owned_.assign(3 * n_, 0.0);
        for (size_t i = 0; i < n_; ++i) {
            if (flags_[i] == 1) continue; // solid
            int x, y, z;
            linear_to_ijk(i, x, y, z);
            double gx = 0.0, gy = 0.0, gz = 0.0;
            bool has_x = false, has_y = false, has_z = false;

            auto dist_at = [&](int xx, int yy, int zz) -> double {
                if (!in_cell(xx, yy, zz, nx_, ny_, nz_)) return std::numeric_limits<double>::infinity();
                size_t j = cidx(xx, yy, zz, ny_, nz_);
                return (flags_[j] == 1) ? std::numeric_limits<double>::infinity() : inlet_distance_owned_[j];
            };

            double d0 = inlet_distance_owned_[i];
            double d_xm = dist_at(x - 1, y, z);
            double d_xp = dist_at(x + 1, y, z);
            if (std::isfinite(d_xm) && std::isfinite(d_xp)) {
                gx = (d_xp - d_xm) * 0.5;
                has_x = true;
            } else if (std::isfinite(d_xp) && std::isfinite(d0)) {
                gx = d_xp - d0;
                has_x = true;
            } else if (std::isfinite(d_xm) && std::isfinite(d0)) {
                gx = d0 - d_xm;
                has_x = true;
            }

            double d_ym = dist_at(x, y - 1, z);
            double d_yp = dist_at(x, y + 1, z);
            if (std::isfinite(d_ym) && std::isfinite(d_yp)) {
                gy = (d_yp - d_ym) * 0.5;
                has_y = true;
            } else if (std::isfinite(d_yp) && std::isfinite(d0)) {
                gy = d_yp - d0;
                has_y = true;
            } else if (std::isfinite(d_ym) && std::isfinite(d0)) {
                gy = d0 - d_ym;
                has_y = true;
            }

            double d_zm = dist_at(x, y, z - 1);
            double d_zp = dist_at(x, y, z + 1);
            if (std::isfinite(d_zm) && std::isfinite(d_zp)) {
                gz = (d_zp - d_zm) * 0.5;
                has_z = true;
            } else if (std::isfinite(d_zp) && std::isfinite(d0)) {
                gz = d_zp - d0;
                has_z = true;
            } else if (std::isfinite(d_zm) && std::isfinite(d0)) {
                gz = d0 - d_zm;
                has_z = true;
            }

            if ((has_x || has_y || has_z) &&
                std::isfinite(gx) && std::isfinite(gy) && std::isfinite(gz)) {
                double mag = std::sqrt(gx * gx + gy * gy + gz * gz);
                if (mag > 1e-12) {
                    double s = inflow_velocity_ / mag;
                    target_velocity_owned_[i] = gx * s;
                    target_velocity_owned_[n_ + i] = gy * s;
                    target_velocity_owned_[2 * n_ + i] = gz * s;
                    continue;
                }
            }
            // Fallback: use the (normalised) gravity vector.
            target_velocity_owned_[i] = gx_ * inflow_velocity_;
            target_velocity_owned_[n_ + i] = gy_ * inflow_velocity_;
            target_velocity_owned_[2 * n_ + i] = gz_ * inflow_velocity_;
        }
        target_velocity_ = target_velocity_owned_.data();
        has_target_ = true;
    }

    void compute_macroscopic() {
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(fluid_list_.size()); ++idx) {
            size_t i = fluid_list_[idx];
            double r = 0.0, px = 0.0, py = 0.0, pz = 0.0;
            const double* fp = &f_[i * Q];
            for (int q = 0; q < Q; ++q) {
                double fq = fp[q];
                r += fq;
                px += static_cast<double>(C[q][0]) * fq;
                py += static_cast<double>(C[q][1]) * fq;
                pz += static_cast<double>(C[q][2]) * fq;
            }
            if (!std::isfinite(r) || r < 1e-12) {
                r = 1.0;
                px = py = pz = 0.0;
                for (int q = 0; q < Q; ++q) f_[i * Q + q] = feq(q, 1.0, 0.0, 0.0, 0.0);
            }
            rho_[i] = r;
            ux_[i] = px / r;
            uy_[i] = py / r;
            uz_[i] = pz / r;

            // Keep the velocity within the lattice stability limit.  If it had
            // to be clipped, re-project the distribution to equilibrium.
            double u2 = ux_[i] * ux_[i] + uy_[i] * uy_[i] + uz_[i] * uz_[i];
            const double umax = 0.25;
            if (u2 > umax * umax) {
                double s = umax / std::sqrt(u2);
                ux_[i] *= s; uy_[i] *= s; uz_[i] *= s;
                for (int q = 0; q < Q; ++q) {
                    f_[i * Q + q] = feq(q, rho_[i], ux_[i], uy_[i], uz_[i]);
                }
            }

            if (flags_[i] == 2) {
                double u = inflow_velocity_ * dt_ / dx_;
                double u2 = u * u;
                if (u2 > 0.25) u = std::sqrt(0.25);
                rho_[i] = rho_in_;
                ux_[i] = gx_ * u;
                uy_[i] = gy_ * u;
                uz_[i] = gz_ * u;
            }
        }
    }

    void collide_and_stream() {
        // 1. Collision (BGK) with Guo forcing and Smagorinsky eddy viscosity.
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(fluid_list_.size()); ++idx) {
            size_t i = fluid_list_[idx];
            if (flags_[i] == 1) {
                for (int q = 0; q < Q; ++q) f_new_[i * Q + q] = feq(q, 1.0, 0.0, 0.0, 0.0);
                continue;
            }

            double u = ux_[i], v = uy_[i], w = uz_[i];
            double r = rho_[i];
            if (!std::isfinite(r) || r <= 0.0 ||
                !std::isfinite(u) || !std::isfinite(v) || !std::isfinite(w)) {
                r = 1.0; u = v = w = 0.0;
                for (int q = 0; q < Q; ++q) f_[i * Q + q] = feq(q, 1.0, 0.0, 0.0, 0.0);
            }

            double pi_xx = 0.0, pi_yy = 0.0, pi_zz = 0.0;
            double pi_xy = 0.0, pi_xz = 0.0, pi_yz = 0.0;
            const double* fp = &f_[i * Q];
            for (int q = 0; q < Q; ++q) {
                double fe = feq(q, r, u, v, w);
                double df = fp[q] - fe;
                int cx = C[q][0], cy = C[q][1], cz = C[q][2];
                pi_xx += cx * cx * df;
                pi_yy += cy * cy * df;
                pi_zz += cz * cz * df;
                pi_xy += cx * cy * df;
                pi_xz += cx * cz * df;
                pi_yz += cy * cz * df;
            }
            double q_norm = std::sqrt(pi_xx * pi_xx + pi_yy * pi_yy + pi_zz * pi_zz
                                      + 2.0 * (pi_xy * pi_xy + pi_xz * pi_xz + pi_yz * pi_yz));
            double tau = tau0_;
            if (smag_const_ > 0.0 && q_norm > 0.0) {
                double s_mag = (3.0 / (2.0 * r * tau0_)) * q_norm;
                double nu_t = smag_const_ * smag_const_ * s_mag;
                double tau_eff = tau0_ + 3.0 * nu_t;
                if (tau_eff < tau_min_) tau_eff = tau_min_;
                tau = tau_eff;
            }

            double one_minus_half_omega = 1.0 - 0.5 / tau;
            double f_body_x = gx_ * g_mag_ * dt_ * dt_ / dx_;
            double f_body_y = gy_ * g_mag_ * dt_ * dt_ / dx_;
            double f_body_z = gz_ * g_mag_ * dt_ * dt_ / dx_;

            // Velocity relaxation: drive the velocity toward a target (e.g. from a
            // Darcy pressure solve).  If no target is supplied, fallback to the
            // global inflow direction once the cell contains metal.
            double relax_rate = has_target_ ? 0.6 : 1.0;
            if (has_target_ || phi_[i] > 0.5) {
                double tu = gx_ * u_in_target_;
                double tv = gy_ * u_in_target_;
                double tw = gz_ * u_in_target_;
                if (has_target_) {
                    tu = target_velocity_[i] * target_scale_;
                    tv = target_velocity_[n_ + i] * target_scale_;
                    tw = target_velocity_[2 * n_ + i] * target_scale_;
                }
                f_body_x += relax_rate * (tu - u);
                f_body_y += relax_rate * (tv - v);
                f_body_z += relax_rate * (tw - w);
            }

            for (int q = 0; q < Q; ++q) {
                double fe = feq(q, r, u, v, w);
                double df = fp[q] - fe;
                int cx = C[q][0], cy = C[q][1], cz = C[q][2];
                double force = one_minus_half_omega * W[q] * 3.0 *
                    ((static_cast<double>(cx) - u) * f_body_x / r +
                     (static_cast<double>(cy) - v) * f_body_y / r +
                     (static_cast<double>(cz) - w) * f_body_z / r);
                f_new_[i * Q + q] = fp[q] - (df / tau) + force;
            }

            // If the collision produced unphysical values (negative populations,
            // NaN, or extreme density), reset this cell to equilibrium with the
            // clamped local velocity.
            double rho_new = 0.0;
            bool bad = false;
            for (int q = 0; q < Q; ++q) {
                double fq = f_new_[i * Q + q];
                if (!std::isfinite(fq) || fq < -1e-3) bad = true;
                rho_new += fq;
            }
            if (bad || !std::isfinite(rho_new) || rho_new < 0.5 || rho_new > 2.0 ||
                !std::isfinite(u) || !std::isfinite(v) || !std::isfinite(w)) {
                // For metal cells, reset to the target velocity; for air cells,
                // reset to rest.  This prevents unphysical populations from
                // freezing a wrong velocity state.
                if (phi_[i] > 0.5) {
                    if (has_target_) {
                        u = target_velocity_[i] * target_scale_;
                        v = target_velocity_[n_ + i] * target_scale_;
                        w = target_velocity_[2 * n_ + i] * target_scale_;
                    } else {
                        u = gx_ * u_in_target_;
                        v = gy_ * u_in_target_;
                        w = gz_ * u_in_target_;
                    }
                } else {
                    u = v = w = 0.0;
                }
                for (int q = 0; q < Q; ++q) {
                    f_new_[i * Q + q] = feq(q, 1.0, u, v, w);
                }
            }
        }

        // 2+3. Streaming with proper half-way bounce-back.  Distributions that
        // would enter a solid cell are returned to the source cell in the
        // opposite direction.  Distributions that leave the domain are discarded;
        // for outlet cells this is the correct open-boundary outflow.
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n_); ++i) {
            for (int q = 0; q < Q; ++q) f_[i * Q + q] = 0.0;
        }
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(fluid_list_.size()); ++idx) {
            size_t s = fluid_list_[idx];
            int x = static_cast<int>(s / (ny_ * nz_));
            int yz = static_cast<int>(s % (ny_ * nz_));
            int y = yz / nz_;
            int z = yz % nz_;
            const double* fp = &f_new_[s * Q];
            for (int q = 0; q < Q; ++q) {
                int tx = x + C[q][0];
                int ty = y + C[q][1];
                int tz = z + C[q][2];
                bool outside = !in_cell(tx, ty, tz, nx_, ny_, nz_);
                if (outside) {
                    // Let distributions leave through outlets; bounce off true
                    // domain walls (non-outlet cells at the grid boundary).
                    if (flags_[s] == 3) continue;
                    int opp = OPP[q];
                    f_[s * Q + opp] += fp[q];
                    continue;
                }
                size_t t = cidx(tx, ty, tz, ny_, nz_);
                if (flags_[t] == 1) {
                    int opp = OPP[q];
                    f_[s * Q + opp] += fp[q];
                } else {
                    f_[t * Q + q] += fp[q];
                }
            }
        }

        // 4. Enforce boundary conditions on post-streamed f.
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (ptrdiff_t idx = 0; idx < static_cast<ptrdiff_t>(fluid_list_.size()); ++idx) {
            size_t i = fluid_list_[idx];
            if (flags_[i] == 1) {
                // Solid wall: no-slip equilibrium.
                for (int q = 0; q < Q; ++q) {
                    f_[i * Q + q] = feq(q, 1.0, 0.0, 0.0, 0.0);
                }
            } else if (flags_[i] == 2) {
                // Velocity inlet: force the whole distribution to the inflow state.
                double ui = inflow_velocity_ * dt_ / dx_;
                double u2 = ui * ui;
                if (u2 > 0.25) ui = std::sqrt(0.25);
                double u = gx_ * ui, v = gy_ * ui, w = gz_ * ui;
                for (int q = 0; q < Q; ++q) {
                    f_[i * Q + q] = feq(q, rho_in_, u, v, w);
                }
            } else if (flags_[i] == 3) {
                // Pressure outlet / vent: extrapolate velocity from the interior
                // neighbour and set the incoming/tangential populations to that
                // equilibrium state.  Populations leaving the domain are kept
                // as streamed.
                int x = static_cast<int>(i / (ny_ * nz_));
                int yz = static_cast<int>(i % (ny_ * nz_));
                int y = yz / nz_;
                int z = yz % nz_;
                const auto& n = outlet_normal_[i];
                int sx = x - static_cast<int>(std::round(n[0]));
                int sy = y - static_cast<int>(std::round(n[1]));
                int sz = z - static_cast<int>(std::round(n[2]));
                double u_b = 0.0, v_b = 0.0, w_b = 0.0;
                if (in_cell(sx, sy, sz, nx_, ny_, nz_)) {
                    size_t j = cidx(sx, sy, sz, ny_, nz_);
                    if (flags_[j] != 1) {
                        u_b = ux_[j]; v_b = uy_[j]; w_b = uz_[j];
                    }
                }
                for (int q = 0; q < Q; ++q) {
                    double cdot = static_cast<double>(C[q][0]) * n[0]
                                + static_cast<double>(C[q][1]) * n[1]
                                + static_cast<double>(C[q][2]) * n[2];
                    if (cdot <= 0.0) {
                        f_[i * Q + q] = feq(q, 1.0, u_b, v_b, w_b);
                    }
                }
            }
        }
    }

    void advect_phi() {
        std::copy(phi_.begin(), phi_.end(), phi_new_.begin());

        auto idx = [&](int x, int y, int z) -> size_t {
            return cidx(x, y, z, ny_, nz_);
        };

        auto valid = [&](int x, int y, int z) -> bool {
            return in_cell(x, y, z, nx_, ny_, nz_) && flags_[cidx(x, y, z, ny_, nz_)] != 1;
        };

        // X faces.
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int x = 1; x < nx_; ++x) {
            for (int y = 0; y < ny_; ++y) {
                for (int z = 0; z < nz_; ++z) {
                    if (!valid(x - 1, y, z) || !valid(x, y, z)) continue;
                    size_t ia = idx(x - 1, y, z);
                    size_t ib = idx(x, y, z);
                    double uface = 0.5 * (ux_[ia] + ux_[ib]);
                    double amount;
                    if (uface > 0.0) {
                        amount = uface * phi_[ia];
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        phi_new_[ia] -= amount;
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        phi_new_[ib] += amount;
                    } else if (uface < 0.0) {
                        amount = -uface * phi_[ib];
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        phi_new_[ib] -= amount;
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        phi_new_[ia] += amount;
                    }
                }
            }
        }

        // Y faces.
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int x = 0; x < nx_; ++x) {
            for (int y = 1; y < ny_; ++y) {
                for (int z = 0; z < nz_; ++z) {
                    if (!valid(x, y - 1, z) || !valid(x, y, z)) continue;
                    size_t ia = idx(x, y - 1, z);
                    size_t ib = idx(x, y, z);
                    double vface = 0.5 * (uy_[ia] + uy_[ib]);
                    double amount;
                    if (vface > 0.0) {
                        amount = vface * phi_[ia];
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        phi_new_[ia] -= amount;
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        phi_new_[ib] += amount;
                    } else if (vface < 0.0) {
                        amount = -vface * phi_[ib];
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        phi_new_[ib] -= amount;
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        phi_new_[ia] += amount;
                    }
                }
            }
        }

        // Z faces.
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int x = 0; x < nx_; ++x) {
            for (int y = 0; y < ny_; ++y) {
                for (int z = 1; z < nz_; ++z) {
                    if (!valid(x, y, z - 1) || !valid(x, y, z)) continue;
                    size_t ia = idx(x, y, z - 1);
                    size_t ib = idx(x, y, z);
                    double wface = 0.5 * (uz_[ia] + uz_[ib]);
                    double amount;
                    if (wface > 0.0) {
                        amount = wface * phi_[ia];
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        phi_new_[ia] -= amount;
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        phi_new_[ib] += amount;
                    } else if (wface < 0.0) {
                        amount = -wface * phi_[ib];
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        phi_new_[ib] -= amount;
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        phi_new_[ia] += amount;
                    }
                }
            }
        }

        // Source term: every filled cell injects metal into its downstream
        // (velocity-direction) neighbour.  When a Darcy target velocity is supplied,
        // ux/uy/uz already point along the gating path, so the front follows the
        // sprue/runner/ingate network instead of only the global gravity axis.
        // Inlet cells are held full so the pour does not run dry.
        double u_in_lb = inflow_velocity_ * dt_ / dx_;
        if (u_in_lb > 1.0) u_in_lb = 1.0;
        if (u_in_lb < 0.01) u_in_lb = 0.01;

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n_); ++i) {
            if (flags_[i] == 1 || flags_[i] == 3) continue;
            if (phi_[i] < 0.5 && flags_[i] != 2) continue;
            int x = static_cast<int>(i / (ny_ * nz_));
            int yz = static_cast<int>(i % (ny_ * nz_));
            int y = yz / nz_;
            int z = yz % nz_;

            // Prefer the supplied Darcy/gating target direction; if missing, use
            // the LBM macroscopic velocity; if that is zero, fall back to gravity.
            double ux, uy, uz;
            if (has_target_) {
                double tx = target_velocity_[i];
                double ty = target_velocity_[n_ + i];
                double tz = target_velocity_[2 * n_ + i];
                double tmag = std::sqrt(tx * tx + ty * ty + tz * tz);
                if (tmag > 1e-12) {
                    ux = tx / tmag;
                    uy = ty / tmag;
                    uz = tz / tmag;
                } else {
                    ux = gx_; uy = gy_; uz = gz_;
                }
            } else if (flags_[i] == 2) {
                ux = gx_; uy = gy_; uz = gz_;
            } else {
                ux = ux_[i]; uy = uy_[i]; uz = uz_[i];
                double umag = std::sqrt(ux * ux + uy * uy + uz * uz);
                if (umag < 1e-12) {
                    ux = gx_; uy = gy_; uz = gz_;
                } else {
                    ux /= umag; uy /= umag; uz /= umag;
                }
            }

            // 6 face directions; pick the one most aligned with the velocity.
            int best_dir = -1;
            double best_dot = 0.0;
            for (int d = 0; d < 6; ++d) {
                int cx = (d == 0 ? 1 : (d == 1 ? -1 : 0));
                int cy = (d == 2 ? 1 : (d == 3 ? -1 : 0));
                int cz = (d == 4 ? 1 : (d == 5 ? -1 : 0));
                double dot = ux * cx + uy * cy + uz * cz;
                if (dot > best_dot) {
                    best_dot = dot;
                    best_dir = d;
                }
            }

            phi_new_[i] = std::max(phi_new_[i], 1.0);  // filled/source cells stay full
            if (best_dir >= 0 && best_dot > 0.1) {
                int cx = (best_dir == 0 ? 1 : (best_dir == 1 ? -1 : 0));
                int cy = (best_dir == 2 ? 1 : (best_dir == 3 ? -1 : 0));
                int cz = (best_dir == 4 ? 1 : (best_dir == 5 ? -1 : 0));
                int sx = x + cx, sy = y + cy, sz = z + cz;
                if (in_cell(sx, sy, sz, nx_, ny_, nz_)) {
                    size_t j = cidx(sx, sy, sz, ny_, nz_);
                    if (flags_[j] != 1 && flags_[j] != 3) {
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        phi_new_[j] += u_in_lb * best_dot;
                    }
                }
            }
        }

        // Deterministic front propagation: a cell is guaranteed to be filled once
        // the injected front has had enough time to travel its geodesic distance
        // from the inlet.  This prevents the donor-cell advection from stalling in
        // complex gating while still letting the LBM compute local velocity and
        // turbulence fields.
        if (has_inlet_dist_) {
            double threshold = (static_cast<double>(current_step_) + 1.0) * u_in_lb;
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n_); ++i) {
                if (flags_[i] == 1 || flags_[i] == 3) continue;
                if (inlet_distance_[i] >= 0.0 && inlet_distance_[i] <= threshold) {
                    phi_new_[i] = std::max(phi_new_[i], 1.0);
                }
            }
        }

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n_); ++i) {
            if (flags_[i] == 1) {
                phi_new_[i] = 0.0;
            } else {
                if (phi_new_[i] < 0.0) phi_new_[i] = 0.0;
                if (phi_new_[i] > 1.0) phi_new_[i] = 1.0;
            }
        }
        phi_.swap(phi_new_);
    }

    void update_fill_times() {
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n_); ++i) {
            if (flags_[i] != 1 && phi_[i] >= 0.5 && fill_time_[i] > t_) {
                fill_time_[i] = t_;
            }
        }
    }

    void detect_entrapment() {
        if (n_ == 0) return;

        std::vector<int> parent(n_, -1);
        for (size_t i = 0; i < n_; ++i) {
            if (flags_[i] != 1 && phi_[i] < 0.5) parent[i] = static_cast<int>(i);
        }

        auto find = [&](int a) {
            while (parent[a] >= 0 && parent[a] != a) {
                parent[a] = parent[parent[a]];
                a = parent[a];
            }
            return a;
        };

        auto unite = [&](int a, int b) {
            int ra = find(a);
            int rb = find(b);
            if (ra == rb) return;
            if (ra > rb) std::swap(ra, rb);
            parent[rb] = ra;
        };

        for (int x = 0; x < nx_; ++x) {
            for (int y = 0; y < ny_; ++y) {
                for (int z = 0; z < nz_; ++z) {
                    size_t i = cidx(x, y, z, ny_, nz_);
                    if (parent[i] < 0) continue;
                    if (x + 1 < nx_) {
                        size_t j = cidx(x + 1, y, z, ny_, nz_);
                        if (parent[j] >= 0) unite(static_cast<int>(i), static_cast<int>(j));
                    }
                    if (y + 1 < ny_) {
                        size_t j = cidx(x, y + 1, z, ny_, nz_);
                        if (parent[j] >= 0) unite(static_cast<int>(i), static_cast<int>(j));
                    }
                    if (z + 1 < nz_) {
                        size_t j = cidx(x, y, z + 1, ny_, nz_);
                        if (parent[j] >= 0) unite(static_cast<int>(i), static_cast<int>(j));
                    }
                }
            }
        }

        std::vector<char> root_open(n_, 0);
        for (int x = 0; x < nx_; ++x) {
            for (int y = 0; y < ny_; ++y) {
                for (int z = 0; z < nz_; ++z) {
                    size_t i = cidx(x, y, z, ny_, nz_);
                    if (parent[i] < 0) continue;
                    bool open = false;
                    if (x == 0 || x == nx_ - 1 || y == 0 || y == ny_ - 1 || z == 0 || z == nz_ - 1) {
                        open = true;
                    }
                    if (flags_[i] == 3) open = true;
                    if (open) {
                        int r = find(static_cast<int>(i));
                        root_open[r] = 1;
                    }
                }
            }
        }

        for (size_t i = 0; i < n_; ++i) {
            if (parent[i] < 0) continue;
            int r = find(static_cast<int>(i));
            if (root_open[r]) continue;
            if (trapped_time_[i] > t_ + dt_ * 0.5) {
                trapped_time_[i] = t_;
            }
        }
    }
};

} // namespace

nb::tuple solve_lbm_filling(
    nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, -1>> grid,
    nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, -1>> inlet_mask,
    nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, -1>> outlet_mask,
    double dx,
    std::array<double, 3> g,
    double rho,
    double nu,
    double inflow_velocity,
    double t_max,
    int max_steps,
    double cfl_target,
    double smagorinsky,
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1, -1>> target_velocity,
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> inlet_distance)
{
    int nx = static_cast<int>(grid.shape(0));
    int ny = static_cast<int>(grid.shape(1));
    int nz = static_cast<int>(grid.shape(2));

    if (inlet_mask.shape(0) != nx || inlet_mask.shape(1) != ny || inlet_mask.shape(2) != nz ||
        outlet_mask.shape(0) != nx || outlet_mask.shape(1) != ny || outlet_mask.shape(2) != nz) {
        throw std::runtime_error("solve_lbm_filling: mask shapes do not match grid");
    }

    const size_t n = static_cast<size_t>(nx) * ny * nz;
    const double* target_ptr = nullptr;
    if (target_velocity.ndim() == 4 && target_velocity.shape(0) == 3 &&
        static_cast<size_t>(target_velocity.size()) == 3 * n) {
        target_ptr = target_velocity.data();
    }
    const double* inlet_dist_ptr = nullptr;
    if (inlet_distance.ndim() == 3 &&
        static_cast<size_t>(inlet_distance.size()) == n) {
        inlet_dist_ptr = inlet_distance.data();
    }

    LBMFilling solver(nx, ny, nz, grid.data(), inlet_mask.data(), outlet_mask.data(),
                      dx, g, rho, nu, inflow_velocity, t_max, max_steps,
                      cfl_target, smagorinsky, target_ptr, inlet_dist_ptr);

    solver.run();

    std::vector<double> ft, vmag, vel, phi, trap;
    solver.get_fill_time(&ft);
    solver.get_velocity_magnitude(&vmag);
    solver.get_velocity(&vel);
    solver.get_phi(&phi);
    solver.get_entrapment(&trap);

    auto* ft_vec = new std::vector<double>(std::move(ft));
    auto* vmag_vec = new std::vector<double>(std::move(vmag));
    auto* vel_vec = new std::vector<double>(std::move(vel));
    auto* phi_vec = new std::vector<double>(std::move(phi));
    auto* trap_vec = new std::vector<double>(std::move(trap));

    auto cap_ft = nb::capsule(ft_vec, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    auto cap_vmag = nb::capsule(vmag_vec, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    auto cap_vel = nb::capsule(vel_vec, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    auto cap_phi = nb::capsule(phi_vec, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    auto cap_trap = nb::capsule(trap_vec, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });

    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> arr_ft(
        ft_vec->data(), {static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz)}, cap_ft);
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> arr_vmag(
        vmag_vec->data(), {static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz)}, cap_vmag);
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> arr_phi(
        phi_vec->data(), {static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz)}, cap_phi);
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> arr_trap(
        trap_vec->data(), {static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz)}, cap_trap);
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> arr_vel(
        vel_vec->data(), {3, static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz)}, cap_vel);

    return nb::make_tuple(
        arr_ft,
        arr_vmag,
        arr_vel,
        arr_phi,
        arr_trap,
        solver.total_entrapped_volume(),
        solver.filled_fraction() >= 0.9999,
        solver.current_time(),
        solver.filled_fraction(),
        solver.steps()
    );
}

} // namespace josecast
