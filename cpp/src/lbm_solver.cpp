#include "josecast/lbm_solver.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

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
               double smagorinsky)
        : nx_(nx), ny_(ny), nz_(nz), dx_(dx), rho0_(rho),
          nu_phys_(nu), inflow_velocity_(inflow_velocity),
          t_max_(t_max), max_steps_(max_steps),
          cfl_target_(cfl_target), smag_const_(smagorinsky)
    {
        n_ = static_cast<size_t>(nx_) * ny_ * nz_;

        // Normalise gravity to a unit vector.
        double gnorm = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
        if (gnorm < 1e-12) gnorm = 1.0;
        gx_ = g[0] / gnorm;
        gy_ = g[1] / gnorm;
        gz_ = g[2] / gnorm;

        // Time step: keep the lattice velocity below cfl_target_ for stability.
        double v = std::max(inflow_velocity_, 1e-6);
        dt_ = cfl_target_ * dx_ / v;
        int nsteps = static_cast<int>(std::ceil(t_max_ / dt_));
        if (nsteps > max_steps_ && max_steps_ > 0) {
            nsteps = max_steps_;
            // Avoid exceeding the requested simulation time, but do not let the
            // lattice velocity grow by shrinking dt.
            t_max_ = nsteps * dt_;
            std::cerr << "[josecast_core] LBM: t_max reduced to " << t_max_
                      << " s so u_lb stays at " << (v * dt_ / dx_) << ".\n";
        }
        if (dt_ <= 0.0) dt_ = 1e-6;

        // Lattice kinematic viscosity.  Use a minimum tau to avoid BGK instability.
        double nu_lb = nu_phys_ * dt_ / (dx_ * dx_);
        tau0_ = 3.0 * nu_lb + 0.5;
        // A larger minimum relaxation time stabilises the pressure boundaries
        // and complex geometry.  Smagorinsky adds the turbulent viscosity in
        // high-shear regions, so the bulk is not over-damped.
        tau_min_ = 0.9;
        if (tau0_ < tau_min_) tau0_ = tau_min_;

        flags_.assign(n_, 0);
        for (size_t i = 0; i < n_; ++i) {
            uint8_t val = grid[i];
            if (val != 0) {
                flags_[i] = 1; // solid (including CORE 9)
            } else {
                flags_[i] = 0; // fluid/gas cavity
            }
        }
        for (size_t i = 0; i < n_; ++i) {
            if (inlet_mask[i]) flags_[i] = 2;
            if (outlet_mask[i]) flags_[i] = 3;
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
                ++cavity_cells_;
                fluid_list_.push_back(i);
            }
        }
    }

    void run() {
        if (cavity_cells_ == 0) return;

        int entrapment_interval = std::max(1, max_steps_ / 100);

        for (int step = 0; step < max_steps_; ++step) {
            if (t_ >= t_max_) break;

            compute_macroscopic();
            advect_phi();
            update_fill_times();
            if (step % entrapment_interval == 0) detect_entrapment();

            if (filled_fraction() >= 0.9999) {
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
        size_t filled = 0;
        for (size_t i = 0; i < n_; ++i) {
            if (flags_[i] != 1 && phi_[i] >= 0.5) ++filled;
        }
        return static_cast<double>(filled) / static_cast<double>(cavity_cells_);
    }

    double current_time() const { return t_; }
    int steps() const { return steps_; }

    void get_velocity_magnitude(std::vector<double>* out) const {
        out->resize(n_);
        for (size_t i = 0; i < n_; ++i) {
            double u = ux_[i], v = uy_[i], w = uz_[i];
            (*out)[i] = std::sqrt(u * u + v * v + w * w);
        }
    }

    void get_velocity(std::vector<double>* out) const {
        out->resize(3 * n_);
        for (size_t i = 0; i < n_; ++i) {
            (*out)[0 * n_ + i] = ux_[i];
            (*out)[1 * n_ + i] = uy_[i];
            (*out)[2 * n_ + i] = uz_[i];
        }
    }

    void get_phi(std::vector<double>* out) const { *out = phi_; }

    void get_fill_time(std::vector<double>* out) const { *out = fill_time_; }

    void get_entrapment(std::vector<double>* out) const {
        out->resize(n_);
        for (size_t i = 0; i < n_; ++i) {
            (*out)[i] = (trapped_time_[i] < std::numeric_limits<double>::infinity() / 2.0) ? 1.0 : 0.0;
        }
    }

    double total_entrapped_volume() const {
        double vol = 0.0;
        for (size_t i = 0; i < n_; ++i) {
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
    double rho_in_ = 1.0;
    double u_in_target_ = 0.0;

    std::vector<uint8_t> flags_;
    std::vector<std::array<double, 3>> outlet_normal_;
    std::vector<size_t> fluid_list_;
    std::vector<double> f_;
    std::vector<double> f_new_;
    std::vector<double> rho_, ux_, uy_, uz_;
    std::vector<double> phi_, phi_new_;
    std::vector<double> fill_time_, trapped_time_;

    void compute_macroscopic() {
        for (size_t idx = 0; idx < fluid_list_.size(); ++idx) {
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
            const double umax = 0.45;
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
        for (size_t idx = 0; idx < fluid_list_.size(); ++idx) {
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
            double f_body_x = gx_ * 9.81 * dt_ * dt_ / dx_;
            double f_body_y = gy_ * 9.81 * dt_ * dt_ / dx_;
            double f_body_z = gz_ * 9.81 * dt_ * dt_ / dx_;

            // Velocity relaxation: once the cell contains metal, gently drive
            // the velocity toward the inflow direction.  This stabilises the
            // filling front while still letting LBM capture turbulence.
            const double relax_rate = 1.0;
            if (phi_[i] > 0.5) {
                f_body_x += relax_rate * (gx_ * u_in_target_ - u);
                f_body_y += relax_rate * (gy_ * u_in_target_ - v);
                f_body_z += relax_rate * (gz_ * u_in_target_ - w);
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
                // For metal cells, reset to the target inflow velocity; for air
                // cells, reset to rest.  This prevents unphysical populations from
                // freezing a wrong velocity state.
                if (phi_[i] > 0.5) {
                    u = gx_ * u_in_target_;
                    v = gy_ * u_in_target_;
                    w = gz_ * u_in_target_;
                } else {
                    u = v = w = 0.0;
                }
                for (int q = 0; q < Q; ++q) {
                    f_new_[i * Q + q] = feq(q, 1.0, u, v, w);
                }
            }
        }

        // 2. Half-way bounce-back for fluid cells adjacent to solids.
        for (int x = 0; x < nx_; ++x) {
            for (int y = 0; y < ny_; ++y) {
                for (int z = 0; z < nz_; ++z) {
                    size_t idx = cidx(x, y, z, ny_, nz_);
                    if (flags_[idx] == 1) continue;
                    if (flags_[idx] == 2 || flags_[idx] == 3) continue;
                    double* fp = &f_new_[idx * Q];
                    for (int q = 1; q < Q; ++q) {
                        int nx2 = x + C[q][0];
                        int ny2 = y + C[q][1];
                        int nz2 = z + C[q][2];
                        bool solid_neighbour = false;
                        if (!in_cell(nx2, ny2, nz2, nx_, ny_, nz_)) {
                            solid_neighbour = true;
                        } else {
                            size_t nidx = cidx(nx2, ny2, nz2, ny_, nz_);
                            if (flags_[nidx] == 1) solid_neighbour = true;
                        }
                        if (solid_neighbour) {
                            int opp = OPP[q];
                            if (opp != q) {
                                std::swap(fp[q], fp[opp]);
                            }
                        }
                    }
                }
            }
        }

        // 3. Streaming: only the fluid (cavity + inlets/outlets) need be processed.
        for (size_t i : fluid_list_) {
            for (int q = 0; q < Q; ++q) f_[i * Q + q] = 0.0;
        }
        for (size_t idx = 0; idx < fluid_list_.size(); ++idx) {
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
                if (!in_cell(tx, ty, tz, nx_, ny_, nz_)) continue;
                size_t t = cidx(tx, ty, tz, ny_, nz_);
                if (flags_[t] == 1) continue;
                f_[t * Q + q] += fp[q];
            }
        }

        // 4. Enforce boundary conditions on post-streamed f.
        for (size_t idx = 0; idx < fluid_list_.size(); ++idx) {
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
                        phi_new_[ia] -= amount;
                        phi_new_[ib] += amount;
                    } else if (uface < 0.0) {
                        amount = -uface * phi_[ib];
                        phi_new_[ib] -= amount;
                        phi_new_[ia] += amount;
                    }
                }
            }
        }

        // Y faces.
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
                        phi_new_[ia] -= amount;
                        phi_new_[ib] += amount;
                    } else if (vface < 0.0) {
                        amount = -vface * phi_[ib];
                        phi_new_[ib] -= amount;
                        phi_new_[ia] += amount;
                    }
                }
            }
        }

        // Z faces.
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
                        phi_new_[ia] -= amount;
                        phi_new_[ib] += amount;
                    } else if (wface < 0.0) {
                        amount = -wface * phi_[ib];
                        phi_new_[ib] -= amount;
                        phi_new_[ia] += amount;
                    }
                }
            }
        }

        // Inlet source term: the inflow volume is injected into the downstream
        // (g-direction) neighbour so the metal front actually enters the cavity.
        // The inlet cell itself is held full.
        double u_in_lb = inflow_velocity_ * dt_ / dx_;
        if (u_in_lb > 1.0) u_in_lb = 1.0;
        int dgx = static_cast<int>(std::round(gx_));
        int dgy = static_cast<int>(std::round(gy_));
        int dgz = static_cast<int>(std::round(gz_));
        for (size_t i = 0; i < n_; ++i) {
            if (flags_[i] != 2) continue;
            int x = static_cast<int>(i / (ny_ * nz_));
            int yz = static_cast<int>(i % (ny_ * nz_));
            int y = yz / nz_;
            int z = yz % nz_;
            int sx = x + dgx;
            int sy = y + dgy;
            int sz = z + dgz;
            phi_new_[i] = 1.0;  // source cell stays full
            if (in_cell(sx, sy, sz, nx_, ny_, nz_)) {
                size_t j = cidx(sx, sy, sz, ny_, nz_);
                if (flags_[j] != 1) {
                    phi_new_[j] += u_in_lb;
                    if (phi_new_[j] > 1.0) phi_new_[j] = 1.0;
                }
            }
        }

        for (size_t i = 0; i < n_; ++i) {
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
        for (size_t i = 0; i < n_; ++i) {
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
    double smagorinsky)
{
    int nx = static_cast<int>(grid.shape(0));
    int ny = static_cast<int>(grid.shape(1));
    int nz = static_cast<int>(grid.shape(2));

    if (inlet_mask.shape(0) != nx || inlet_mask.shape(1) != ny || inlet_mask.shape(2) != nz ||
        outlet_mask.shape(0) != nx || outlet_mask.shape(1) != ny || outlet_mask.shape(2) != nz) {
        throw std::runtime_error("solve_lbm_filling: mask shapes do not match grid");
    }

    LBMFilling solver(nx, ny, nz, grid.data(), inlet_mask.data(), outlet_mask.data(),
                      dx, g, rho, nu, inflow_velocity, t_max, max_steps,
                      cfl_target, smagorinsky);

    solver.run();

    std::vector<double> ft, vmag, vel, phi, trap;
    solver.get_fill_time(&ft);
    solver.get_velocity_magnitude(&vmag);
    solver.get_velocity(&vel);
    solver.get_phi(&phi);
    solver.get_entrapment(&trap);

    const size_t n = static_cast<size_t>(nx) * ny * nz;
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
