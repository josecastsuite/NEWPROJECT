#include "josecast/ns_solver.h"
#include "josecast/arena.hpp"

#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <memory_resource>
#include <vector>

namespace josecast {

namespace {

// Grid data layout: shape (nx, ny, nz), C-contiguous, fastest axis = z.
inline size_t cidx(int x, int y, int z, int ny, int nz) {
    return static_cast<size_t>((x * ny + y) * nz + z);
}

inline bool in_cell(int x, int y, int z, int nx, int ny, int nz) {
    return x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz;
}

double trilerp(const std::pmr::vector<double>& f, int nx, int ny, int nz,
               double x, double y, double z)
{
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int z0 = static_cast<int>(std::floor(z));
    double wx = x - x0;
    double wy = y - y0;
    double wz = z - z0;

    double val = 0.0;
    for (int dx = 0; dx < 2; ++dx) {
        int xx = std::clamp(x0 + dx, 0, nx - 1);
        double wxx = dx ? wx : (1.0 - wx);
        for (int dy = 0; dy < 2; ++dy) {
            int yy = std::clamp(y0 + dy, 0, ny - 1);
            double wyy = dy ? wy : (1.0 - wy);
            for (int dz = 0; dz < 2; ++dz) {
                int zz = std::clamp(z0 + dz, 0, nz - 1);
                double wzz = dz ? wz : (1.0 - wz);
                val += wxx * wyy * wzz * f[cidx(xx, yy, zz, ny, nz)];
            }
        }
    }
    return val;
}

std::vector<double> amgcl_solve_cg(
    const std::pmr::vector<int>& rows,
    const std::pmr::vector<int>& cols,
    const std::pmr::vector<double>& data,
    const std::pmr::vector<double>& rhs,
    std::pmr::memory_resource* ar,
    int max_iter,
    double tol)
{
    const size_t n = rhs.size();
    const size_t nnz = rows.size();

    // COO -> CSR, index arrays carved from the arena.  The AMGCL CG solver
    // requires standard std::vectors for its right-hand-side and solution
    // vectors, so those remain heap-owned.
    std::pmr::vector<ptrdiff_t> ptr(n + 1, 0, ar);
    for (size_t i = 0; i < nnz; ++i) {
        ptr[static_cast<size_t>(rows[i]) + 1]++;
    }
    for (size_t i = 0; i < n; ++i) {
        ptr[i + 1] += ptr[i];
    }

    std::pmr::vector<ptrdiff_t> tmp = ptr;
    std::pmr::vector<ptrdiff_t> col(nnz, 0, ar);
    std::pmr::vector<double> val(nnz, 0.0, ar);
    for (size_t i = 0; i < nnz; ++i) {
        size_t r = static_cast<size_t>(rows[i]);
        size_t pos = tmp[r]++;
        col[pos] = static_cast<ptrdiff_t>(cols[i]);
        val[pos] = -data[i]; // solve -A x = -b
    }

    for (size_t i = 0; i < n; ++i) {
        size_t start = ptr[i];
        size_t end = ptr[i + 1];
        std::pmr::vector<std::pair<ptrdiff_t, double>> pairs(ar);
        pairs.reserve(end - start);
        for (size_t j = start; j < end; ++j) pairs.emplace_back(col[j], -val[j]);
        std::sort(pairs.begin(), pairs.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        for (size_t j = 0; j < pairs.size(); ++j) {
            col[start + j] = pairs[j].first;
            val[start + j] = -pairs[j].second;
        }
    }

    std::vector<double> bneg(n);
    for (size_t i = 0; i < n; ++i) bneg[i] = -rhs[i];

    typedef amgcl::backend::builtin<double, ptrdiff_t, ptrdiff_t> Backend;
    typedef amgcl::make_solver<
        amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>,
        amgcl::solver::cg<Backend>
    > Solver;

    typename Solver::params prm;
    prm.solver.maxiter = static_cast<size_t>(max_iter);
    prm.solver.tol = tol;

    Solver solver(std::tie(n, ptr, col, val), prm);
    std::vector<double> x(n, 0.0);
    auto [iters, err] = solver(bneg, x);

    if (iters >= prm.solver.maxiter) {
        std::cerr << "[josecast_core] NS pressure solve did not converge: iters="
                  << iters << " err=" << err << "\n";
    }
    return x;
}

struct Vec3 { double x, y, z; };

class NavierStokesVOF {
public:
    NavierStokesVOF(int nx, int ny, int nz, double dx,
                    const uint8_t* grid,
                    const double* phi_init,
                    const uint8_t* source,
                    std::array<double, 3> g,
                    double rho, double nu, double inflow_velocity,
                    double gravity_magnitude = 9.81)
        : nx_(nx), ny_(ny), nz_(nz), dx_(dx), rho_(rho), nu_(nu),
          inflow_(inflow_velocity), grav_(gravity_magnitude),
          arena_(make_arena(nx, ny, nz)),
          solid_(arena_.get()), source_(arena_.get()),
          phi_(arena_.get()), phi_new_(arena_.get()), phi_prev_(arena_.get()),
          u_(arena_.get()), v_(arena_.get()), w_(arena_.get()),
          u_prev_(arena_.get()), v_prev_(arena_.get()), w_prev_(arena_.get()),
          p_(arena_.get()), fill_time_(arena_.get())
    {
        const size_t n = static_cast<size_t>(nx_) * ny_ * nz_;
        solid_.assign(n, 0);
        source_.assign(n, 0);
        phi_.resize(n);
        phi_new_.resize(n);
        fill_time_.assign(n, std::numeric_limits<double>::infinity());

        for (size_t i = 0; i < n; ++i) {
            solid_[i] = (grid[i] == 0 || grid[i] == 9) ? 1 : 0;
            source_[i] = source[i] ? 1 : 0;
            // The input phi_init is a signed-distance field; convert it to a
            // colour function (0 = air, 1 = metal) for VOF-style advection.
            double c = (phi_init[i] <= 0.0) ? 1.0 : 0.0;
            if (solid_[i]) c = 0.0;
            if (source_[i]) c = 1.0;
            phi_[i] = c;
            if (c >= 0.5) {
                fill_time_[i] = source_[i] ? 0.0 : 0.0;
            }
        }

        double gnorm = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
        if (gnorm < 1e-12) gnorm = 1.0;
        g_[0] = g[0] / gnorm;
        g_[1] = g[1] / gnorm;
        g_[2] = g[2] / gnorm;

        u_.assign(static_cast<size_t>(nx_ + 1) * ny_ * nz_, 0.0); // x-face, x-velocity
        v_.assign(static_cast<size_t>(nx_) * (ny_ + 1) * nz_, 0.0); // y-face, y-velocity
        w_.assign(static_cast<size_t>(nx_) * ny_ * (nz_ + 1), 0.0); // z-face, z-velocity
        u_prev_ = u_;
        v_prev_ = v_;
        w_prev_ = w_;
        phi_prev_ = phi_;
        p_.assign(n, 0.0);
    }

    void run(double t_max, int max_steps, double cfl,
             int max_pressure_iter, double pressure_tol)
    {
        set_solid_boundary();
        set_inflow();

        while (t_ < t_max && steps_ < max_steps) {
            double dt = compute_dt(cfl);
            if (dt < 1e-9) break;
            if (t_ + dt > t_max) dt = t_max - t_;

            add_forces(dt);
            advect_velocity(dt);
            set_solid_boundary();
            set_inflow();

            project(dt, max_pressure_iter, pressure_tol);
            set_solid_boundary();
            set_inflow();

            advect_phi(dt);
            t_ += dt;
            steps_++;

            if (filled_fraction() >= 0.9999) {
                success_ = true;
                break;
            }
        }

        if (t_ >= t_max - 1e-9) success_ = true;
    }

    const std::pmr::vector<double>& phi() const { return phi_; }
    const std::pmr::vector<double>& fill_time() const { return fill_time_; }
    int steps() const { return steps_; }
    double current_time() const { return t_; }
    bool success() const { return success_; }

    double filled_fraction() const {
        size_t total = 0, filled = 0;
        const size_t n = static_cast<size_t>(nx_) * ny_ * nz_;
        for (size_t i = 0; i < n; ++i) {
            if (solid_[i]) continue;
            ++total;
            if (phi_[i] >= 0.5) ++filled;
        }
        return total ? static_cast<double>(filled) / static_cast<double>(total) : 0.0;
    }

    void get_velocity_magnitude(std::vector<double>* out) const {
        out->resize(static_cast<size_t>(nx_) * ny_ * nz_);
        for (int x = 0; x < nx_; ++x) {
            for (int y = 0; y < ny_; ++y) {
                for (int z = 0; z < nz_; ++z) {
                    double vx = 0.5 * (u_[cidx(x, y, z, ny_, nz_)] + u_[cidx(x + 1, y, z, ny_, nz_)]);
                    double vy = 0.5 * (v_[cidx(x, y, z, ny_ + 1, nz_)] + v_[cidx(x, y + 1, z, ny_ + 1, nz_)]);
                    double vz = 0.5 * (w_[cidx(x, y, z, ny_, nz_ + 1)] + w_[cidx(x, y, z + 1, ny_, nz_ + 1)]);
                    (*out)[cidx(x, y, z, ny_, nz_)] = std::sqrt(vx * vx + vy * vy + vz * vz);
                }
            }
        }
    }

    void get_velocity(std::vector<double>* out) const {
        const size_t n = static_cast<size_t>(nx_) * ny_ * nz_;
        out->resize(3 * n);
        for (int x = 0; x < nx_; ++x) {
            for (int y = 0; y < ny_; ++y) {
                for (int z = 0; z < nz_; ++z) {
                    size_t i = cidx(x, y, z, ny_, nz_);
                    double vx = 0.5 * (u_[cidx(x, y, z, ny_, nz_)] + u_[cidx(x + 1, y, z, ny_, nz_)]);
                    double vy = 0.5 * (v_[cidx(x, y, z, ny_ + 1, nz_)] + v_[cidx(x, y + 1, z, ny_ + 1, nz_)]);
                    double vz = 0.5 * (w_[cidx(x, y, z, ny_, nz_ + 1)] + w_[cidx(x, y, z + 1, ny_, nz_ + 1)]);
                    (*out)[i] = vx;
                    (*out)[i + n] = vy;
                    (*out)[i + 2 * n] = vz;
                }
            }
        }
    }

private:
    int nx_, ny_, nz_;
    double dx_, rho_, nu_, inflow_, grav_;
    std::array<double, 3> g_;

    std::unique_ptr<VirtualArena> arena_;
    std::pmr::vector<char> solid_, source_;
    std::pmr::vector<double> phi_, phi_new_, phi_prev_;
    std::pmr::vector<double> u_, v_, w_, u_prev_, v_prev_, w_prev_;
    std::pmr::vector<double> p_, fill_time_;
    double t_ = 0.0;
    int steps_ = 0;
    bool success_ = false;

    static std::unique_ptr<VirtualArena> make_arena(int nx, int ny, int nz) {
        size_t n = static_cast<size_t>(nx) * ny * nz;
        const std::size_t multipliers[] = {200, 150, 120, 100, 80};
        for (std::size_t m : multipliers) {
            try {
                return std::make_unique<VirtualArena>(VirtualArena::recommended(n, m));
            } catch (const std::bad_alloc&) {
                continue;
            }
        }
        throw std::bad_alloc();
    }

    inline bool is_solid(int x, int y, int z) const {
        if (!in_cell(x, y, z, nx_, ny_, nz_)) return true;
        return solid_[cidx(x, y, z, ny_, nz_)];
    }

    inline bool is_source(int x, int y, int z) const {
        if (!in_cell(x, y, z, nx_, ny_, nz_)) return false;
        return source_[cidx(x, y, z, ny_, nz_)];
    }

    void set_solid_boundary() {
        // u (x-face): between cells (x-1,y,z) and (x,y,z)
        for (int x = 0; x <= nx_; ++x) {
            for (int y = 0; y < ny_; ++y) {
                for (int z = 0; z < nz_; ++z) {
                    size_t idx = cidx(x, y, z, ny_, nz_);
                    if (x == 0 || x == nx_ || is_solid(x - 1, y, z) || is_solid(x, y, z)) {
                        u_[idx] = 0.0;
                    }
                }
            }
        }
        // v (y-face): between cells (x,y-1,z) and (x,y,z)
        for (int x = 0; x < nx_; ++x) {
            for (int y = 0; y <= ny_; ++y) {
                for (int z = 0; z < nz_; ++z) {
                    size_t idx = cidx(x, y, z, ny_ + 1, nz_);
                    if (y == 0 || y == ny_ || is_solid(x, y - 1, z) || is_solid(x, y, z)) {
                        v_[idx] = 0.0;
                    }
                }
            }
        }
        // w (z-face): between cells (x,y,z-1) and (x,y,z)
        for (int x = 0; x < nx_; ++x) {
            for (int y = 0; y < ny_; ++y) {
                for (int z = 0; z <= nz_; ++z) {
                    size_t idx = cidx(x, y, z, ny_, nz_ + 1);
                    if (z == 0 || z == nz_ || is_solid(x, y, z - 1) || is_solid(x, y, z)) {
                        w_[idx] = 0.0;
                    }
                }
            }
        }
    }

    void set_inflow() {
        double ux = inflow_ * g_[0];
        double vy = inflow_ * g_[1];
        double wz = inflow_ * g_[2];

        // u (x-face)
        for (int x = 0; x <= nx_; ++x) {
            for (int y = 0; y < ny_; ++y) {
                for (int z = 0; z < nz_; ++z) {
                    if (x == 0 || x == nx_) continue;
                    if ((is_source(x - 1, y, z) || is_source(x, y, z)) &&
                        !is_solid(x - 1, y, z) && !is_solid(x, y, z)) {
                        u_[cidx(x, y, z, ny_, nz_)] = ux;
                    }
                }
            }
        }
        // v (y-face)
        for (int x = 0; x < nx_; ++x) {
            for (int y = 0; y <= ny_; ++y) {
                for (int z = 0; z < nz_; ++z) {
                    if (y == 0 || y == ny_) continue;
                    if ((is_source(x, y - 1, z) || is_source(x, y, z)) &&
                        !is_solid(x, y - 1, z) && !is_solid(x, y, z)) {
                        v_[cidx(x, y, z, ny_ + 1, nz_)] = vy;
                    }
                }
            }
        }
        // w (z-face)
        for (int x = 0; x < nx_; ++x) {
            for (int y = 0; y < ny_; ++y) {
                for (int z = 0; z <= nz_; ++z) {
                    if (z == 0 || z == nz_) continue;
                    if ((is_source(x, y, z - 1) || is_source(x, y, z)) &&
                        !is_solid(x, y, z - 1) && !is_solid(x, y, z)) {
                        w_[cidx(x, y, z, ny_, nz_ + 1)] = wz;
                    }
                }
            }
        }
    }

    Vec3 sample_velocity(double x, double y, double z) const {
        double vx = trilerp(u_, nx_ + 1, ny_, nz_, x, y, z);
        double vy = trilerp(v_, nx_, ny_ + 1, nz_, x, y, z);
        double vz = trilerp(w_, nx_, ny_, nz_ + 1, x, y, z);
        return {vx, vy, vz};
    }

    void add_forces(double dt) {
        double ax = grav_ * g_[0];
        double ay = grav_ * g_[1];
        double az = grav_ * g_[2];

        // u (x-face)
        for (int x = 0; x <= nx_; ++x) {
            for (int y = 0; y < ny_; ++y) {
                for (int z = 0; z < nz_; ++z) {
                    if (x == 0 || x == nx_) continue;
                    if (!is_solid(x - 1, y, z) || !is_solid(x, y, z)) {
                        u_[cidx(x, y, z, ny_, nz_)] += dt * ax;
                    }
                }
            }
        }
        // v (y-face)
        for (int x = 0; x < nx_; ++x) {
            for (int y = 0; y <= ny_; ++y) {
                for (int z = 0; z < nz_; ++z) {
                    if (y == 0 || y == ny_) continue;
                    if (!is_solid(x, y - 1, z) || !is_solid(x, y, z)) {
                        v_[cidx(x, y, z, ny_ + 1, nz_)] += dt * ay;
                    }
                }
            }
        }
        // w (z-face)
        for (int x = 0; x < nx_; ++x) {
            for (int y = 0; y < ny_; ++y) {
                for (int z = 0; z <= nz_; ++z) {
                    if (z == 0 || z == nz_) continue;
                    if (!is_solid(x, y, z - 1) || !is_solid(x, y, z)) {
                        w_[cidx(x, y, z, ny_, nz_ + 1)] += dt * az;
                    }
                }
            }
        }
    }

    void advect_velocity(double dt) {
        u_prev_ = u_;
        v_prev_ = v_;
        w_prev_ = w_;

        double s = dt / dx_;
        // u (x-face)
        for (int x = 0; x <= nx_; ++x) {
            for (int y = 0; y < ny_; ++y) {
                for (int z = 0; z < nz_; ++z) {
                    size_t idx = cidx(x, y, z, ny_, nz_);
                    if (x == 0 || x == nx_ || (is_solid(x - 1, y, z) && is_solid(x, y, z))) {
                        u_[idx] = 0.0; continue;
                    }
                    Vec3 vel = sample_velocity(static_cast<double>(x), static_cast<double>(y), static_cast<double>(z));
                    double px = static_cast<double>(x) - s * vel.x;
                    double py = static_cast<double>(y) - s * vel.y;
                    double pz = static_cast<double>(z) - s * vel.z;
                    u_[idx] = trilerp(u_prev_, nx_ + 1, ny_, nz_, px, py, pz);
                }
            }
        }
        // v (y-face)
        for (int x = 0; x < nx_; ++x) {
            for (int y = 0; y <= ny_; ++y) {
                for (int z = 0; z < nz_; ++z) {
                    size_t idx = cidx(x, y, z, ny_ + 1, nz_);
                    if (y == 0 || y == ny_ || (is_solid(x, y - 1, z) && is_solid(x, y, z))) {
                        v_[idx] = 0.0; continue;
                    }
                    Vec3 vel = sample_velocity(static_cast<double>(x), static_cast<double>(y), static_cast<double>(z));
                    double px = static_cast<double>(x) - s * vel.x;
                    double py = static_cast<double>(y) - s * vel.y;
                    double pz = static_cast<double>(z) - s * vel.z;
                    v_[idx] = trilerp(v_prev_, nx_, ny_ + 1, nz_, px, py, pz);
                }
            }
        }
        // w (z-face)
        for (int x = 0; x < nx_; ++x) {
            for (int y = 0; y < ny_; ++y) {
                for (int z = 0; z <= nz_; ++z) {
                    size_t idx = cidx(x, y, z, ny_, nz_ + 1);
                    if (z == 0 || z == nz_ || (is_solid(x, y, z - 1) && is_solid(x, y, z))) {
                        w_[idx] = 0.0; continue;
                    }
                    Vec3 vel = sample_velocity(static_cast<double>(x), static_cast<double>(y), static_cast<double>(z));
                    double px = static_cast<double>(x) - s * vel.x;
                    double py = static_cast<double>(y) - s * vel.y;
                    double pz = static_cast<double>(z) - s * vel.z;
                    w_[idx] = trilerp(w_prev_, nx_, ny_, nz_ + 1, px, py, pz);
                }
            }
        }
    }

    void advect_phi(double dt) {
        phi_prev_ = phi_;
        const size_t n = static_cast<size_t>(nx_) * ny_ * nz_;
        phi_new_.assign(n, 0.0);

        double s = dt / dx_;
        for (int x = 0; x < nx_; ++x) {
            for (int y = 0; y < ny_; ++y) {
                for (int z = 0; z < nz_; ++z) {
                    size_t i = cidx(x, y, z, ny_, nz_);
                    if (solid_[i]) continue;
                    if (source_[i]) {
                        phi_new_[i] = 1.0;
                        continue;
                    }
                    Vec3 vel = sample_velocity(static_cast<double>(x) + 0.5,
                                               static_cast<double>(y) + 0.5,
                                               static_cast<double>(z) + 0.5);
                    double px = static_cast<double>(x) + 0.5 - s * vel.x;
                    double py = static_cast<double>(y) + 0.5 - s * vel.y;
                    double pz = static_cast<double>(z) + 0.5 - s * vel.z;
                    double c = trilerp(phi_prev_, nx_, ny_, nz_, px, py, pz);
                    phi_new_[i] = std::clamp(c, 0.0, 1.0);
                }
            }
        }

        // Update fill time for cells that crossed into metal this step.
        for (size_t i = 0; i < n; ++i) {
            if (solid_[i] || source_[i]) continue;
            if (phi_new_[i] >= 0.5 && std::isinf(fill_time_[i])) {
                if (phi_prev_[i] >= 0.5) {
                    fill_time_[i] = t_;
                } else {
                    double denom = phi_new_[i] - phi_prev_[i];
                    double theta = (denom > 1e-18) ? (0.5 - phi_prev_[i]) / denom : 1.0;
                    fill_time_[i] = t_ + theta * dt;
                }
            }
        }

        phi_.swap(phi_new_);
    }

    void project(double dt, int max_iter, double tol) {
        const size_t n = static_cast<size_t>(nx_) * ny_ * nz_;
        std::pmr::memory_resource* ar = arena_.get();

        std::pmr::vector<int> flat(n, -1, ar);
        size_t nunk = 0;
        for (int x = 0; x < nx_; ++x) {
            for (int y = 0; y < ny_; ++y) {
                for (int z = 0; z < nz_; ++z) {
                    size_t i = cidx(x, y, z, ny_, nz_);
                    if (!solid_[i] && phi_[i] >= 0.5) {
                        flat[i] = static_cast<int>(nunk++);
                    }
                }
            }
        }
        if (nunk == 0) return;

        std::pmr::vector<int> rows(ar), cols(ar);
        std::pmr::vector<double> data(ar);
        std::pmr::vector<double> rhs(nunk, 0.0, ar);
        const double inv_dx2 = 1.0 / (dx_ * dx_);
        const double alpha = 1e-6;

        auto add_neighbor = [&](int c, int x2, int y2, int z2, double& diag) {
            if (!in_cell(x2, y2, z2, nx_, ny_, nz_)) return;
            size_t ni = cidx(x2, y2, z2, ny_, nz_);
            if (solid_[ni]) return;
            diag += inv_dx2;
            if (phi_[ni] >= 0.5) {
                rows.push_back(c);
                cols.push_back(flat[ni]);
                data.push_back(inv_dx2);
            }
        };

        for (int x = 0; x < nx_; ++x) {
            for (int y = 0; y < ny_; ++y) {
                for (int z = 0; z < nz_; ++z) {
                    size_t i = cidx(x, y, z, ny_, nz_);
                    int c = flat[i];
                    if (c < 0) continue;

                    double diag = alpha * inv_dx2;
                    double div = 0.0;
                    div += (u_[cidx(x + 1, y, z, ny_, nz_)] - u_[cidx(x, y, z, ny_, nz_)]);
                    div += (v_[cidx(x, y + 1, z, ny_ + 1, nz_)] - v_[cidx(x, y, z, ny_ + 1, nz_)]);
                    div += (w_[cidx(x, y, z + 1, ny_, nz_ + 1)] - w_[cidx(x, y, z, ny_, nz_ + 1)]);
                    div /= dx_;

                    add_neighbor(c, x + 1, y, z, diag);
                    add_neighbor(c, x - 1, y, z, diag);
                    add_neighbor(c, x, y + 1, z, diag);
                    add_neighbor(c, x, y - 1, z, diag);
                    add_neighbor(c, x, y, z + 1, diag);
                    add_neighbor(c, x, y, z - 1, diag);

                    rows.push_back(c);
                    cols.push_back(c);
                    data.push_back(-diag);
                    rhs[c] = rho_ / dt * div;
                }
            }
        }

        // Zero-mean RHS for better convergence in sealed / Dirichlet-free regions.
        double rmean = 0.0;
        for (double v : rhs) rmean += v;
        rmean /= static_cast<double>(nunk);
        for (double& v : rhs) v -= rmean;

        auto p_sol = amgcl_solve_cg(rows, cols, data, rhs, ar, max_iter, tol);
        p_.assign(p_sol.begin(), p_sol.end());

        const double scale = dt / (rho_ * dx_);
        // u (x-face)
        for (int x = 0; x <= nx_; ++x) {
            for (int y = 0; y < ny_; ++y) {
                for (int z = 0; z < nz_; ++z) {
                    if (x == 0 || x == nx_) continue;
                    if (is_solid(x - 1, y, z) || is_solid(x, y, z)) continue;
                    double pL = 0.0, pR = 0.0;
                    size_t li = cidx(x - 1, y, z, ny_, nz_);
                    size_t ri = cidx(x, y, z, ny_, nz_);
                    if (phi_[li] >= 0.5 && flat[li] >= 0) pL = p_[flat[li]];
                    if (phi_[ri] >= 0.5 && flat[ri] >= 0) pR = p_[flat[ri]];
                    u_[cidx(x, y, z, ny_, nz_)] -= scale * (pR - pL);
                }
            }
        }
        // v (y-face)
        for (int x = 0; x < nx_; ++x) {
            for (int y = 0; y <= ny_; ++y) {
                for (int z = 0; z < nz_; ++z) {
                    if (y == 0 || y == ny_) continue;
                    if (is_solid(x, y - 1, z) || is_solid(x, y, z)) continue;
                    double pB = 0.0, pT = 0.0;
                    size_t bi = cidx(x, y - 1, z, ny_, nz_);
                    size_t ti = cidx(x, y, z, ny_, nz_);
                    if (phi_[bi] >= 0.5 && flat[bi] >= 0) pB = p_[flat[bi]];
                    if (phi_[ti] >= 0.5 && flat[ti] >= 0) pT = p_[flat[ti]];
                    v_[cidx(x, y, z, ny_ + 1, nz_)] -= scale * (pT - pB);
                }
            }
        }
        // w (z-face)
        for (int x = 0; x < nx_; ++x) {
            for (int y = 0; y < ny_; ++y) {
                for (int z = 0; z <= nz_; ++z) {
                    if (z == 0 || z == nz_) continue;
                    if (is_solid(x, y, z - 1) || is_solid(x, y, z)) continue;
                    double pB = 0.0, pF = 0.0;
                    size_t bi = cidx(x, y, z - 1, ny_, nz_);
                    size_t fi = cidx(x, y, z, ny_, nz_);
                    if (phi_[bi] >= 0.5 && flat[bi] >= 0) pB = p_[flat[bi]];
                    if (phi_[fi] >= 0.5 && flat[fi] >= 0) pF = p_[flat[fi]];
                    w_[cidx(x, y, z, ny_, nz_ + 1)] -= scale * (pF - pB);
                }
            }
        }
    }

    double compute_dt(double cfl) const {
        double vmax = 0.0;
        for (double val : u_) vmax = std::max(vmax, std::abs(val));
        for (double val : v_) vmax = std::max(vmax, std::abs(val));
        for (double val : w_) vmax = std::max(vmax, std::abs(val));

        double dt = 0.01;
        if (vmax > 1e-12) dt = cfl * dx_ / vmax;

        if (nu_ > 1e-12) {
            double dt_visc = 0.2 * dx_ * dx_ / nu_;
            dt = std::min(dt, dt_visc);
        }
        return std::max(dt, 1e-6);
    }
};

} // namespace

nb::tuple solve_ns_vof(
    nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, -1>> grid,
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> phi_init,
    nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, -1>> source_mask,
    double dx,
    std::array<double, 3> g,
    double rho,
    double nu,
    double inflow_velocity,
    double t_max,
    int max_steps,
    double cfl,
    int max_pressure_iter,
    double pressure_tol)
{
    int nx = static_cast<int>(grid.shape(0));
    int ny = static_cast<int>(grid.shape(1));
    int nz = static_cast<int>(grid.shape(2));

    if (phi_init.shape(0) != nx || phi_init.shape(1) != ny || phi_init.shape(2) != nz ||
        source_mask.shape(0) != nx || source_mask.shape(1) != ny || source_mask.shape(2) != nz) {
        throw std::runtime_error("solve_ns_vof: input shapes do not match");
    }

    NavierStokesVOF solver(nx, ny, nz, dx, grid.data(), phi_init.data(), source_mask.data(),
                           g, rho, nu, inflow_velocity, 9.81);

    solver.run(t_max, max_steps, cfl, max_pressure_iter, pressure_tol);

    std::vector<double> phi_out(solver.phi().begin(), solver.phi().end());
    std::vector<double> ft_out(solver.fill_time().begin(), solver.fill_time().end());
    std::vector<double> vmag_out, vel_out;
    solver.get_velocity_magnitude(&vmag_out);
    solver.get_velocity(&vel_out);

    const size_t n = static_cast<size_t>(nx) * ny * nz;

    auto* phi_vec = new std::vector<double>(std::move(phi_out));
    auto* ft_vec = new std::vector<double>(std::move(ft_out));
    auto* vmag_vec = new std::vector<double>(std::move(vmag_out));
    auto* vel_vec = new std::vector<double>(std::move(vel_out));

    auto cap_phi = nb::capsule(phi_vec, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    auto cap_ft = nb::capsule(ft_vec, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    auto cap_vmag = nb::capsule(vmag_vec, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    auto cap_vel = nb::capsule(vel_vec, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });

    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> arr_phi(
        phi_vec->data(), {static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz)}, cap_phi);
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> arr_ft(
        ft_vec->data(), {static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz)}, cap_ft);
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> arr_vmag(
        vmag_vec->data(), {static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz)}, cap_vmag);
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> arr_vel(
        vel_vec->data(), {3, static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz)}, cap_vel);

    return nb::make_tuple(
        arr_ft,
        arr_vmag,
        arr_vel,
        arr_phi,
        solver.success(),
        solver.current_time(),
        solver.filled_fraction(),
        solver.steps()
    );
}

} // namespace josecast
