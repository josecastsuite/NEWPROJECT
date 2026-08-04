#include "josecast/spline_tube_solver.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <nanoflann.hpp>

namespace nb = nanobind;

namespace josecast {

namespace {

constexpr double MM_TO_M = 1.0e-3;
constexpr double M2_TO_MM2 = 1.0e6;
constexpr double J_PI = 3.141592653589793238462643383279502884;

inline double clamp(double x, double lo, double hi) {
    return (x < lo) ? lo : (x > hi ? hi : x);
}

inline double norm2(double x, double y, double z) {
    return std::sqrt(x * x + y * y + z * z);
}

// ----------------------------------------------------------------------------
// 1-D natural cubic spline for non-uniform knots.
// ----------------------------------------------------------------------------
class NaturalCubicSpline1D {
public:
    NaturalCubicSpline1D() = default;
    NaturalCubicSpline1D(const std::vector<double>& u,
                         const std::vector<double>& y)
        : u_(u), y_(y) {
        const int n = static_cast<int>(u_.size());
        if (n < 2) return;
        h_.resize(n - 1);
        for (int i = 0; i < n - 1; ++i) h_[i] = u_[i + 1] - u_[i];
        M_.assign(n, 0.0);
        if (n < 3) return;

        std::vector<double> a(n), b(n), c(n), d(n);
        // natural boundary
        b[0] = 1.0; c[0] = 0.0; d[0] = 0.0;
        a[n - 1] = 0.0; b[n - 1] = 1.0; d[n - 1] = 0.0;
        for (int i = 1; i < n - 1; ++i) {
            a[i] = h_[i - 1];
            b[i] = 2.0 * (h_[i - 1] + h_[i]);
            c[i] = h_[i];
            d[i] = 6.0 * ((y_[i + 1] - y_[i]) / h_[i] -
                          (y_[i] - y_[i - 1]) / h_[i - 1]);
        }
        // Thomas algorithm
        for (int i = 1; i < n; ++i) {
            double w = a[i] / b[i - 1];
            b[i] -= w * c[i - 1];
            d[i] -= w * d[i - 1];
        }
        M_[n - 1] = d[n - 1] / b[n - 1];
        for (int i = n - 2; i >= 0; --i) {
            M_[i] = (d[i] - c[i] * M_[i + 1]) / b[i];
        }
    }

    int find_segment(double uq) const {
        const int n = static_cast<int>(u_.size());
        if (n < 2) return 0;
        if (uq <= u_.front()) return 0;
        if (uq >= u_.back()) return n - 2;
        auto it = std::upper_bound(u_.begin(), u_.end(), uq);
        return static_cast<int>(it - u_.begin()) - 1;
    }

    double eval(double uq) const {
        if (u_.size() < 2) return y_.empty() ? 0.0 : y_[0];
        const int i = find_segment(uq);
        const double h = h_[i];
        if (h <= 0.0) return y_[i];
        const double A = (u_[i + 1] - uq) / h;
        const double B = (uq - u_[i]) / h;
        return A * y_[i] + B * y_[i + 1] +
               (A * A * A - A) * M_[i] * h * h / 6.0 +
               (B * B * B - B) * M_[i + 1] * h * h / 6.0;
    }

    double deriv(double uq) const {
        if (u_.size() < 2) return 0.0;
        const int i = find_segment(uq);
        const double h = h_[i];
        if (h <= 0.0) return 0.0;
        const double A = (u_[i + 1] - uq) / h;
        const double B = (uq - u_[i]) / h;
        return (y_[i + 1] - y_[i]) / h +
               (-(3.0 * A * A - 1.0) * M_[i] +
                (3.0 * B * B - 1.0) * M_[i + 1]) * h / 6.0;
    }

private:
    std::vector<double> u_, y_, h_, M_;
};

// ----------------------------------------------------------------------------
// 3-D natural cubic spline through node centroids.
// ----------------------------------------------------------------------------
class NaturalCubicSpline3D {
public:
    explicit NaturalCubicSpline3D(const std::vector<std::array<double, 3>>& pts)
        : pts_(pts) {
        const int n = static_cast<int>(pts_.size());
        if (n < 2) return;
        u_.resize(n);
        u_[0] = 0.0;
        for (int i = 1; i < n; ++i) {
            const auto& a = pts_[i - 1];
            const auto& b = pts_[i];
            u_[i] = u_[i - 1] + norm2(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
        }
        const double L = u_.back();
        if (L > 0.0) {
            for (double& v : u_) v /= L;
        }
        std::vector<double> x(n), y(n), z(n);
        for (int i = 0; i < n; ++i) {
            x[i] = pts_[i][0];
            y[i] = pts_[i][1];
            z[i] = pts_[i][2];
        }
        sx_ = NaturalCubicSpline1D(u_, x);
        sy_ = NaturalCubicSpline1D(u_, y);
        sz_ = NaturalCubicSpline1D(u_, z);
    }

    std::array<double, 3> eval(double u) const {
        return {sx_.eval(u), sy_.eval(u), sz_.eval(u)};
    }

    std::array<double, 3> deriv(double u) const {
        return {sx_.deriv(u), sy_.deriv(u), sz_.deriv(u)};
    }

    const std::vector<double>& u() const { return u_; }

private:
    std::vector<std::array<double, 3>> pts_;
    std::vector<double> u_;
    NaturalCubicSpline1D sx_, sy_, sz_;
};

// ----------------------------------------------------------------------------
// Trilinear SDF sampler.
// ----------------------------------------------------------------------------
struct SDFGrid {
    const double* data;
    int nx, ny, nz;
    double ox, oy, oz; // mm
    double dx;         // mm

    inline bool in_bounds(int ix, int iy, int iz) const {
        return ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz;
    }

    inline double get(int ix, int iy, int iz) const {
        return data[static_cast<size_t>((ix * ny + iy) * nz + iz)];
    }

    double sample(double x, double y, double z) const {
        double xi = (x - ox) / dx - 0.5;
        double yi = (y - oy) / dx - 0.5;
        double zi = (z - oz) / dx - 0.5;
        int ix0 = static_cast<int>(std::floor(xi));
        int iy0 = static_cast<int>(std::floor(yi));
        int iz0 = static_cast<int>(std::floor(zi));
        double fx = xi - ix0;
        double fy = yi - iy0;
        double fz = zi - iz0;

        double val = 0.0;
        double wsum = 0.0;
        for (int i = 0; i < 2; ++i) {
            int ix = ix0 + i;
            double wx = (i == 0) ? (1.0 - fx) : fx;
            for (int j = 0; j < 2; ++j) {
                int iy = iy0 + j;
                double wy = (j == 0) ? (1.0 - fy) : fy;
                for (int k = 0; k < 2; ++k) {
                    int iz = iz0 + k;
                    double wz = (k == 0) ? (1.0 - fz) : fz;
                    if (in_bounds(ix, iy, iz)) {
                        double w = wx * wy * wz;
                        val += w * get(ix, iy, iz);
                        wsum += w;
                    }
                }
            }
        }
        if (wsum <= 0.0) return -1.0;
        return val / wsum;
    }
};

// ----------------------------------------------------------------------------
// Branch discretised into spline samples, one nanoflann tree per branch.
// ----------------------------------------------------------------------------
struct BranchSamples {
    std::vector<double> s;            // arc length (mm)
    std::vector<double> x, y, z;      // position (mm)
    std::vector<double> tx, ty, tz;   // unit tangent
    std::vector<double> R;            // effective radius (mm)
    std::vector<double> v_bulk;       // calibrated axis velocity (m/s)
    std::vector<double> v_raw;        // Q / A_eff (m/s) before calibration

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> points;
    std::unique_ptr<nanoflann::KDTreeEigenMatrixAdaptor<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
        3, nanoflann::metric_L2, true>> kdtree;
};

void build_branch_samples(
    const std::vector<int>& node_indices,
    const std::vector<std::array<double, 3>>& node_centroids,
    const std::vector<double>& node_velocity,
    const std::vector<double>& node_area,
    const SDFGrid& sdf,
    double dx,
    BranchSamples& out) {

    const int n_nodes = static_cast<int>(node_indices.size());
    if (n_nodes < 1) return;

    std::vector<std::array<double, 3>> pts;
    for (int i = 0; i < n_nodes; ++i) {
        pts.push_back(node_centroids[node_indices[i]]);
    }
    NaturalCubicSpline3D spline(pts);

    // Degenerate single-node branch: one sample at the node position.
    if (n_nodes == 1) {
        out.x.resize(1);
        out.y.resize(1);
        out.z.resize(1);
        out.tx.resize(1);
        out.ty.resize(1);
        out.tz.resize(1);
        out.s.resize(1);
        out.R.resize(1);
        out.v_bulk.resize(1);
        out.v_raw.resize(1);
        out.x[0] = pts[0][0];
        out.y[0] = pts[0][1];
        out.z[0] = pts[0][2];
        out.tx[0] = 0.0;
        out.ty[0] = 0.0;
        out.tz[0] = 1.0;
        out.s[0] = 0.0;
        double A = std::max(0.0, node_area[node_indices[0]]);
        double R_hydr = std::sqrt(A / J_PI) * 1000.0;
        auto pos = pts[0];
        double R_sdf = sdf.sample(pos[0], pos[1], pos[2]);
        double R_eff = R_hydr;
        if (std::isfinite(R_sdf) && R_sdf >= 0.5 * dx && R_sdf <= 2.0 * R_hydr) {
            R_eff = R_sdf;
        }
        out.R[0] = R_eff;
        out.v_bulk[0] = node_velocity[node_indices[0]];
        out.v_raw[0] = node_velocity[node_indices[0]];

        out.points.resize(1, 3);
        out.points(0, 0) = pos[0];
        out.points(0, 1) = pos[1];
        out.points(0, 2) = pos[2];
        using MatrixType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        out.kdtree = std::make_unique<nanoflann::KDTreeEigenMatrixAdaptor<MatrixType, 3>>(
            3, std::cref(out.points), 10);
        return;
    }

    // Physical chord length for sample density.
    double chord_length = 0.0;
    for (int i = 1; i < n_nodes; ++i) {
        chord_length += norm2(pts[i][0] - pts[i - 1][0],
                              pts[i][1] - pts[i - 1][1],
                              pts[i][2] - pts[i - 1][2]);
    }

    // Include node parameter values as samples so node s-values align exactly.
    const std::vector<double>& u_all = spline.u();
    std::vector<double> u_samples;
    int n_extra = 64;
    if (n_nodes > 1) {
        n_extra = std::max(64, static_cast<int>(std::ceil(chord_length / (0.5 * dx))));
        const double du = 1.0 / static_cast<double>(n_extra);
        for (int i = 0; i <= n_extra; ++i) u_samples.push_back(i * du);
    } else {
        u_samples.push_back(0.0);
    }
    for (double uu : u_all) u_samples.push_back(uu);
    std::sort(u_samples.begin(), u_samples.end());
    u_samples.erase(std::unique(u_samples.begin(), u_samples.end(),
                                [](double a, double b) { return std::abs(a - b) < 1e-12; }),
                    u_samples.end());

    // Node s-values and interpolated node area / velocity for this branch.
    // s computed by trapezoidal integration of |P'(u)|.
    std::vector<double> u_node = u_all;
    std::vector<double> s_node(n_nodes);
    std::vector<double> A_node_node(n_nodes);
    std::vector<double> v_node_node(n_nodes);
    double total_s = 0.0;
    s_node[0] = 0.0;
    for (int i = 0; i < n_nodes; ++i) {
        A_node_node[i] = node_area[node_indices[i]];
        v_node_node[i] = node_velocity[node_indices[i]];
    }
    for (int i = 1; i < n_nodes; ++i) {
        auto d0 = spline.deriv(0.5 * (u_node[i - 1] + u_node[i]));
        double mag = norm2(d0[0], d0[1], d0[2]);
        total_s += mag * (u_node[i] - u_node[i - 1]);
        s_node[i] = total_s;
    }
    if (total_s <= 0.0) total_s = 1.0;

    // Scale s_node to true cumulative length by sampling the spline.
    std::vector<double> s_sample_raw(u_samples.size(), 0.0);
    for (size_t k = 1; k < u_samples.size(); ++k) {
        double du = u_samples[k] - u_samples[k - 1];
        auto dmid = spline.deriv(0.5 * (u_samples[k] + u_samples[k - 1]));
        double mag = norm2(dmid[0], dmid[1], dmid[2]);
        s_sample_raw[k] = s_sample_raw[k - 1] + mag * du;
    }
    double spline_len = s_sample_raw.back();
    if (spline_len <= 0.0) spline_len = 1.0;

    // Map u_node to s_node using integration so the node labels sit at exact s.
    for (int i = 0; i < n_nodes; ++i) {
        // find u in u_samples and take corresponding s
        auto it = std::lower_bound(u_samples.begin(), u_samples.end(), u_node[i]);
        size_t k = std::min<size_t>(it - u_samples.begin(), u_samples.size() - 1);
        s_node[i] = s_sample_raw[k];
    }

    // Node flow rates (m^3/s) and helpers.
    std::vector<double> Q_node(n_nodes);
    for (int i = 0; i < n_nodes; ++i) {
        Q_node[i] = node_velocity[node_indices[i]] * node_area[node_indices[i]];
    }

    auto linear_node = [&](double s, const std::vector<double>& node_vals) -> double {
        if (n_nodes == 1) return node_vals[0];
        if (s <= s_node.front()) return node_vals[0];
        if (s >= s_node.back()) return node_vals.back();
        auto it = std::upper_bound(s_node.begin(), s_node.end(), s);
        int i = static_cast<int>(it - s_node.begin());
        int i0 = i - 1;
        int i1 = i;
        double t = (s - s_node[i0]) / (s_node[i1] - s_node[i0]);
        return node_vals[i0] * (1.0 - t) + node_vals[i1] * t;
    };

    // Segment downstream index for Q(s).  Q is conserved along a branch, but the
    // source node stores the *total* pre-split flow.  For any s past the source
    // use the first downstream node's Q (half of the source Q after a split).
    auto downstream_index = [&](double s) -> int {
        if (n_nodes == 1) return 0;
        if (s <= s_node[1]) return 1;
        if (s >= s_node.back()) return n_nodes - 1;
        auto it = std::upper_bound(s_node.begin() + 1, s_node.end(), s);
        return static_cast<int>(it - s_node.begin());
    };

    const size_t n_samples = u_samples.size();
    out.s.resize(n_samples);
    out.x.resize(n_samples);
    out.y.resize(n_samples);
    out.z.resize(n_samples);
    out.tx.resize(n_samples);
    out.ty.resize(n_samples);
    out.tz.resize(n_samples);
    out.R.resize(n_samples);
    out.v_bulk.resize(n_samples);
    out.v_raw.resize(n_samples);

    const double node_eps = 1e-6 * spline_len;

    for (size_t k = 0; k < n_samples; ++k) {
        double uu = u_samples[k];
        auto pos = spline.eval(uu);
        auto der = spline.deriv(uu);
        double mag = norm2(der[0], der[1], der[2]);
        if (mag > 1e-18) {
            out.tx[k] = der[0] / mag;
            out.ty[k] = der[1] / mag;
            out.tz[k] = der[2] / mag;
        } else {
            out.tx[k] = 1.0; out.ty[k] = 0.0; out.tz[k] = 0.0;
        }
        out.x[k] = pos[0];
        out.y[k] = pos[1];
        out.z[k] = pos[2];
        out.s[k] = s_sample_raw[k];

        // Effective cross-section is the node-derived hydraulic area interpolated
        // along the spline.  SDF may be used for the local wall radius, but the
        // bulk velocity v = Q / A is driven by the hydraulic area from the
        // gating-node data, which is robust for non-circular cross-sections.
        double s_val = out.s[k];
        double A_eff = linear_node(s_val, A_node_node);
        double R_hydr_mm = std::sqrt(std::max(0.0, A_eff / J_PI)) * 1000.0;

        double R_sdf = sdf.sample(pos[0], pos[1], pos[2]);
        double R_eff_mm = R_hydr_mm;
        if (std::isfinite(R_sdf) && R_sdf >= 0.5 * dx && R_sdf <= 2.0 * R_hydr_mm) {
            // Use the sampled SDF radius if it is plausible for this cross-section,
            // otherwise fall back to the hydraulic radius.
            R_eff_mm = R_sdf;
        }
        out.R[k] = R_eff_mm;

        // v = Q / A(s).  Q is the downstream flow rate for the current segment
        // so a pre-split source node does not inflate velocities after the split.
        double Q_s = Q_node[downstream_index(s_val)];
        double v_bulk = (A_eff > 1e-18) ? Q_s / A_eff : 0.0;

        // Force exact node velocities at the node samples.
        for (int i = 0; i < n_nodes; ++i) {
            if (std::abs(s_val - s_node[i]) < node_eps) {
                v_bulk = v_node_node[i];
                A_eff = A_node_node[i];
                R_eff_mm = std::sqrt(std::max(0.0, A_eff / J_PI)) * 1000.0;
                out.R[k] = R_eff_mm;
                break;
            }
        }

        out.v_raw[k] = v_bulk;
        out.v_bulk[k] = v_bulk;
        if (!std::isfinite(out.v_bulk[k])) out.v_bulk[k] = 0.0;
    }

    // Build KD-tree on sample points (RowMajor: each row = one point).
    out.points.resize(n_samples, 3);
    for (size_t k = 0; k < n_samples; ++k) {
        out.points(k, 0) = out.x[k];
        out.points(k, 1) = out.y[k];
        out.points(k, 2) = out.z[k];
    }
    using MatrixType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    out.kdtree = std::make_unique<nanoflann::KDTreeEigenMatrixAdaptor<MatrixType, 3>>(
        3, std::cref(out.points), 10);
}

} // namespace

// ----------------------------------------------------------------------------
// Public binding.
// ----------------------------------------------------------------------------
nb::tuple solve_spline_tube_field(
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> sdf,
    nb::ndarray<nb::numpy, int32_t, nb::shape<-1, -1, -1>> voxel_branch,
    double dx,
    std::array<double, 3> origin,
    nb::ndarray<nb::numpy, double, nb::shape<-1, 3>> node_centroids,
    nb::ndarray<nb::numpy, double, nb::shape<-1>> node_velocity,
    nb::ndarray<nb::numpy, double, nb::shape<-1>> node_area,
    nb::ndarray<nb::numpy, int32_t, nb::shape<-1>> branch_node_indices,
    nb::ndarray<nb::numpy, int32_t, nb::shape<-1>> branch_offsets) {

    int nx = static_cast<int>(sdf.shape(0));
    int ny = static_cast<int>(sdf.shape(1));
    int nz = static_cast<int>(sdf.shape(2));

    if (voxel_branch.shape(0) != nx || voxel_branch.shape(1) != ny || voxel_branch.shape(2) != nz) {
        throw std::runtime_error("solve_spline_tube_field: voxel_branch shape must match sdf");
    }

    const int n_nodes = static_cast<int>(node_centroids.shape(0));
    if (n_nodes == 0 || static_cast<int>(node_velocity.shape(0)) != n_nodes ||
        static_cast<int>(node_area.shape(0)) != n_nodes) {
        throw std::runtime_error("solve_spline_tube_field: node arrays size mismatch");
    }

    SDFGrid sdfg;
    sdfg.data = sdf.data();
    sdfg.nx = nx; sdfg.ny = ny; sdfg.nz = nz;
    sdfg.ox = origin[0]; sdfg.oy = origin[1]; sdfg.oz = origin[2];
    sdfg.dx = dx;

    std::vector<std::array<double, 3>> centroids(n_nodes);
    for (int i = 0; i < n_nodes; ++i) {
        centroids[i][0] = node_centroids(i, 0);
        centroids[i][1] = node_centroids(i, 1);
        centroids[i][2] = node_centroids(i, 2);
    }
    std::vector<double> velocities(n_nodes), areas(n_nodes);
    for (int i = 0; i < n_nodes; ++i) {
        velocities[i] = node_velocity(i);
        areas[i] = node_area(i);
    }

    // Build per-branch sample sets.
    const int n_branches = static_cast<int>(branch_offsets.shape(0) - 1);
    if (n_branches < 0) {
        throw std::runtime_error("solve_spline_tube_field: branch_offsets must contain at least 2 entries");
    }
    std::vector<std::unique_ptr<BranchSamples>> branches;
    for (int b = 0; b < n_branches; ++b) {
        int off0 = branch_offsets(b);
        int off1 = branch_offsets(b + 1);
        std::vector<int> bn;
        for (int k = off0; k < off1; ++k) {
            bn.push_back(branch_node_indices(k));
        }
        auto bs = std::make_unique<BranchSamples>();
        build_branch_samples(bn, centroids, velocities, areas, sdfg, dx, *bs);
        branches.push_back(std::move(bs));
    }

    const size_t n_total = static_cast<size_t>(nx) * ny * nz;
    auto bulk_vec = new std::vector<double>(n_total, 0.0);
    auto pois_vec = new std::vector<double>(n_total, 0.0);

    const int32_t* branch_ptr = voxel_branch.data();
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1024)
#endif
    for (int ix = 0; ix < nx; ++ix) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int iz = 0; iz < nz; ++iz) {
                size_t idx = static_cast<size_t>((ix * ny + iy) * nz + iz);
                int b = branch_ptr[idx];
                if (b < 0 || b >= static_cast<int>(branches.size())) {
                    continue;
                }
                const auto& bs = *branches[b];
                if (bs.kdtree == nullptr || bs.s.empty()) continue;

                double px = origin[0] + (ix + 0.5) * dx;
                double py = origin[1] + (iy + 0.5) * dx;
                double pz = origin[2] + (iz + 0.5) * dx;

                Eigen::Vector3d q(px, py, pz);
                Eigen::Index nearest;
                double dist_sq;
                bs.kdtree->index_->knnSearch(q.data(), 1, &nearest, &dist_sq);

                const double tx = bs.tx[nearest];
                const double ty = bs.ty[nearest];
                const double tz = bs.tz[nearest];
                const double sx = bs.x[nearest];
                const double sy = bs.y[nearest];
                const double sz = bs.z[nearest];

                double vx = px - sx;
                double vy = py - sy;
                double vz = pz - sz;
                double t_proj = vx * tx + vy * ty + vz * tz;

                // Find second sample in the direction of projection for linear interp.
                Eigen::Index next = nearest;
                if (t_proj >= 0.0) {
                    next = std::min<Eigen::Index>(nearest + 1, static_cast<Eigen::Index>(bs.s.size() - 1));
                } else if (nearest > 0) {
                    next = nearest - 1;
                }
                if (next == nearest && bs.s.size() > 1) {
                    next = (nearest == 0) ? 1 : nearest - 1;
                }
                double denom = bs.s[next] - bs.s[nearest];
                double alpha = 0.0;
                if (std::abs(denom) > 1e-12) {
                    // t_proj is the projected arc-length from the nearest sample.
                    // denom is the signed arc-length to the next sample in the same
                    // direction, so the ratio is valid for both forward/backward.
                    alpha = clamp(t_proj / denom, 0.0, 1.0);
                }

                auto lerp = [&](const std::vector<double>& v) -> double {
                    if (bs.s.size() == 1 || next == nearest) return v[nearest];
                    return v[nearest] * (1.0 - alpha) + v[next] * alpha;
                };

                double v_bulk = lerp(bs.v_bulk);
                double R_eff = lerp(bs.R);

                // Perpendicular distance
                double q_proj_x = sx + t_proj * tx;
                double q_proj_y = sy + t_proj * ty;
                double q_proj_z = sz + t_proj * tz;
                double r = norm2(px - q_proj_x, py - q_proj_y, pz - q_proj_z);

                bulk_vec->operator[](idx) = v_bulk;
                if (R_eff > 1e-9 && r < R_eff) {
                    double ratio = r / R_eff;
                    pois_vec->operator[](idx) = v_bulk * (1.0 - ratio * ratio);
                } else {
                    pois_vec->operator[](idx) = 0.0;
                }
            }
        }
    }

    auto cap_bulk = nb::capsule(bulk_vec, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    auto cap_pois = nb::capsule(pois_vec, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });

    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> arr_bulk(
        bulk_vec->data(), {static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz)}, cap_bulk);
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> arr_pois(
        pois_vec->data(), {static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz)}, cap_pois);

    return nb::make_tuple(arr_bulk, arr_pois);
}

} // namespace josecast
