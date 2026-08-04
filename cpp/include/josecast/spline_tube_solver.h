#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <array>
#include <cstdint>
#include <vector>

namespace nb = nanobind;

namespace josecast {

/**
 * Analytic conformal spline-tube velocity field.
 *
 * Builds a 3-D natural cubic spline through each gating branch, samples the
 * smooth SDF to obtain R(s), and distributes velocity as a Poiseuille profile
 * v(s,r) = v_axis(s) * (1 - (r/R(s))^2).  Two scalar fields are returned:
 *
 *   - bulk:   v_axis(s) for every gate voxel (used for surface colour / labels).
 *   - full:   the parabolic 3-D Poiseuille magnitude (kept for cross-sections).
 *
 * The function is fully isolated from the thermal/filling solvers; it is only
 * intended for the UI flow-velocity visualization.
 */
nb::tuple solve_spline_tube_field(
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> sdf,
    nb::ndarray<nb::numpy, int32_t, nb::shape<-1, -1, -1>> voxel_branch,
    double dx,
    std::array<double, 3> origin,
    nb::ndarray<nb::numpy, double, nb::shape<-1, 3>> node_centroids,
    nb::ndarray<nb::numpy, double, nb::shape<-1>> node_velocity,
    nb::ndarray<nb::numpy, double, nb::shape<-1>> node_area,
    nb::ndarray<nb::numpy, int32_t, nb::shape<-1>> branch_node_indices,
    nb::ndarray<nb::numpy, int32_t, nb::shape<-1>> branch_offsets);

} // namespace josecast
