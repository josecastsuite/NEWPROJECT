#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <array>
#include <cstdint>

namespace nb = nanobind;

namespace josecast {

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
    double pressure_tol);

} // namespace josecast
