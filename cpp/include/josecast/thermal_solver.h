#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <array>
#include <cstdint>
#include <map>
#include <string>

namespace nb = nanobind;

namespace josecast {

// Enthalpy-based 3-D solidification solver.  Returns (T, fs, t_liq, t_s, G, R, niyama).
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
    std::array<double, 3> gravity_vector);

// Carlson-Beckermann dimensionless Niyama -> pore size / volume.
nb::tuple compute_porosity(
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> niyama,
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> M_mod,
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> feed_risk,
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> feed_eff,
    nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, -1>> part_mask,
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> velocity_magnitude,
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> darcy_factor,
    std::map<std::string, double> alloy,
    std::string carlson_curve_key);

} // namespace josecast
