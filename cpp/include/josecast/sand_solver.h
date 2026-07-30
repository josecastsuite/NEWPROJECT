#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <array>
#include <cstdint>

namespace nb = nanobind;

namespace josecast {

nb::tuple compute_sand_permeability(
    nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, -1>> sand_mask,
    double afs_grain_size_mm,
    double moisture_percent,
    double binder_percent,
    double compactability_percent,
    double pressure_pa,
    double air_viscosity_pa_s,
    double dx_m);

} // namespace josecast
