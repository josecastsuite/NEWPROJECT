#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cstdint>

namespace nb = nanobind;

namespace josecast {

nb::ndarray<nb::numpy, double, nb::shape<-1>> solve_pressure(
    nb::ndarray<nb::numpy, int64_t, nb::shape<-1>> indptr,
    nb::ndarray<nb::numpy, int64_t, nb::shape<-1>> indices,
    nb::ndarray<nb::numpy, double, nb::shape<-1>> data,
    nb::ndarray<nb::numpy, double, nb::shape<-1>> rhs,
    int max_iter = 1000,
    double rtol = 1e-5,
    double abstol = 1e-12);

} // namespace josecast
