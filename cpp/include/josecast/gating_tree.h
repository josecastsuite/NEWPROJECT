#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cstdint>

namespace nb = nanobind;

namespace josecast {

nb::tuple solve_gating_times(
    int32_t source_bidx,
    double source_q,
    nb::ndarray<nb::numpy, double, nb::shape<-1>> body_volumes,
    nb::ndarray<nb::numpy, int32_t, nb::shape<-1, 2>> edges,
    nb::ndarray<nb::numpy, double, nb::shape<-1>> edge_q,
    nb::ndarray<nb::numpy, double, nb::shape<-1>> q_in);

} // namespace josecast
