#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/tuple.h>

#include "josecast/voxelizer.h"

namespace nb = nanobind;

NB_MODULE(josecast_core, m) {
    m.doc() = "JoseCast C++ core (OpenVDB + AMGCL + nanobind)";

    nb::class_<josecast::Voxelizer>(m, "Voxelizer")
        .def(nb::init<>())
        .def("add_body", &josecast::Voxelizer::add_body,
             nb::arg("body_id"), nb::arg("body_type"),
             nb::arg("vertices"), nb::arg("faces"))
        .def("build", &josecast::Voxelizer::build,
             nb::arg("voxel_size"), nb::arg("origin"), nb::arg("dims"),
             "Returns (grid, body_index, sdf) as 3-D numpy arrays.");
}
