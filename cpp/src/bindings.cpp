#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/tuple.h>

#include "josecast/voxelizer.h"
#include "josecast/darcy_solver.h"

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

    m.def("solve_pressure", &josecast::solve_pressure,
          nb::arg("indptr"), nb::arg("indices"), nb::arg("data"), nb::arg("rhs"),
          nb::arg("max_iter") = 1000, nb::arg("rtol") = 1e-5, nb::arg("abstol") = 1e-12,
          "Solve a sparse symmetric-positive-definite pressure system with AMGCL (BiCGStab + AMG).");
}
