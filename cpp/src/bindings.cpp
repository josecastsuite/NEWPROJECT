#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/tuple.h>

#include "josecast/voxelizer.h"
#include "josecast/darcy_solver.h"
#include "josecast/gating_tree.h"
#include "josecast/ns_solver.h"

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

    m.def("solve_gating_times", &josecast::solve_gating_times,
          nb::arg("source_bidx"), nb::arg("source_q"),
          nb::arg("body_volumes"), nb::arg("edges"), nb::arg("edge_q"), nb::arg("q_in"),
          "Compute t_enter/t_exit/Q for the gating tree using Kahn topological sort.");

    m.def("solve_ns_vof", &josecast::solve_ns_vof,
          nb::arg("grid"), nb::arg("phi_init"), nb::arg("source_mask"),
          nb::arg("dx"), nb::arg("g"), nb::arg("rho"), nb::arg("nu"),
          nb::arg("inflow_velocity"), nb::arg("t_max"), nb::arg("max_steps"),
          nb::arg("cfl"), nb::arg("max_pressure_iter"), nb::arg("pressure_tol"),
          "Run a 3-D fractional-step Navier-Stokes + level-set mold-filling solver.");
}
