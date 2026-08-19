#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <array>
#include <cstdint>

namespace nb = nanobind;

namespace josecast {

// D3Q19 free-surface LBM filling solver.
//
// Solves the incompressible lattice Boltzmann equation on the mold cavity
// (grid == 0) with BGK collision + Smagorinsky sub-grid turbulence.  A
// volume-of-fluid-like marker phi tracks the metal/air interface; trapped
// air pockets are detected by connected-component analysis and returned as an
// entrapment risk map.
nb::tuple solve_lbm_filling(
    nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, -1>> grid,
    nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, -1>> inlet_mask,
    nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, -1>> outlet_mask,
    double dx,
    std::array<double, 3> g,
    double rho,
    double nu,
    double inflow_velocity,
    double t_max,
    int max_steps,
    double cfl_target,
    double smagorinsky,
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1, -1>> target_velocity = {},
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> inlet_distance = {});

// Transient one-way coupled LBM filling solver that invokes ``callback`` every
// ``callback_every_n`` steps with (step, dt, dx, cs2, vx, vy, vz, F, nu_t).
// This exposes the live velocity, free-surface marker and Smagorinsky eddy
// viscosity fields so that a Python-side D3Q7 gas-transport solver can run in
// lockstep with the C++ D3Q19 LBM.
nb::tuple solve_lbm_filling_callback(
    nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, -1>> grid,
    nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, -1>> inlet_mask,
    nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, -1>> outlet_mask,
    double dx,
    std::array<double, 3> g,
    double rho,
    double nu,
    double inflow_velocity,
    double t_max,
    int max_steps,
    double cfl_target,
    double smagorinsky,
    nb::callable callback,
    int callback_every_n = 10,
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1, -1>> target_velocity = {},
    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> inlet_distance = {});

} // namespace josecast
