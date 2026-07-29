#include "josecast/darcy_solver.h"

#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

#include <cstdint>
#include <vector>
#include <iostream>
#include <tuple>

namespace josecast {

nb::ndarray<nb::numpy, double, nb::shape<-1>> solve_pressure(
    nb::ndarray<nb::numpy, int64_t, nb::shape<-1>> indptr,
    nb::ndarray<nb::numpy, int64_t, nb::shape<-1>> indices,
    nb::ndarray<nb::numpy, double, nb::shape<-1>> data,
    nb::ndarray<nb::numpy, double, nb::shape<-1>> rhs,
    int max_iter,
    double rtol,
    double abstol)
{
    const size_t n = rhs.shape(0);
    const size_t nnz = data.shape(0);

    // Copy into STL containers with the signed-index types AMGCL expects.
    const int64_t* ptr_raw = indptr.data();
    const int64_t* col_raw = indices.data();
    const double*   val_raw = data.data();
    const double*   rhs_raw = rhs.data();

    std::vector<ptrdiff_t> ptr(ptr_raw, ptr_raw + n + 1);
    std::vector<ptrdiff_t> col(col_raw, col_raw + nnz);
    std::vector<double>    val(val_raw, val_raw + nnz);
    std::vector<double>    b(rhs_raw, rhs_raw + n);

    typedef amgcl::backend::builtin<double, ptrdiff_t, ptrdiff_t> Backend;
    typedef amgcl::make_solver<
        amgcl::amg<
            Backend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0
        >,
        amgcl::solver::bicgstab<Backend>
    > Solver;

    typename Solver::params prm;
    prm.solver.maxiter = static_cast<size_t>(max_iter);
    prm.solver.tol = rtol;
    prm.solver.abstol = abstol;

    Solver solver(std::tie(n, ptr, col, val), prm);

    std::vector<double>* x = new std::vector<double>(n, 0.0);
    auto [iters, err] = solver(b, *x);

    if (iters >= prm.solver.maxiter) {
        std::cerr << "[josecast_core] AMGCL pressure solver did not converge: "
                  << "iters=" << iters << " err=" << err << "\n";
    }

    auto owner = nb::capsule(x, [](void* p) noexcept {
        delete static_cast<std::vector<double>*>(p);
    });

    return nb::ndarray<nb::numpy, double, nb::shape<-1>>(
        x->data(), {static_cast<size_t>(n)}, owner);
}

} // namespace josecast
