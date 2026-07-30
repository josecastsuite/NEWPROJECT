#include "josecast/sand_solver.h"

#include <cmath>
#include <cstdint>
#include <vector>
#include <iostream>

namespace josecast {

nb::tuple compute_sand_permeability(
    nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, -1>> sand_mask,
    double afs_grain_size_mm,
    double moisture_percent,
    double binder_percent,
    double compactability_percent,
    double pressure_pa,
    double air_viscosity_pa_s,
    double dx_m)
{
    // sand_mask shape order from numpy is (nx, ny, nz).  nanobind's shape
    // annotation uses C-order dimensions, so the first index is the slowest.
    const size_t nx = sand_mask.shape(0);
    const size_t ny = sand_mask.shape(1);
    const size_t nz = sand_mask.shape(2);
    const size_t n = nx * ny * nz;

    // Convert AFS grain fineness number to an average grain diameter [mm].
    // AFS 50 -> ~0.28 mm, AFS 120 -> ~0.18 mm, which is in the foundry-sand
    // ballpark.  Clamp to avoid nonsensical values.
    double afs = std::max(1.0, afs_grain_size_mm);
    double grain_d_mm = 2.0 / std::sqrt(afs);
    if (grain_d_mm > 2.0) grain_d_mm = 2.0;
    if (grain_d_mm < 0.03) grain_d_mm = 0.03;
    double grain_d_m = grain_d_mm * 1e-3;

    // Base sand porosity.  Moisture and binder fill the voids, so they reduce
    // the connected pore space and therefore permeability.  Compactability is
    // a proxy for ramming density: higher compactability -> lower porosity.
    double phi0 = 0.48;
    double moisture_reduction = 0.0015 * std::max(0.0, moisture_percent);
    double binder_reduction   = 0.0020 * std::max(0.0, binder_percent);
    double compact_reduction  = 0.0008 * std::max(0.0, compactability_percent - 40.0);
    double porosity = phi0 - moisture_reduction - binder_reduction - compact_reduction;
    if (porosity < 0.15) porosity = 0.15;
    if (porosity > 0.60) porosity = 0.60;

    // Kozeny-Carman permeability for a packed bed of roughly spherical grains.
    // K = (1/180) * (phi^3 / (1-phi)^2) * d^2
    double kozeny = (porosity * porosity * porosity) /
                    ((1.0 - porosity) * (1.0 - porosity));
    double permeability_m2 = (1.0 / 180.0) * kozeny * grain_d_m * grain_d_m;

    // Apply a binder/clay coating correction factor (clay films reduce the
    // effective pore throats).  A simple exponential penalty based on binder.
    double binder_factor = std::exp(-0.02 * std::max(0.0, binder_percent));
    permeability_m2 *= binder_factor;

    // Very loose safety clamp: typical green-sand permeability is 1e-12..1e-9 m².
    if (permeability_m2 < 1e-14) permeability_m2 = 1e-14;
    if (permeability_m2 > 1e-7)  permeability_m2 = 1e-7;

    // Build the permeability field and estimate air leakage through the mould.
    // Leakage across a sand face adjacent to the cavity is approximated as
    // q = K * A * dp / (mu * dx), summed over all sand-cavity interfaces.
    const uint8_t* mask = sand_mask.data();
    std::vector<double> k_field(n, 0.0);
    double air_leak_rate_m3_s = 0.0;

    if (dx_m <= 0.0) dx_m = 1e-3;
    double mu_air = std::max(1e-12, air_viscosity_pa_s);
    double dp = std::max(0.0, pressure_pa);
    double face_area = dx_m * dx_m;

    auto idx = [&](size_t x, size_t y, size_t z) -> size_t {
        return (x * ny + y) * nz + z;
    };

    for (size_t x = 0; x < nx; ++x) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t z = 0; z < nz; ++z) {
                size_t i = idx(x, y, z);
                if (mask[i]) {
                    k_field[i] = permeability_m2;

                    // Neighbour check: if a neighbour is not sand, treat this
                    // face as a possible air-escape path.
                    auto add_leak = [&](size_t tx, size_t ty, size_t tz) {
                        if (tx >= nx || ty >= ny || tz >= nz) return;
                        size_t j = idx(tx, ty, tz);
                        if (mask[j] == 0) {
                            air_leak_rate_m3_s += permeability_m2 * face_area * dp /
                                                   (mu_air * dx_m);
                        }
                    };
                    if (x > 0)      add_leak(x - 1, y, z);
                    if (x + 1 < nx) add_leak(x + 1, y, z);
                    if (y > 0)      add_leak(x, y - 1, z);
                    if (y + 1 < ny) add_leak(x, y + 1, z);
                    if (z > 0)      add_leak(x, y, z - 1);
                    if (z + 1 < nz) add_leak(x, y, z + 1);
                }
            }
        }
    }

    auto* k_vec = new std::vector<double>(std::move(k_field));
    auto cap_k = nb::capsule(k_vec, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });

    nb::ndarray<nb::numpy, double, nb::shape<-1, -1, -1>> arr_k(
        k_vec->data(), {nx, ny, nz}, cap_k);

    return nb::make_tuple(
        arr_k,
        permeability_m2,
        air_leak_rate_m3_s,
        porosity,
        grain_d_mm
    );
}

} // namespace josecast
