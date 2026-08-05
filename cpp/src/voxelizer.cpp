#include "josecast/voxelizer.h"

#include <openvdb/tools/ChangeBackground.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/SignedFloodFill.h>
#include <openvdb/math/Transform.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <vector>

namespace josecast {

namespace {
// Body-type priority for overlapping voxels.  Higher priority bodies win so
// that embedded inserts (chills, filters, cores) are not overwritten by the
// surrounding part or gating metal.
int body_priority(int32_t body_type) {
    switch (body_type) {
        case 1:  return 1; // PART
        case 3:  return 2; // RISER
        case 5:  return 2; // INGATE
        case 6:  return 2; // RUNNER
        case 7:  return 2; // SPRUE
        case 15: return 2; // POURING_BASIN
        case 17: return 2; // SPRUE_THROAT
        case 19: return 2; // DISTRIBUTOR
        case 21: return 2; // CURUFLUK
        case 9:  return 3; // CORE
        case 13: return 4; // FILTER
        case 11: return 5; // COOLING_SPRUE
        default: return 0;
    }
}
} // namespace

void Voxelizer::add_body(int32_t body_id, int32_t body_type,
                         const Vertices& vertices, const Faces& faces) {
    BodyInput body;
    body.id = body_id;
    body.type = body_type;

    const size_t n_vtx = vertices.shape(0);
    body.points.reserve(n_vtx);
    const float* vdata = vertices.data();
    for (size_t i = 0; i < n_vtx; ++i) {
        body.points.emplace_back(vdata[3 * i + 0],
                                 vdata[3 * i + 1],
                                 vdata[3 * i + 2]);
    }

    const size_t n_tri = faces.shape(0);
    body.triangles.reserve(n_tri);
    const int32_t* fdata = faces.data();
    for (size_t i = 0; i < n_tri; ++i) {
        body.triangles.emplace_back(fdata[3 * i + 0],
                                    fdata[3 * i + 1],
                                    fdata[3 * i + 2]);
    }

    bodies_.push_back(std::move(body));
}

nb::tuple Voxelizer::build(float voxel_size,
                           const std::array<float, 3>& origin,
                           const std::array<int32_t, 3>& dims) {
    // OpenVDB initialization is not thread-safe; guard it with std::call_once.
    static std::once_flag init_flag;
    std::call_once(init_flag, []() { openvdb::initialize(); });

    const int32_t nx = dims[0], ny = dims[1], nz = dims[2];
    const size_t total = static_cast<size_t>(nx) * ny * nz;

    // Output arrays owned by C++; nanobind ndarray will hold them alive.
    std::vector<int32_t>* grid_vec = new std::vector<int32_t>(total, 0);
    std::vector<int32_t>* bidx_vec = new std::vector<int32_t>(total, -1);
    std::vector<float>* sdf_vec = new std::vector<float>(total, 0.0f);

    // meshToVolume's adapter expects points in *index* space (i.e. voxel units)
    // and the transform maps index to world units.  Convert world-mm vertices to
    // index units by dividing by voxel_size and use an identity transform so
    // one index unit equals one voxel.
    const float inv_dx = 1.0f / voxel_size;
    openvdb::math::Transform::Ptr xform =
        openvdb::math::Transform::createLinearTransform(1.0f);

    for (const BodyInput& body : bodies_) {
        std::vector<openvdb::Vec3s> shifted;
        shifted.reserve(body.points.size());
        for (const auto& p : body.points) {
            shifted.emplace_back((p[0] - origin[0]) * inv_dx,
                                 (p[1] - origin[1]) * inv_dx,
                                 (p[2] - origin[2]) * inv_dx);
        }

        openvdb::tools::QuadAndTriangleDataAdapter<openvdb::Vec3s, openvdb::Vec3I>
            mesh(shifted, body.triangles);
        openvdb::FloatGrid::Ptr ls_grid = openvdb::tools::meshToVolume<openvdb::FloatGrid>(
            mesh, *xform, 3.0f, 3.0f);

        // meshToVolume produces a narrow band; inactive tiles have the grid's
        // background value.  Set a non-zero background so signedFloodFill can
        // propagate a signed distance into the deep interior and exterior.
        float far_field = std::fabs(voxel_size) * 100.0f;
        openvdb::tools::changeBackground(ls_grid->tree(), far_field);
        openvdb::tools::signedFloodFill(ls_grid->tree());
        ls_grid->setGridClass(openvdb::GRID_LEVEL_SET);

        auto accessor = ls_grid->getAccessor();

        for (int32_t i = 0; i < nx; ++i) {
            for (int32_t j = 0; j < ny; ++j) {
                for (int32_t k = 0; k < nz; ++k) {
                    openvdb::Coord coord(i, j, k);
                    float value = accessor.getValue(coord);

                    const size_t idx = static_cast<size_t>(i) * ny * nz +
                                       static_cast<size_t>(j) * nz +
                                       static_cast<size_t>(k);

                    // Level-set sign convention: negative inside, positive outside.
                    // sdf = max(0, -value) is positive inside, zero outside.
                    float d = std::max(0.0f, -value);
                    int cur_type = (*grid_vec)[idx];
                    int cur_prio = (cur_type == 0) ? -1 : body_priority(cur_type);
                    int new_prio = body_priority(body.type);
                    bool overwrite = false;
                    if (new_prio > cur_prio) {
                        overwrite = true;
                    } else if (new_prio == cur_prio) {
                        overwrite = d > (*sdf_vec)[idx] + 1e-6f;
                    }
                    if (overwrite) {
                        if (value < 0.0f) {
                            (*sdf_vec)[idx] = d;
                            (*grid_vec)[idx] = body.type;
                            (*bidx_vec)[idx] = body.id;
                        } else {
                            // Just outside the narrow band; leave as empty unless
                            // another body has already claimed this voxel.
                            (*sdf_vec)[idx] = d;
                            if ((*bidx_vec)[idx] < 0) {
                                (*grid_vec)[idx] = 0;
                                (*bidx_vec)[idx] = -1;
                            }
                        }
                    }
                }
            }
        }
    }

    auto grid_owner = nb::capsule(grid_vec, [](void* p) noexcept {
        delete static_cast<std::vector<int32_t>*>(p);
    });
    auto bidx_owner = nb::capsule(bidx_vec, [](void* p) noexcept {
        delete static_cast<std::vector<int32_t>*>(p);
    });
    auto sdf_owner = nb::capsule(sdf_vec, [](void* p) noexcept {
        delete static_cast<std::vector<float>*>(p);
    });

    nb::ndarray<nb::numpy, int32_t> grid_arr(
        grid_vec->data(), {static_cast<size_t>(nx),
                           static_cast<size_t>(ny),
                           static_cast<size_t>(nz)},
        grid_owner);

    nb::ndarray<nb::numpy, int32_t> bidx_arr(
        bidx_vec->data(), {static_cast<size_t>(nx),
                           static_cast<size_t>(ny),
                           static_cast<size_t>(nz)},
        bidx_owner);

    nb::ndarray<nb::numpy, float> sdf_arr(
        sdf_vec->data(), {static_cast<size_t>(nx),
                          static_cast<size_t>(ny),
                          static_cast<size_t>(nz)},
        sdf_owner);

    return nb::make_tuple(grid_arr, bidx_arr, sdf_arr);
}

} // namespace josecast
