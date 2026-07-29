#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <openvdb/openvdb.h>
#include <vector>
#include <array>
#include <cstdint>

namespace nb = nanobind;

namespace josecast {

struct BodyInput {
    int32_t id = -1;          // body index (0, 1, 2 ...)
    int32_t type = 0;         // BodyType enum value (PART, INGATE, ...)
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> triangles;
};

class Voxelizer {
public:
    using Vertices = nb::ndarray<nb::numpy, float, nb::shape<-1, 3>>;
    using Faces = nb::ndarray<nb::numpy, int32_t, nb::shape<-1, 3>>;

    /// Add one closed body mesh.  
    /// @param body_id    0-based body index used for `body_index` output.
    /// @param body_type  BodyType enum value used for `grid` output.
    void add_body(int32_t body_id, int32_t body_type,
                  const Vertices& vertices, const Faces& faces);

    /// Voxelize all added bodies and return the dense fields.
    /// @param voxel_size  voxel pitch in mm (same unit as vertices).
    /// @param origin      world coordinate of grid corner (grid[0,0,0]).
    /// @param dims        (nx, ny, nz) number of voxels.
    /// @returns (grid, body_index, sdf) as 3-D numpy arrays.
    nb::tuple build(float voxel_size,
                    const std::array<float, 3>& origin,
                    const std::array<int32_t, 3>& dims);

private:
    std::vector<BodyInput> bodies_;
    static bool openvdb_initialized_;
};

} // namespace josecast
