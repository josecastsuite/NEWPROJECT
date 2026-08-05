#include "josecast/spline_tube_solver.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/extract_mean_curvature_flow_skeleton.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/Polygon_mesh_processing/manifoldness.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/boost/graph/helpers.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace josecast {

namespace {

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = Kernel::Point_3;
using Mesh = CGAL::Surface_mesh<Point>;
using Skeletonization = CGAL::Mean_curvature_flow_skeletonization<Mesh>;
using Skeleton = Skeletonization::Skeleton;

Mesh build_mesh_from_arrays(nb::ndarray<nb::numpy, double, nb::shape<-1, 3>> vertices,
                            nb::ndarray<nb::numpy, int32_t, nb::shape<-1, 3>> faces,
                            bool repair) {
    const size_t n_verts = vertices.shape(0);
    const size_t n_faces = faces.shape(0);

    std::vector<Point> points;
    points.reserve(n_verts);
    for (size_t i = 0; i < n_verts; ++i) {
        points.emplace_back(vertices(i, 0), vertices(i, 1), vertices(i, 2));
    }

    std::vector<std::vector<std::size_t>> polygons;
    polygons.reserve(n_faces);
    for (size_t i = 0; i < n_faces; ++i) {
        polygons.push_back({static_cast<std::size_t>(faces(i, 0)),
                            static_cast<std::size_t>(faces(i, 1)),
                            static_cast<std::size_t>(faces(i, 2))});
    }

    if (repair) {
        CGAL::Polygon_mesh_processing::repair_polygon_soup(points, polygons);
        CGAL::Polygon_mesh_processing::orient_polygon_soup(points, polygons);
    }

    Mesh mesh;
    CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(points, polygons, mesh);

    if (repair) {
        CGAL::Polygon_mesh_processing::stitch_borders(mesh);
        CGAL::Polygon_mesh_processing::remove_isolated_vertices(mesh);
        CGAL::Polygon_mesh_processing::duplicate_non_manifold_vertices(mesh);
    }

    // Try to close any remaining holes using CGAL hole filling.
    for (int iter = 0; iter < 10 && !CGAL::is_closed(mesh); ++iter) {
        typename Mesh::Halfedge_index border_h = Mesh::null_halfedge();
        for (auto h : mesh.halfedges()) {
            if (CGAL::is_border(h, mesh)) {
                border_h = h;
                break;
            }
        }
        if (border_h == Mesh::null_halfedge()) break;
        std::vector<typename Mesh::Face_index> patch;
        CGAL::Polygon_mesh_processing::triangulate_hole(mesh, border_h, std::back_inserter(patch));
    }

    if (!CGAL::is_closed(mesh)) {
        throw std::runtime_error("cgal skeleton: mesh is not closed after repair; cannot skeletonize");
    }
    return mesh;
}

} // namespace

nb::tuple extract_skeleton(nb::ndarray<nb::numpy, double, nb::shape<-1, 3>> vertices,
                           nb::ndarray<nb::numpy, int32_t, nb::shape<-1, 3>> faces,
                           bool repair) {
    Mesh mesh = build_mesh_from_arrays(vertices, faces, repair);

    Skeleton skeleton;
    CGAL::extract_mean_curvature_flow_skeleton(mesh, skeleton);

    const size_t n_skel_verts = boost::num_vertices(skeleton);
    const size_t n_skel_edges = boost::num_edges(skeleton);

    std::vector<double> pts(n_skel_verts * 3);
    for (size_t i = 0; i < n_skel_verts; ++i) {
        pts[3 * i + 0] = skeleton[i].point.x();
        pts[3 * i + 1] = skeleton[i].point.y();
        pts[3 * i + 2] = skeleton[i].point.z();
    }

    std::vector<int32_t> eds(n_skel_edges * 2);
    size_t k = 0;
    auto es = boost::edges(skeleton);
    for (auto eit = es.first; eit != es.second; ++eit, ++k) {
        eds[2 * k + 0] = static_cast<int32_t>(source(*eit, skeleton));
        eds[2 * k + 1] = static_cast<int32_t>(target(*eit, skeleton));
    }

    auto pts_vec = new std::vector<double>(std::move(pts));
    auto eds_vec = new std::vector<int32_t>(std::move(eds));

    auto cap_pts = nb::capsule(pts_vec, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    auto cap_eds = nb::capsule(eds_vec, [](void* p) noexcept { delete static_cast<std::vector<int32_t>*>(p); });

    nb::ndarray<nb::numpy, double, nb::shape<-1, 3>> arr_pts(
        pts_vec->data(), {n_skel_verts, 3}, cap_pts);
    nb::ndarray<nb::numpy, int32_t, nb::shape<-1, 2>> arr_eds(
        eds_vec->data(), {n_skel_edges, 2}, cap_eds);

    return nb::make_tuple(arr_pts, arr_eds);
}

} // namespace josecast
