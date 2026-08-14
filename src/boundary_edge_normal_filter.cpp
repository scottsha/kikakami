#include "boundary_edge_normal_filter.h"

#include "igl/boundary_loop.h"
#include "igl/per_face_normals.h"

#include "kikakami_utils.h"
#include "math_utilities.h"

namespace kikakami {

void BoundaryEdgeNormalFilter::update() {
    igl::boundary_loop(*mesh.triangles, *loops);
    igl::per_face_normals(*mesh.points, *mesh.triangles, *face_normals);
    edge_to_faces = compute_edge_vertex_pair_to_face_map(*mesh.triangles);
    auto num_boundary_edges = 0;
    for (const auto& loop : *loops) {
        num_boundary_edges += loop.size();
    }
    auto& b_normals = *boundary_normals;
    b_normals = Eigen::MatrixXd(num_boundary_edges, 3);
    auto loop_base = 0;
    for (const auto& loop : *loops) {
        const PointsR3 loop_points = (*mesh.points)(loop, Eigen::all);
        const PointsR3 boundary_edge_dirs = cyclic_row_differences(loop_points).rowwise().normalized();
        const PointsR3 surface_normals = find_surface_normal_at_loop(loop);
        const auto loop_size = loop.size();
        for (auto ii = 0; ii < loop_size; ++ii) {
            const auto& surf_norm = surface_normals.row(ii);
            const auto& edge_dir = boundary_edge_dirs.row(ii);
            b_normals.row(loop_base + ii) = edge_dir.cross(surf_norm);
        }
        loop_base += loop_size;
    }
}

auto BoundaryEdgeNormalFilter::get_boundary_adjacent_faces() const -> std::vector<Eigen::Index> {
    std::vector<Eigen::Index> bounding_faces;
    for (const auto& loop : *loops) {
        const auto loop_faces = find_boundary_adjacent_faces_at_loop(loop);
        bounding_faces.insert(bounding_faces.end(), loop_faces.begin(), loop_faces.end());
    }
    return bounding_faces;
}

auto BoundaryEdgeNormalFilter::find_boundary_adjacent_faces_at_loop(const Loop& loop
) const -> std::vector<Eigen::Index> {
    std::vector<Eigen::Index> boundary_faces;
    for (auto v_ii = 0; v_ii < loop.size(); ++v_ii) {
        const auto v0 = loop[v_ii];
        const auto v1 = loop[(v_ii + 1) % loop.size()];
        const auto triangles = edge_to_faces.at({ v0, v1 });
        assert(triangles.size() == 1);
        boundary_faces.push_back(triangles[0]);
    }
    return boundary_faces;
}


auto BoundaryEdgeNormalFilter::find_surface_normal_at_loop(const Loop& loop) const -> PointsR3 {
    const auto boundary_faces = find_boundary_adjacent_faces_at_loop(loop);
    PointsR3 loop_surface_normals = (*face_normals)(boundary_faces, Eigen::all);
    return loop_surface_normals;
}

auto loop_to_edges(const std::vector<Loop>& loops) -> GraphEdges {
    auto num_verts = 0;
    for (const auto& loop : loops) {
        num_verts += loop.size();
    }
    GraphEdges edges(num_verts, 2);
    auto count = 0;
    for (const auto& loop : loops) {
        for (auto v_ii = 0; v_ii < loop.size(); ++v_ii) {
            const auto v0 = loop[v_ii];
            const auto v1 = loop[(v_ii + 1) % loop.size()];
            edges(count, 0) = v0;
            edges(count, 1) = v1;
            ++count;
        }
    }
    return edges;
}

auto extract_boundary_edges(const EigenTriangleMesh& mesh) -> GraphEdges {
    std::vector<Loop> loops;
    igl::boundary_loop(*mesh.triangles, loops);
    return loop_to_edges(loops);
}

auto extract_boundary_edge_lengths(const EigenTriangleMesh& mesh) -> Eigen::VectorXd {
    const auto edges = extract_boundary_edges(mesh);
    const auto tips = (*mesh.points)(edges.col(0), Eigen::all);
    const auto tails = (*mesh.points)(edges.col(1), Eigen::all);
    const auto lengths = (tips - tails).rowwise().norm();
    return lengths;
}

auto extract_boundary_perimeter(const EigenTriangleMesh& mesh) -> double {
    const auto lengths = extract_boundary_edge_lengths(mesh);
    const auto total = lengths.sum();
    return total;
}


} // namespace kikakami
