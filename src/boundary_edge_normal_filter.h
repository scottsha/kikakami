#ifndef KIKAKAMI_BOUNDARY_EDGE_NORMAL_FILTER_H
#define KIKAKAMI_BOUNDARY_EDGE_NORMAL_FILTER_H

#include "eigenmesh.h"

namespace kikakami {

using Loop = std::vector<Eigen::Index>;

struct BoundaryEdgeNormalFilter {
    /*
     * For a surface compute the boundary edge normals, tangent to the surface
     */

    // Inputs
    EigenTriangleMesh mesh;

    void update();

    // Outputs
    EdgeToFacesMap edge_to_faces;
    std::shared_ptr<std::vector<Loop>> loops = std::make_shared<std::vector<Loop>>();
    std::shared_ptr<PointsR3> face_normals = std::make_shared<PointsR3>();
    std::shared_ptr<PointsR3> boundary_normals = std::make_shared<PointsR3>();
    std::shared_ptr<std::vector<Eigen::Index>> boundary_adjacent_faces = std::make_shared<std::vector<Eigen::Index>>();

    auto get_boundary_adjacent_faces() const -> std::vector<Eigen::Index>;

private:
    auto find_boundary_adjacent_faces_at_loop(const Loop& loop) const -> std::vector<Eigen::Index>;
    auto find_surface_normal_at_loop(const Loop& loop) const -> PointsR3;
};

auto loop_to_edges(const std::vector<Loop>& loops) -> GraphEdges;

auto extract_boundary_edges(const EigenTriangleMesh& mesh) -> GraphEdges;

auto extract_boundary_edge_lengths(const EigenTriangleMesh& mesh) -> Eigen::VectorXd;

auto extract_boundary_perimeter(const EigenTriangleMesh& mesh) -> double;

} // namespace kikakami

#endif // KIKAKAMI_BOUNDARY_EDGE_NORMAL_FILTER_H
