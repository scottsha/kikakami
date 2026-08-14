#include "boundary_loop_arclength_locator.h"

#include <numeric>

#include "kikakami_utils.h"
#include "math_utilities.h"
#include "igl/boundary_loop.h"

namespace kikakami {

void PiecewiseLinearLoopArclengthLocator::from_graph(
    const PointsR3& points, const std::vector<Eigen::Index>& loop_ids
) {
    loop = loop_ids;
    const PointsR3 loop_points = points(loop, Eigen::all);
    const PointsR3 boundary_edge_dirs = cyclic_row_differences(loop_points);
    edge_lengths = boundary_edge_dirs.rowwise().norm();
    cumulative_lengths = Eigen::VectorXd(edge_lengths.size());
    std::partial_sum(edge_lengths.begin(), edge_lengths.end(), cumulative_lengths.data());
}

void PiecewiseLinearLoopArclengthLocator::from_mesh(const EigenTriangleMesh& mesh) {
    igl::boundary_loop(*mesh.triangles, loop);
    from_graph(*mesh.points, loop);
}

double PiecewiseLinearLoopArclengthLocator::get_loop_length() const { return *(cumulative_lengths.end() - 1); }

std::map<Eigen::Index, double> PiecewiseLinearLoopArclengthLocator::find_by_relative_arclength(double relative_length) {
    const double length = relative_length * *(cumulative_lengths.end() - 1);
    const auto below_iter = std::lower_bound(cumulative_lengths.begin(), cumulative_lengths.end(), length);
    const auto below = (below_iter - cumulative_lengths.begin()) % cumulative_lengths.size();
    const double tt = (cumulative_lengths[below] - length) / edge_lengths[below];
    const auto above = (below + 1) % cumulative_lengths.size();
    std::map<Eigen::Index, double> found;
    found[loop[below]] = tt;
    found[loop[above]] = 1 - tt;
    return found;
}


void ArclengthLoopsLocator::from_mesh(const EigenTriangleMesh& mesh) {
    loops = std::make_shared<std::vector<std::vector<Eigen::Index>>>();
    igl::boundary_loop(*mesh.triangles, *loops);
    points = mesh.points;
    update();
}

void ArclengthLoopsLocator::update() {
    edge_lengths = std::vector<Eigen::VectorXd>();
    cumulative_lengths = std::vector<Eigen::VectorXd>();
    cumulative_lengths.reserve(loops->size());
    for (const auto& loop : *loops) {
        const PointsR3 loop_points = (*points)(loop, Eigen::all);
        const PointsR3 boundary_edge_dirs = cyclic_row_differences(loop_points);
        edge_lengths.emplace_back(boundary_edge_dirs.rowwise().norm());
        auto& loop_edge_lengths = edge_lengths.back();
        cumulative_lengths.emplace_back(Eigen::VectorXd(loop_edge_lengths.size()));
        auto& cumulative_lengths_ref = cumulative_lengths.back();
        std::partial_sum(loop_edge_lengths.begin(), loop_edge_lengths.end(), cumulative_lengths_ref.data());
    }
}

std::map<Eigen::Index, double>
    ArclengthLoopsLocator::find_by_relative_arclength(size_t loop_index, double relative_length) const {
    const auto& lengths = cumulative_lengths[loop_index];
    const double length = relative_length * *(lengths.end() - 1);
    const auto below_iter = std::lower_bound(lengths.begin(), lengths.end(), length);
    const auto below = below_iter - lengths.begin();
    const auto& loop_edge_lengths = edge_lengths[loop_index];
    const double tt = (lengths[below] - length) / loop_edge_lengths[below];
    const auto above = (below + 1) % lengths.size();
    std::map<Eigen::Index, double> found;
    const auto& loop = (*loops)[loop_index];
    found[loop[below]] = tt;
    found[loop[above]] = 1 - tt;
    return found;
}

double ArclengthLoopsLocator::get_loop_length(size_t loop_index) const {
    return *(cumulative_lengths[loop_index].end() - 1);
}

PointR3 weights_to_coordinate(const PointsR3& points, const std::map<Eigen::Index, double>& weights) {
    PointR3 point = PointR3::Zero();
    for (const auto& weight : weights) {
        point += weight.second * points.row(weight.first);
    }
    return point;
}


} // namespace kikakami
