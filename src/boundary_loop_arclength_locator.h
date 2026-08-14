#ifndef kikakami_BOUNDARY_LOOP_ARCLENGTH_LOCATOR_H
#define kikakami_BOUNDARY_LOOP_ARCLENGTH_LOCATOR_H

#include <map>

#include "eigenmesh.h"

namespace kikakami {

struct PiecewiseLinearLoopArclengthLocator {
    // Find points along the boundary loop by relative arclength from loop base
    std::vector<Eigen::Index> loop;
    Eigen::VectorXd edge_lengths;
    Eigen::VectorXd cumulative_lengths;

    void from_graph(const PointsR3& points, const std::vector<Eigen::Index>& loop_ids);

    void from_mesh(const EigenTriangleMesh& mesh);

    double get_loop_length() const;

    std::map<Eigen::Index, double> find_by_relative_arclength(double relative_length);
};

struct ArclengthLoopsLocator {
    std::shared_ptr<std::vector<std::vector<Eigen::Index>>> loops;
    std::shared_ptr<PointsR3> points;

    void from_mesh(const EigenTriangleMesh& mesh);

    void update();

    // Find points along the boundary loop by relative arclength from loop base
    std::vector<Eigen::VectorXd> edge_lengths;
    std::vector<Eigen::VectorXd> cumulative_lengths;

    std::map<Eigen::Index, double> find_by_relative_arclength(size_t loop_index, double relative_length) const;

    double get_loop_length(size_t loop_index) const;
};

PointR3 weights_to_coordinate(const PointsR3& points, const std::map<Eigen::Index, double>& weights);

} // namespace kikakami


#endif // kikakami_BOUNDARY_LOOP_ARCLENGTH_LOCATOR_H
