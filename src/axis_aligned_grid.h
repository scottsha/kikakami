#ifndef KIKAKAMI_AXIS_ALIGNED_GRID_H
#define KIKAKAMI_AXIS_ALIGNED_GRID_H

#include "eigenmesh.h"

namespace kikakami {

struct AxisAlignedGrid {
    std::array<int, 2> shape = { 0, 0 };
    PointR2 mins;
    PointR2 maxs;

    void update();
    void set_shape_by_resolution(double resolution);

    std::shared_ptr<PointsR2> points = std::make_shared<PointsR2>();
    std::shared_ptr<EigenTriangles> triangles = std::make_shared<EigenTriangles>();

    auto get_grid_point_index(int ii, int jj) const -> Eigen::Index;
    auto get_grid_point(int ii, int jj) const -> PointR2;

    AxisAlignedGrid() = default;
    AxisAlignedGrid(const PointR2& mins, const PointR2& maxs, const std::array<int, 2>& shape);

private:
    void make_points();
    void make_triangles();
};


} // namespace kikakami

#endif // KIKAKAMI_AXIS_ALIGNED_GRID_H
