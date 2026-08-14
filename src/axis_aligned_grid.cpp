#include "axis_aligned_grid.h"

#include "math_utilities.h"

namespace kikakami {

void AxisAlignedGrid::update() {
    make_points();
    make_triangles();
}

auto AxisAlignedGrid::get_grid_point_index(int ii, int jj) const -> Eigen::Index {
    const Eigen::Index row = (shape[0] * jj) + ii;
    return row;
}

auto AxisAlignedGrid::get_grid_point(int ii, int jj) const -> PointR2 {
    return points->row(get_grid_point_index(ii, jj));
}

AxisAlignedGrid::AxisAlignedGrid(const PointR2& mins, const PointR2& maxs, const std::array<int, 2>& shape)
    : mins(mins), maxs(maxs), shape(shape) {
    update();
}

void AxisAlignedGrid::make_points() {
    const auto xxs = linspace(mins[0], maxs[0], shape[0]);
    const auto yys = linspace(mins[1], maxs[1], shape[1]);
    const Eigen::Index num_points = shape[0] * shape[1];
    auto& grid_points = *points;
    grid_points = PointsR2(num_points, 2);
    for (auto ii = 0; ii < shape[0]; ++ii) {
        for (auto jj = 0; jj < shape[1]; ++jj) {
            const auto row = get_grid_point_index(ii, jj);
            grid_points(row, 0) = xxs(ii);
            grid_points(row, 1) = yys(jj);
        }
    }
}


void AxisAlignedGrid::make_triangles() {
    const Eigen::Index total = 2 * (shape[0] - 1) * (shape[1] - 1);
    auto& tris = *triangles;
    tris = EigenTriangles(total, 3);
    auto triangle_count = 0;
    for (auto ii = 0; ii < shape[0] - 1; ++ii) {
        for (auto jj = 0; jj < shape[1] - 1; ++jj) {
            const auto v0 = get_grid_point_index(ii, jj);
            const auto v1 = get_grid_point_index(ii + 1, jj);
            const auto v2 = get_grid_point_index(ii + 1, jj + 1);
            const auto v3 = get_grid_point_index(ii, jj + 1);
            tris.row(triangle_count) =
                Eigen::RowVector3i{ static_cast<int>(v0), static_cast<int>(v1), static_cast<int>(v2) };
            triangle_count++;
            tris.row(triangle_count) =
                Eigen::RowVector3i{ static_cast<int>(v0), static_cast<int>(v2), static_cast<int>(v3) };
            triangle_count++;
        }
    }
}

void AxisAlignedGrid::set_shape_by_resolution(double resolution) {
    const PointR2 deltas = maxs - mins;
    shape[0] = static_cast<int>(std::round(deltas[0] / resolution));
    shape[1] = static_cast<int>(std::round(deltas[1] / resolution));
}

} // namespace kikakami
