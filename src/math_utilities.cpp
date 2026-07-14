#include "math_utilities.h"

namespace kikaku {

auto cyclic_row_differences(const Eigen::MatrixXd &points) -> Eigen::MatrixXd {
    const auto n_points = points.rows();
    Eigen::MatrixXd diff_mat(n_points, points.cols());
    diff_mat.row(0) = points.row(0) - points.row(n_points - 1);
    const auto n_midrows = n_points - 2;
    diff_mat.middleRows(1, n_midrows) = points.bottomRows(n_midrows) - points.topRows(n_midrows);
    diff_mat.row(n_points - 1) = points.row(0) - points.row(n_points - 1);
    return diff_mat;
}

Eigen::VectorXd linspace(double start, double stop, Eigen::Index size) {
    Eigen::VectorXd result(size);
    const auto denominator = static_cast<double>(size - 1);
    for (auto ii = 0; ii < size; ++ii) {
        const double tt = static_cast<double>(ii) / denominator;
        result(ii) = start * (1 - tt) + stop * tt;
    }
    return result;
}
}
