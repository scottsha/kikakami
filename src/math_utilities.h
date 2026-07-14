#ifndef KIKAKAMI_MATH_UTILITIES_H
#define KIKAKAMI_MATH_UTILITIES_H

#include "Eigen/Core"

namespace kikakami {

/**
 * @brief Compute cyclic (wrap-around) row-wise differences of a point matrix.
 * Commonly used for closed polylines / loops.
 * This gives the edge vectors for a loop represented by points.
 *
 * @param matrix N×D matrix of points
 * @return N×D matrix of forward differences with wrap-around.
 */
Eigen::MatrixXd cyclic_row_differences(const Eigen::MatrixXd& matrix);


Eigen::VectorXd linspace(double start, double stop, Eigen::Index size);

}

#endif //KIKAKAMI_MATH_UTILITIES_H
