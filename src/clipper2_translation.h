#ifndef CLIPPER2_TRANSLATION_H
#define CLIPPER2_TRANSLATION_H

#include "clipper2/clipper.h"

#include "vtkPolyData.h"

#include "eigenmesh.h"

namespace kikakami {

auto clipper_paths_to_vtk_lines(const Clipper2Lib::PathsD& paths) -> vtkSmartPointer<vtkPolyData>;

auto clipper_paths_to_polygons(const Clipper2Lib::PathsD& paths) -> vtkSmartPointer<vtkPolyData>;

auto polydata_to_clipper_paths(const vtkSmartPointer<vtkPolyData>& poly_data) -> Clipper2Lib::PathsD;

auto eigen_to_clipper_path(const PointsR2& loop) -> Clipper2Lib::PathD;

auto eigen_to_clipper_path(const PointsR2& points, const Eigen::VectorXi& loop_indices) -> Clipper2Lib::PathD;

auto eigen_to_clipper_paths(const PointsR2& points, const std::vector<std::vector<Eigen::Index>>& loops)
    -> Clipper2Lib::PathsD;

auto clipper_paths_to_eigenpoints(const Clipper2Lib::PathsD& paths) -> PointsR2;

auto clipper_paths_to_eigen_graph(const Clipper2Lib::PathsD& paths) -> EigenPlanarGraph;

auto winding_number(const Clipper2Lib::PathsD& paths, const PointsR2& points) -> Eigen::VectorXi;

} // namespace kikakami

#endif // CLIPPER2_TRANSLATION_H
