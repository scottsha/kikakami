#ifndef KIKAKAMI_IO_UTILITIES_H
#define KIKAKAMI_IO_UTILITIES_H

#include <filesystem>

#include "vtkDataSet.h"
#include "vtkPolyData.h"
#include "vtkUnstructuredGrid.h"

#include "eigenmesh.h"

namespace kikakami {

void write_mesh(const vtkSmartPointer<vtkDataSet>& mesh, const std::filesystem::path& fout_path);

auto read_to_eigenmesh(const std::filesystem::path& fin_path) -> EigenTriangleMesh;

auto read_to_triangulated_polydata(const std::filesystem::path& fin_path) -> vtkSmartPointer<vtkPolyData>;

auto read_to_bool_tagged_mesh(const std::filesystem::path& mesh_path) -> vtkSmartPointer<vtkUnstructuredGrid>;

} // namespace kikakami

#endif // KIKAKAMI_IO_UTILITIES_H
