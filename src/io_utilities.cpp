#include "io_utilities.h"

#include "eigenmesh_vtk_translation.h"

#include <filesystem>
#include <iostream>

#include <vtkAppendFilter.h>
#include <vtkCellData.h>
#include <vtkGeometryFilter.h>
#include <vtkOBJWriter.h>
#include <vtkPolyData.h>
#include <vtkSTLReader.h>
#include <vtkSTLWriter.h>
#include <vtkTriangleFilter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLPolyDataReader.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridWriter.h>

#include <igl/read_triangle_mesh.h>

namespace {
vtkSmartPointer<vtkUnstructuredGrid> to_unstructured_grid(const vtkSmartPointer<vtkPolyData>& polydata) {
    // Combine the two data sets.
    vtkNew<vtkAppendFilter> appender;
    appender->SetInputData(polydata);
    appender->Update();
    vtkNew<vtkUnstructuredGrid> unstructured_grid;
    unstructured_grid->ShallowCopy(appender->GetOutput());
    return unstructured_grid;
}
} // namespace


namespace kikakami {
void write_mesh(const vtkSmartPointer<vtkDataSet>& mesh, const std::filesystem::path& fout_path) {
    auto extension = fout_path.extension().string();
    if (extension == ".vtu") {
        // Write Unstructured Grid
        auto writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
        writer->SetFileName(fout_path.c_str());
        writer->SetDataModeToBinary();
        vtkSmartPointer<vtkUnstructuredGrid> unstructured_grid;
        if (mesh->IsA("vtkUnstructuredGrid")) {
            unstructured_grid = vtkUnstructuredGrid::SafeDownCast(mesh);
        }
        if (mesh->IsA("vtkPolyData")) {
            auto polydata = vtkPolyData::SafeDownCast(mesh);
            unstructured_grid = to_unstructured_grid(polydata);
        }
        writer->SetInputData(unstructured_grid);
        writer->Write();
    } else if (extension == ".vtp") {
        // Write PolyData
        auto writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
        writer->SetFileName(fout_path.c_str());
        writer->SetInputData(vtkPolyData::SafeDownCast(mesh));
        writer->SetDataModeToBinary();
        writer->Write();
    } else if (extension == ".stl") {
        // Write STL
        auto writer = vtkSmartPointer<vtkSTLWriter>::New();
        writer->SetFileName(fout_path.c_str());
        writer->SetInputData(vtkPolyData::SafeDownCast(mesh));
        writer->Write();
    } else if (extension == ".obj") {
        // Write OBJ
        auto writer = vtkSmartPointer<vtkOBJWriter>::New();
        writer->SetFileName(fout_path.c_str());
        writer->SetInputData(vtkPolyData::SafeDownCast(mesh));
        writer->Write();
    } else {
        std::cerr << "Unsupported file extension: " << extension << '\n';
    }
}

auto read_to_eigenmesh(const std::filesystem::path& fin_path) -> EigenTriangleMesh {
    EigenTriangleMesh mesh;
    igl::read_triangle_mesh(fin_path, *mesh.points, *mesh.triangles);
    return mesh;
}

auto read_to_triangulated_polydata(const std::filesystem::path& fin_path) -> vtkSmartPointer<vtkPolyData> {
    auto extension = fin_path.extension().string();
    vtkSmartPointer<vtkPolyData> poly_data;
    if (extension == ".stl") {
        auto reader = vtkSmartPointer<vtkSTLReader>::New();
        reader->SetFileName(fin_path.c_str());
        reader->Update();
        poly_data = reader->GetOutput();
    } else if (extension == ".vtp") {
        auto reader = vtkSmartPointer<vtkXMLPolyDataReader>::New();
        reader->SetFileName(fin_path.c_str());
        reader->Update();
        const auto mesh = reader->GetOutput();
        auto triangler = vtkSmartPointer<vtkTriangleFilter>::New();
        triangler->SetInputData(mesh);
        triangler->PassVertsOff();
        triangler->PassLinesOff();
        triangler->Update();
        poly_data = triangler->GetOutput();
    } else if (extension == ".vtu") {
        auto reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
        reader->SetFileName(fin_path.c_str());
        reader->Update();
        const auto mesh = reader->GetOutput();
        auto geomer = vtkSmartPointer<vtkGeometryFilter>::New();
        geomer->SetInputData(mesh);
        geomer->PassThroughCellIdsOff();
        geomer->PassThroughPointIdsOff();
        geomer->Update();
        auto triangler = vtkSmartPointer<vtkTriangleFilter>::New();
        triangler->SetInputData(geomer->GetOutput());
        triangler->PassVertsOff();
        triangler->PassLinesOff();
        triangler->Update();
        poly_data = triangler->GetOutput();
    } else {
        auto eigen_style_mesh = read_to_eigenmesh(fin_path);
        poly_data = to_polydata(eigen_style_mesh);
    }
    return poly_data;
}

auto read_to_bool_tagged_mesh(const std::filesystem::path& mesh_path) -> vtkSmartPointer<vtkUnstructuredGrid> {
    auto reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
    reader->SetFileName(mesh_path.c_str());
    reader->Update();
    vtkSmartPointer<vtkUnstructuredGrid> ugrid = reader->GetOutput();
    return ugrid;
}

} // namespace kikakami
