#include "clipper2_translation.h"

#include "vtkCell.h"
#include "vtkLine.h"
#include "vtkPolygon.h"

namespace kikakami {

namespace {
    size_t count_points(const Clipper2Lib::PathsD& paths) {
        size_t n_points = 0;
        for (const auto& path : paths) {
            n_points += path.size();
        }
        return n_points;
    }
} // namespace


auto clipper_paths_to_vtk_lines(const Clipper2Lib::PathsD& paths) -> vtkSmartPointer<vtkPolyData> {
    auto vtk_points = vtkSmartPointer<vtkPoints>::New();
    auto vtk_lines = vtkSmartPointer<vtkCellArray>::New();
    for (const auto& path : paths) {
        int at_base_point = vtk_points->GetNumberOfPoints();
        for (const auto& point : path) {
            vtk_points->InsertNextPoint(static_cast<double>(point.x), static_cast<double>(point.y), 0.0);
        }
        const auto num_points = path.size();
        for (auto foo = 0; foo < num_points; ++foo) {
            auto line = vtkSmartPointer<vtkLine>::New();
            line->GetPointIds()->SetId(0, foo + at_base_point);
            line->GetPointIds()->SetId(1, (foo + 1) % num_points + at_base_point);
            vtk_lines->InsertNextCell(line);
        }
    }
    auto poly_data = vtkSmartPointer<vtkPolyData>::New();
    poly_data->SetPoints(vtk_points);
    poly_data->SetLines(vtk_lines);
    return poly_data;
}

auto clipper_paths_to_polygons(const Clipper2Lib::PathsD& paths) -> vtkSmartPointer<vtkPolyData> {
    auto poly_data = vtkSmartPointer<vtkPolyData>::New();
    auto points = vtkSmartPointer<vtkPoints>::New();
    auto polygons = vtkSmartPointer<vtkCellArray>::New();
    for (const auto& path : paths) {
        auto polygon = vtkSmartPointer<vtkPolygon>::New();
        polygon->GetPointIds()->SetNumberOfIds(path.size());
        for (size_t point_ii = 0; point_ii < path.size(); ++point_ii) {
            vtkIdType point_id = points->InsertNextPoint(path[point_ii].x, path[point_ii].y, 0.0);
            polygon->GetPointIds()->SetId(point_ii, point_id);
        }
        polygons->InsertNextCell(polygon);
    }
    poly_data->SetPoints(points);
    poly_data->SetPolys(polygons);
    return poly_data;
}

auto polydata_to_clipper_paths(const vtkSmartPointer<vtkPolyData>& poly_data) -> Clipper2Lib::PathsD {
    Clipper2Lib::PathsD paths;
    for (auto cell_foo = 0; cell_foo < poly_data->GetNumberOfCells(); cell_foo++) {
        std::vector<double> point_data;
        auto cell = poly_data->GetCell(cell_foo);
        for (auto point_ii = 0; point_ii < cell->GetNumberOfPoints(); point_ii++) {
            const auto pid = cell->GetPointId(point_ii);
            const auto point = poly_data->GetPoint(pid);
            point_data.emplace_back(point[0]);
            point_data.emplace_back(point[1]);
        }
        paths.emplace_back(Clipper2Lib::MakePathD(point_data));
    }
    return paths;
}

auto eigen_to_clipper_path(const PointsR2& loop) -> Clipper2Lib::PathD {
    // It should be possible to avoid a double copy with a Eigen::Map
    const Eigen::MatrixXd loop_t = loop.transpose();
    const std::vector<double> raw(loop_t.data(), loop_t.data() + loop_t.size());
    Clipper2Lib::PathD path = Clipper2Lib::MakePathD(raw);
    return path;
}

auto eigen_to_clipper_path(const PointsR2& points, const Eigen::VectorXi& loop_indices) -> Clipper2Lib::PathD {
    Eigen::MatrixXd loop_points = points(loop_indices, Eigen::all);
    return eigen_to_clipper_path(loop_points);
}

auto eigen_to_clipper_paths(const PointsR2& points, const std::vector<std::vector<Eigen::Index>>& loops)
    -> Clipper2Lib::PathsD {
    Clipper2Lib::PathsD paths;
    for (const auto& loop : loops) {
        Eigen::MatrixXd loop_points = points(loop, Eigen::all);
        paths.push_back(eigen_to_clipper_path(loop_points));
    }
    return paths;
}

auto clipper_paths_to_eigenpoints(const Clipper2Lib::PathsD& paths) -> PointsR2 {
    const auto n_points = count_points(paths);
    PointsR2 points(n_points, 2);
    auto at_point = 0;
    for (const auto& path : paths) {
        for (const auto& point : path) {
            points(at_point, 0) = static_cast<double>(point.x);
            points(at_point, 1) = static_cast<double>(point.y);
            ++at_point;
        }
    }
    return points;
}

namespace {

    GraphEdges make_cycle(int num_points, int id_start = 0) {
        GraphEdges edges(num_points, 2);
        edges(0, 0) = id_start;
        auto vid = id_start + 1;
        for (auto foo = 0; foo < num_points - 1; ++foo) {
            edges(foo, 1) = vid;
            edges(foo + 1, 0) = vid;
            vid++;
        }
        edges(num_points - 1, 0) = id_start + num_points - 1;
        edges(num_points - 1, 1) = id_start;
        return edges;
    }
} // namespace

auto clipper_paths_to_eigen_graph(const Clipper2Lib::PathsD& paths) -> EigenPlanarGraph {
    EigenPlanarGraph planar_graph;
    planar_graph.points = std::make_shared<PointsR2>(clipper_paths_to_eigenpoints(paths));
    planar_graph.edges = std::make_shared<GraphEdges>(planar_graph.points->rows(), 2);
    auto& edges = *planar_graph.edges;
    auto base_point = 0;
    for (const auto& path : paths) {
        const auto num_points = path.size();
        const auto cycle = make_cycle(num_points, base_point);
        edges.middleRows(base_point, num_points) = cycle;
        base_point += path.size();
    }
    return planar_graph;
}

Eigen::VectorXi winding_number(const Clipper2Lib::PathsD& paths, const PointsR2& points) {
    Eigen::VectorXi is_inside = Eigen::VectorXi::Zero(points.rows());
    for (auto ii = 0; ii < points.rows(); ++ii) {
        PointR2 point = points.row(ii);
        Clipper2Lib::PointD pp(point[0], point[1]);
        for (const auto& path : paths) {
            auto result = Clipper2Lib::PointInPolygon(pp, path);
            if (result != Clipper2Lib::PointInPolygonResult::IsOutside) {
                is_inside(ii) += 1;
            }
        }
    }
    return is_inside;
}

} // namespace kikakami