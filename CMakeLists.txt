cmake_minimum_required(VERSION 4.2)
project(kikakami LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Clipper2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(libigl REQUIRED)
find_package(nlohmann_json REQUIRED)
#find_package(OpenCASCADE REQUIRED)
find_package(Spectra REQUIRED)
find_package(VTK REQUIRED)

add_library(
        kikakami
        src/eigenmesh.h
        src/axis_aligned_grid.cpp
        src/axis_aligned_grid.h
#        src/boundary_edge_normal_filter.cpp
#        src/boundary_edge_normal_filter.h
#        src/boundary_loop_arclength_locator.cpp
#        src/boundary_loop_arclength_locator.h
#        src/clipper2_translation.cpp
#        src/clipper2_translation.h
#        src/conformal_coordinate_charter.cpp
#        src/conformal_coordinate_charter.h
#        src/coordinate_chart_interpolater.cpp
#        src/coordinate_chart_interpolater.h
#        src/coordinate_chart_sampler.cpp
#        src/coordinate_chart_sampler.h
#        src/design_constraint_extraction.cpp
#        src/design_constraint_extraction.h
#        src/differentiable_implicits.cpp
#        src/differentiable_implicits.h
#        src/differentiable_occupancy_pyinterface.cpp
#        src/differentiable_occupancy_pyinterface.h
#        src/differentiable_topography_projector.cpp
#        src/differentiable_topography_projector.h
#        src/differentiable_voronoi_tessellator.cpp
#        src/differentiable_voronoi_tessellator.h
#        src/diffusion_flow_solver.cpp
#        src/diffusion_flow_solver.h
#        src/dijkstra_path_relaxation.cpp
#        src/dijkstra_path_relaxation.h
#        src/distance_utils.cpp
#        src/distance_utils.h
#        src/eigenmesh.h
#        src/eigenmesh_vtk_translation.cpp
#        src/eigenmesh_vtk_translation.h
#        src/eigenmode_solver.cpp
#        src/eigenmode_solver.h
#        src/embedded_graph_debug_utils.cpp
#        src/embedded_graph_debug_utils.h
#        src/geodesic_utils.cpp
#        src/geodesic_utils.h
#        src/kikakami_json_utils.h
#        src/kikakami_utils.cpp
#        src/kikakami_utils.h
#        src/kikakami_vtk_utils.cpp
#        src/kikakami_vtk_utils.h
#        src/harmonic_coordinate_charter.cpp
#        src/harmonic_coordinate_charter.h
#        src/inkblot_subsurface_splatterer.cpp
#        src/inkblot_subsurface_splatterer.h
#        src/intrinsic_triangulation.cpp
#        src/intrinsic_triangulation.h
#        src/io_utilities.cpp
#        src/io_utilities.h
#        src/planar_polygon_signed_distancer.cpp
#        src/planar_polygon_signed_distancer.h
#        src/quilter_diffgeo_data.cpp
#        src/quilter_diffgeo_data.h
#        src/quilter_domain.cpp
#        src/quilter_domain.h
#        src/quilter_domain_params.h
#        src/quilter_field_generator.cpp
#        src/quilter_field_generator.h
#        src/quilter_field_operator.cpp
#        src/quilter_field_operator.h
#        src/quilter_field_request_params.h
#        src/quilter_fused_offsetter.cpp
#        src/quilter_fused_offsetter.h
#        src/quilter_operation_params.h
#        src/quilter_params_json.cpp
#        src/quilter_params_json.h
#        src/quilter_run_params.h
#        src/quilter_sdf_envelope.cpp
#        src/quilter_sdf_envelope.h
#        src/quilter_tag_utils.cpp
#        src/quilter_tag_utils.h
#        src/quilter_tet_thicken.cpp
#        src/quilter_tet_thicken.h
#        src/shrink_wrapping_tools.cpp
#        src/shrink_wrapping_tools.h
#        src/soft_distance.cpp
#        src/soft_distance.h
#        src/spline_surface_utils.cpp
#        src/spline_surface_utils.h
#        src/sub_mesh_field_map.cpp
#        src/sub_mesh_field_map.h
#        src/svg_to_winding_field.cpp
#        src/svg_to_winding_field.h
#        src/symmetry_group_field_projection.cpp
#        src/symmetry_group_field_projection.h
#        src/symmetry_group_occupancy_projector.cpp
#        src/symmetry_group_occupancy_projector.h
#        src/transport_along_field_gradient_solver.cpp
#        src/transport_along_field_gradient_solver.h
#        src/triangle-mesh-distance.h
#        src/voronoi_quilter.cpp
#        src/voronoi_quilter.h
#        src/math_utilities.cpp
#        src/math_utilities.h
)

target_link_libraries(kikakami PUBLIC
        Eigen3::Eigen
        igl::igl
        igl::common
        ${VTK_LIBRARIES}
)


#############
### Tests ###
#############
SET(IS_BUILD_KIKAKAMI_TESTS OFF)

if (IS_BUILD_KIKAKAMI_TESTS)
    # CONFIGURE ASSET DIRECTORY #
    set(KIKAKAMI_ASSETS "${PROJECT_SOURCE_DIR}/assets")
    configure_file(${CMAKE_SOURCE_DIR}/assets/asset_dir_config.h.in ${CMAKE_BINARY_DIR}/assets/asset_dir_config.h @ONLY)
    include_directories(${CMAKE_BINARY_DIR}/apps)


    find_package(Catch2 CONFIG REQUIRED)
    include(Catch) # Include the Catch2 CMake module, to enable catch_discover_tests(), which does `add_test()` for ctest
    file(GLOB_RECURSE KIKAKAMI_TEST_SOURCES CONFIGURE_DEPENDS test/*.cpp) # Automaticall adds all .cpp files in the test directory using wild cards
    add_executable(kikakami_tests ${KIKAKAMI_TEST_SOURCES})
    target_include_directories(kikakami_tests PRIVATE "../apps")
    target_link_libraries(kikakami_tests PUBLIC
            autodiff
            kikakami
            Catch2::Catch2
            Catch2::Catch2WithMain
    )
    catch_discover_tests(kikakami_tests)
endif ()

#add_subdirectory(unit_tests)
#add_subdirectory(apps)
