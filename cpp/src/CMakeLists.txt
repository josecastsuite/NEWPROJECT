set(JOSECAST_SOURCES
    bindings.cpp
    cgal_skeleton.cpp
    darcy_solver.cpp
    gating_tree.cpp
    lbm_solver.cpp
    ns_solver.cpp
    sand_solver.cpp
    spline_tube_solver.cpp
    thermal_solver.cpp
)

if(JOSECAST_OPENVDB_FOUND)
    list(APPEND JOSECAST_SOURCES voxelizer.cpp)
endif()

nanobind_add_module(josecast_core ${JOSECAST_SOURCES})

target_compile_definitions(josecast_core PRIVATE AMGCL_NO_BOOST=1)

if(JOSECAST_OPENVDB_FOUND)
    target_compile_definitions(josecast_core PRIVATE JOSECAST_HAS_OPENVDB=1)
    if(JOSECAST_OPENVDB_TARGET)
        target_link_libraries(josecast_core PRIVATE ${JOSECAST_OPENVDB_TARGET})
    else()
        target_link_libraries(josecast_core PRIVATE ${JOSECAST_OPENVDB_LIBS})
    endif()
    # The Linux system openvdb needs the old TBB ABI define.
    if(NOT OpenVDB_FOUND)
        target_compile_definitions(josecast_core PRIVATE __TBB_NO_IMPLICIT_LINKAGE=1)
    endif()
else()
    message(WARNING "OpenVDB not found; C++ Voxelizer will be disabled")
endif()

if(Eigen3_FOUND)
    target_link_libraries(josecast_core PRIVATE Eigen3::Eigen)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(josecast_core PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(josecast_core PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_options(josecast_core PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# Copy the compiled extension module next to the Python package so it can be
# imported without setting PYTHONPATH (core/josecast_core.<soab i>.so/.pyd).
set(JOSECAST_CORE_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/../core")
add_custom_command(TARGET josecast_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${JOSECAST_CORE_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:josecast_core>
        ${JOSECAST_CORE_OUTPUT_DIR}/$<TARGET_FILE_NAME:josecast_core>
    COMMENT "Copying josecast_core extension to core/"
    VERBATIM
)
