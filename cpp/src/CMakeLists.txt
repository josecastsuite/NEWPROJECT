set(JOSECAST_SOURCES
    bindings.cpp
    voxelizer.cpp
    darcy_solver.cpp
    gating_tree.cpp
    ns_solver.cpp
)

nanobind_add_module(josecast_core ${JOSECAST_SOURCES})

target_link_libraries(josecast_core PRIVATE
    ${OPENVDB_LIBRARY}
    ${TBB_LIBRARY}
    ${HALF_LIBRARY}
    ${LOG4CPLUS_LIBRARY}
    ${BOOST_IOSTREAMS_LIBRARY}
    ${BLOSC_LIBRARY}
)

# OpenVDB needs C++14/17 and TBB atomic definitions.
target_compile_definitions(josecast_core PRIVATE
    __TBB_NO_IMPLICIT_LINKAGE=1
)

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
