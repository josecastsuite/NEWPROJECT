set(JOSECAST_SOURCES
    bindings.cpp
    voxelizer.cpp
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
