cmake_minimum_required(VERSION 3.15...3.22)
project(josecast_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Python + nanobind -------------------------------------------------
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c
        "import nanobind, os; print(os.path.join(os.path.dirname(nanobind.__file__), 'cmake'))"
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(nanobind CONFIG REQUIRED)

# AMGCL header-only Poisson/linear solver ---------------------------
set(AMGCL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/amgcl")
if(NOT EXISTS "${AMGCL_ROOT}/amgcl/make_solver.hpp")
    message(FATAL_ERROR "AMGCL headers not found at ${AMGCL_ROOT}")
endif()

# Eigen3: prefer config (vcpkg), fallback to system include path ----
find_package(Eigen3 CONFIG QUIET)
if(NOT Eigen3_FOUND)
    find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Core PATHS /usr/include/eigen3)
    if(NOT EIGEN3_INCLUDE_DIR)
        message(FATAL_ERROR "Eigen3 headers not found")
    endif()
    set(EIGEN3_FOUND TRUE)
endif()

# OpenVDB: prefer CMake config (vcpkg/conda), fallback to Linux -----
set(JOSECAST_OPENVDB_FOUND FALSE)
find_package(OpenVDB CONFIG QUIET)
if(OpenVDB_FOUND)
    set(JOSECAST_OPENVDB_FOUND TRUE)
    set(JOSECAST_OPENVDB_TARGET OpenVDB::openvdb)
    message(STATUS "OpenVDB found via CMake config")
else()
    find_library(OPENVDB_LIBRARY openvdb)
    find_library(BLOSC_LIBRARY blosc)
    find_library(BOOST_IOSTREAMS_LIBRARY boost_iostreams)
    find_library(HALF_LIBRARY NAMES Half Half-2_5)
    find_library(TBB_LIBRARY
        NAMES libtbb.so.2 tbb
        PATHS /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu
        NO_DEFAULT_PATH
    )
    find_library(LOG4CPLUS_LIBRARY
        NAMES log4cplus-2.0 log4cplus
        PATHS /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu
    )
    if(NOT TBB_LIBRARY)
        set(TBB_LIBRARY "/usr/lib/x86_64-linux-gnu/libtbb.so.2")
    endif()
    if(NOT LOG4CPLUS_LIBRARY)
        set(LOG4CPLUS_LIBRARY "/lib/x86_64-linux-gnu/liblog4cplus-2.0.so.3")
    endif()
    if(OPENVDB_LIBRARY)
        set(JOSECAST_OPENVDB_FOUND TRUE)
        set(JOSECAST_OPENVDB_LIBS
            ${OPENVDB_LIBRARY}
            ${TBB_LIBRARY}
            ${HALF_LIBRARY}
            ${LOG4CPLUS_LIBRARY}
            ${BOOST_IOSTREAMS_LIBRARY}
            ${BLOSC_LIBRARY}
        )
        message(STATUS "OpenVDB found via find_library: ${OPENVDB_LIBRARY}")
    endif()
endif()

# nanoflann (header-only KD-tree) -------------------------------------
set(NANOFLANN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nanoflann")
if(NOT EXISTS "${NANOFLANN_DIR}/nanoflann.hpp")
    message(FATAL_ERROR "nanoflann.hpp not found at ${NANOFLANN_DIR}")
endif()

# OpenMP -------------------------------------------------------------
find_package(OpenMP COMPONENTS CXX QUIET)

# Include dirs common to all sources
include_directories(
    BEFORE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${AMGCL_ROOT}"
    "${NANOFLANN_DIR}"
)

if(NOT OpenVDB_FOUND AND OPENVDB_LIBRARY)
    # The system libopenvdb on Ubuntu 22.04 was built against the old TBB ABI;
    # use the vendored TBB 2020 headers to match that ABI.
    set(JOSECAST_OLD_TBB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tbb2020")
    if(NOT EXISTS "${JOSECAST_OLD_TBB_ROOT}/include/tbb/tbb.h")
        message(FATAL_ERROR "Old TBB 2020 headers not found at ${JOSECAST_OLD_TBB_ROOT}")
    endif()
    include_directories(BEFORE "${JOSECAST_OLD_TBB_ROOT}/include")
endif()

if(NOT Eigen3_FOUND)
    include_directories(BEFORE "${EIGEN3_INCLUDE_DIR}")
endif()

add_subdirectory(src)
