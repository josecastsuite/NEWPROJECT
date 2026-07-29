cmake_minimum_required(VERSION 3.15...3.22)
project(josecast_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use old TBB 2020 headers for ABI compatibility with system libopenvdb.
set(JOSECAST_OLD_TBB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tbb2020")
if(NOT EXISTS "${JOSECAST_OLD_TBB_ROOT}/include/tbb/tbb.h")
    message(FATAL_ERROR "Old TBB 2020 headers not found at ${JOSECAST_OLD_TBB_ROOT}")
endif()

# Python + nanobind -------------------------------------------------
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c
        "import nanobind, os; print(os.path.join(os.path.dirname(nanobind.__file__), 'cmake'))"
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(nanobind CONFIG REQUIRED)

# OpenVDB / support libs --------------------------------------------
find_library(OPENVDB_LIBRARY openvdb REQUIRED)
find_library(BLOSC_LIBRARY blosc REQUIRED)
find_library(BOOST_IOSTREAMS_LIBRARY boost_iostreams REQUIRED)
find_library(HALF_LIBRARY NAMES Half Half-2_5 REQUIRED)

# libopenvdb was linked against the old libtbb.so.2 runtime.
find_library(TBB_LIBRARY
    NAMES libtbb.so.2 tbb
    PATHS /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu
    NO_DEFAULT_PATH
)
if(NOT TBB_LIBRARY)
    set(TBB_LIBRARY "/usr/lib/x86_64-linux-gnu/libtbb.so.2")
endif()

# log4cplus soname is not a plain liblog4cplus.so on Ubuntu 22.04.
find_library(LOG4CPLUS_LIBRARY
    NAMES log4cplus-2.0 log4cplus
    PATHS /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu
)
if(NOT LOG4CPLUS_LIBRARY)
    set(LOG4CPLUS_LIBRARY "/lib/x86_64-linux-gnu/liblog4cplus-2.0.so.3")
endif()

# AMGCL header-only Poisson/linear solver ---------------------------
set(AMGCL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/amgcl")
if(NOT EXISTS "${AMGCL_ROOT}/amgcl/make_solver.hpp")
    message(FATAL_ERROR "AMGCL headers not found at ${AMGCL_ROOT}")
endif()

include_directories(
    BEFORE
    "${JOSECAST_OLD_TBB_ROOT}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${AMGCL_ROOT}"
)

add_subdirectory(src)
