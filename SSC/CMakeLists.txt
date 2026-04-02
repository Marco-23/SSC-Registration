cmake_minimum_required(VERSION 3.16)
project(ssc_full_pipeline LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PCL REQUIRED COMPONENTS common io filters features kdtree)
find_package(PCL QUIET COMPONENTS visualization)

set(SSC_SOURCES
    src/main.cpp
    src/utilities.cpp
    src/dynamic_seed_growth.cpp
    src/calculate_quality.cpp
    src/file_io.cpp
    src/ssc_pipeline.cpp
    src/visualization_utils.cpp)

add_executable(ssc_full_pipeline ${SSC_SOURCES})

target_include_directories(ssc_full_pipeline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PCL_INCLUDE_DIRS})

# Some PCL installs expose imported targets (PCL::common, etc.), others only expose
# the legacy variables PCL_LIBRARIES / PCL_DEFINITIONS / PCL_LIBRARY_DIRS.
# This project uses the legacy variables for maximum compatibility.
if(PCL_LIBRARY_DIRS)
    target_link_directories(ssc_full_pipeline PRIVATE ${PCL_LIBRARY_DIRS})
endif()

if(PCL_DEFINITIONS)
    separate_arguments(_pcl_defs NATIVE_COMMAND "${PCL_DEFINITIONS}")
    target_compile_options(ssc_full_pipeline PRIVATE ${_pcl_defs})
endif()

if(PCL_COMPILE_OPTIONS)
    target_compile_options(ssc_full_pipeline PRIVATE ${PCL_COMPILE_OPTIONS})
endif()

target_link_libraries(ssc_full_pipeline PRIVATE ${PCL_LIBRARIES})

if(PCL_visualization_FOUND)
    target_compile_definitions(ssc_full_pipeline PRIVATE SSC_HAS_PCL_VIS=1)
else()
    target_compile_definitions(ssc_full_pipeline PRIVATE SSC_HAS_PCL_VIS=0)
endif()
