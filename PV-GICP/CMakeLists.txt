# ============================================================================
# PlaneBasedGICP
# ----------------------------------------------------------------------------
# Original implementation written by the repository author.
#
# Related article:
#   "Quality-controlled registration of urban MLS point clouds reducing drift
#   effects by adaptive fragmentation"
#   Marco Antonio Ortiz Rincón, Yihui Yang, Christoph Holst
#   International Journal of Applied Earth Observation and Geoinformation
#   149 (2026) 105272
#   DOI: 10.1016/j.jag.2026.105272
#
# This CMake file builds the documented PV-GICP example contained in
# FinalGICP.cpp.
# ============================================================================

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(PlaneBasedGICP VERSION 1.0 LANGUAGES CXX)

# PCL modules used by the program.
find_package(PCL 1.8 REQUIRED COMPONENTS common io features registration search)

# Apply include paths, library directories, and compile definitions exported by PCL.
include_directories(SYSTEM ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Build the executable.
add_executable(gicp PV-GICP.cpp)

# std::filesystem is used in the source file, so C++17 is required.
target_compile_features(gicp PRIVATE cxx_std_17)

# Link all required PCL libraries.
target_link_libraries(gicp PRIVATE ${PCL_LIBRARIES})
