cmake_minimum_required(VERSION 3.20)
project(pointcloud_io_demo LANGUAGES CXX)

option(USE_PDAL "Enable LAS/LAZ/E57/etc via PDAL" ON)
option(ENABLE_FINE_GICP "Enable per-fragment fine registration with voxel-based GICP" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- PCL ----
find_package(PCL CONFIG REQUIRED COMPONENTS
  common io io_ply filters
  registration kdtree search sample_consensus features
)

# ---- PDAL (optional) ----
set(HAVE_PDAL OFF)
if(USE_PDAL)
  find_package(PDAL CONFIG QUIET)
  if(PDAL_FOUND)
    set(HAVE_PDAL ON)
  else()
    message(STATUS "PDAL not found; continuing without it (extended formats disabled).")
  endif()
endif()

# ---- Locate PCL include root (your original logic) ----
if(NOT DEFINED _PCL_INC_ROOT)
  if(DEFINED ENV{VCPKG_ROOT})
    set(_PCL_INC_ROOT "$ENV{VCPKG_ROOT}/installed/x64-windows/include")
  else()
    set(_PCL_INC_ROOT "C:/Phd/vcpkg/installed/x64-windows/include")
  endif()
endif()

set(_PCL_CANDIDATES
  "${_PCL_INC_ROOT}/pcl-1.15" "${_PCL_INC_ROOT}/pcl-1.14"
  "${_PCL_INC_ROOT}/pcl-1.13" "${_PCL_INC_ROOT}/pcl-1.12"
  "${_PCL_INC_ROOT}/pcl" "${_PCL_INC_ROOT}"
)
set(PCL_INCLUDE_FIXED "")
foreach(dir IN LISTS _PCL_CANDIDATES)
  if(EXISTS "${dir}/pcl/io/auto_io.h" OR EXISTS "${dir}/pcl/io/pcd_io.h")
    list(APPEND PCL_INCLUDE_FIXED "${dir}")
  endif()
endforeach()
list(REMOVE_DUPLICATES PCL_INCLUDE_FIXED)
if(NOT PCL_INCLUDE_FIXED)
  message(FATAL_ERROR "Couldn't find PCL headers under ${_PCL_INC_ROOT}")
endif()

message(STATUS "Using PCL headers from: ${PCL_INCLUDE_FIXED}")
message(STATUS "USE_PDAL = ${USE_PDAL} (found: ${HAVE_PDAL})")
message(STATUS "ENABLE_FINE_GICP = ${ENABLE_FINE_GICP}")

# ---- helper to pick imported vs. plain targets ----
macro(pick_pcl out imported plain)
  if(TARGET ${imported})
    set(${out} ${imported})
  elseif(TARGET ${plain})
    set(${out} ${plain})
  else()
    message(FATAL_ERROR "PCL component not found: ${imported} or ${plain}")
  endif()
endmacro()

pick_pcl(_PCL_COMMON_TARGET         PCL::common           pcl_common)
pick_pcl(_PCL_IO_TARGET             PCL::io               pcl_io)
pick_pcl(_PCL_IO_PLY_TARGET         PCL::io_ply           pcl_io_ply)
pick_pcl(_PCL_FILTERS_TARGET        PCL::filters          pcl_filters)
pick_pcl(_PCL_REG_TARGET            PCL::registration     pcl_registration)
pick_pcl(_PCL_KDTREE_TARGET         PCL::kdtree           pcl_kdtree)
pick_pcl(_PCL_SEARCH_TARGET         PCL::search           pcl_search)
pick_pcl(_PCL_SAC_TARGET            PCL::sample_consensus pcl_sample_consensus)
pick_pcl(_PCL_FEATURES_TARGET       PCL::features         pcl_features)

# ---- pointcloud_io library ----
add_library(pointcloud_io STATIC
  src/pointcloud_io.cpp
  src/preprocess.cpp
  src/preprocess_matlab.cpp
)

target_include_directories(pointcloud_io PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${PCL_INCLUDE_FIXED}
)
target_link_libraries(pointcloud_io PUBLIC
  ${_PCL_COMMON_TARGET}
  ${_PCL_IO_TARGET}
  ${_PCL_IO_PLY_TARGET}
)
if(HAVE_PDAL)
  if(TARGET PDAL::pdalcpp)
    target_link_libraries(pointcloud_io PUBLIC PDAL::pdalcpp)
  else()
    target_link_libraries(pointcloud_io PUBLIC pdalcpp)
  endif()
  target_compile_definitions(pointcloud_io PUBLIC USE_PDAL)
endif()

# ---- executable ----
add_executable(read_point_cloud
  src/main.cpp
  src/gror_registration.cpp
  src/gror_pre.cpp
  src/fragmentation.cpp
  src/semisphere.cpp
)

# Fine GICP refinement (optional module integrated into the main pipeline)
if(ENABLE_FINE_GICP)
  target_sources(read_point_cloud PRIVATE
    src/gicp_refine.cpp
  )
  target_compile_definitions(read_point_cloud PRIVATE ENABLE_FINE_GICP)
endif()

target_include_directories(read_point_cloud PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${PCL_INCLUDE_FIXED}
)

target_link_libraries(read_point_cloud PRIVATE
  pointcloud_io
  ${_PCL_FILTERS_TARGET}
  ${_PCL_REG_TARGET}
  ${_PCL_KDTREE_TARGET}
  ${_PCL_SEARCH_TARGET}
  ${_PCL_SAC_TARGET}
  ${_PCL_FEATURES_TARGET}
)

# Make sure banner in main sees USE_PDAL too
if(HAVE_PDAL)
  target_compile_definitions(read_point_cloud PRIVATE USE_PDAL)
endif()

# ---- toolchain quirks ----
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
  target_link_libraries(pointcloud_io PUBLIC stdc++fs)
  target_link_libraries(read_point_cloud PRIVATE stdc++fs)
endif()

if(MSVC)
  target_compile_options(pointcloud_io    PRIVATE /permissive- /Zc:__cplusplus)
  target_compile_options(read_point_cloud PRIVATE /permissive- /Zc:__cplusplus)
  add_definitions(-DNOMINMAX)
endif()
