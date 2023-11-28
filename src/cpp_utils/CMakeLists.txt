cmake_minimum_required(VERSION 3.1...3.15)
project(voxeloffset LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
if(NOT POLICY CMP0070 AND POLICY CMP0053)
  # Only set CMP0053 to OLD with CMake<3.10, otherwise there is a warning.
  cmake_policy(SET CMP0053 OLD)
endif()

if(POLICY CMP0071)
  cmake_policy(SET CMP0071 NEW)
endif()

include(CMakeDependentOption)

# Find CGAL and CGAL Qt5
find_package(CGAL)

add_executable(self_intersect self_intersect.cpp)
target_link_libraries(self_intersect PRIVATE CGAL::CGAL)