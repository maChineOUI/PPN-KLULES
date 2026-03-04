cmake_minimum_required(VERSION 3.16)
project(LULESH VERSION 2.0 LANGUAGES CXX)

# C++20, serial build
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(SOURCES
    src/lulesh.cc
    src/lulesh-geometry.cc
    src/lulesh-stress.cc
    src/lulesh-nodal.cc
    src/lulesh-kinematics.cc
    src/lulesh-viscosity.cc
    src/lulesh-eos.cc
    src/lulesh-timestep.cc
    src/lulesh-integration.cc
    src/lulesh-init.cc
    src/lulesh-util.cc
)

add_executable(lulesh ${SOURCES})
set_target_properties(lulesh PROPERTIES OUTPUT_NAME "lulesh2.0")

target_include_directories(lulesh PRIVATE include)

target_compile_options(lulesh PRIVATE
    -Wall
    -Wextra
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:RelWithDebInfo>:-O2 -g -fno-omit-frame-pointer>
)

target_link_libraries(lulesh PRIVATE m)

# Kokkos (required — OpenMP backend on CPU)
find_package(Kokkos 5.0 REQUIRED)
target_link_libraries(lulesh PRIVATE Kokkos::kokkos)
message(STATUS "Kokkos found — backend: ${Kokkos_DEVICES}")
