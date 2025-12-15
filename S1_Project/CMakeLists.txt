cmake_minimum_required(VERSION 3.16)

# ---------------
# 项目信息
# ---------------
project(LULESH_Kokkos
        VERSION 1.0
        LANGUAGES CXX)

# 强制 C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------
# 让用户可以指定 Kokkos_DIR
# -----------------------------
# 用法：
#   cmake -DKokkos_DIR=/Users/wei/kokkos-install/lib/cmake/Kokkos ..
#
if(NOT Kokkos_DIR)
    message(FATAL_ERROR "Kokkos_DIR is not set. Please run:
    cmake -DKokkos_DIR=/Users/wei/kokkos-install/lib/cmake/Kokkos ..")
endif()

# -----------------------------
# 查找 Kokkos
# -----------------------------
find_package(Kokkos REQUIRED)

message(STATUS "Kokkos found at: ${Kokkos_DIR}")
message(STATUS "Kokkos version: ${Kokkos_VERSION}")
message(STATUS "Kokkos options: ${Kokkos_ENABLED_DEVICES}")

# -----------------------------
# 源文件
# -----------------------------
set(LULESH_SOURCES
    lulesh.cc
    lulesh-init.cc
    lulesh-util.cc
    lulesh-comm.cc
    lulesh-viz.cc
)

# -----------------------------
# 设置可执行文件
# -----------------------------
add_executable(lulesh.host ${LULESH_SOURCES})

# -----------------------------
# 链接 Kokkos
# -----------------------------
target_link_libraries(lulesh.host Kokkos::kokkos)

# -----------------------------
# 包含路径
# -----------------------------
target_include_directories(lulesh.host PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# -----------------------------
# 编译优化（根据编译器自动选择）
# -----------------------------
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(lulesh.host PRIVATE -O3 -Wall -Wextra -march=native)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(lulesh.host PRIVATE -O3 -Wall -Wextra -march=native)
endif()

# -----------------------------
# 构建输出信息
# -----------------------------
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Executable: lulesh.host")
