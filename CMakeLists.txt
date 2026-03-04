cmake_minimum_required(VERSION 3.16)
project(LULESH VERSION 2.0 LANGUAGES CXX)

# --- 1. 基础编译设置 ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- 2. 区分系统查找 OpenMP ---
if(APPLE)
    # macOS 逻辑：Apple Clang 默认不带 OpenMP，通常需要 Homebrew 安装的 libomp
    message(STATUS "Detected macOS system")
    
    # 尝试查找 Homebrew 默认路径下的 OpenMP
    # 如果你通过 brew install libomp 安装，路径通常如下：
    if(NOT OpenMP_ROOT)
        set(OpenMP_ROOT "/usr/local/opt/libomp") # Intel Mac
        if(NOT EXISTS ${OpenMP_ROOT})
            set(OpenMP_ROOT "/opt/homebrew/opt/libomp") # Apple Silicon (M1/M2/M3) Mac
        endif()
    endif()

    find_package(OpenMP REQUIRED)
else()
    # Linux (Ubuntu) 逻辑：直接使用标准方式查找
    message(STATUS "Detected Linux system")
    find_package(OpenMP REQUIRED)
endif()

# 查找 Kokkos
find_package(Kokkos 5.0 REQUIRED)

# --- 3. 定义源文件 ---
set(SOURCES
    apps/lulesh_main.cc
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

# --- 4. 创建目标 ---
add_executable(lulesh ${SOURCES})
set_target_properties(lulesh PROPERTIES OUTPUT_NAME "lulesh2.0")

target_include_directories(lulesh PRIVATE include)

target_compile_options(lulesh PRIVATE
    -Wall
    -Wextra
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-g -O0>
)

# --- 5. 链接逻辑 ---
target_link_libraries(lulesh PRIVATE m)

# 无论什么系统，找到 OpenMP 后都使用这个标准 Target
if(OpenMP_CXX_FOUND)
    target_link_libraries(lulesh PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(lulesh PRIVATE Kokkos::kokkos)

# --- 6. 状态打印 ---
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenMP CXX Flags: ${OpenMP_CXX_FLAGS}")
message(STATUS "Kokkos devices: ${Kokkos_DEVICES}")