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

# --- 6. CUDA backend 支持（可选）---
# 若 Kokkos 安装时开启了 CUDA，自动将所有含 KOKKOS_LAMBDA 的源文件标记为 CUDA 语言，
# 使 nvcc_wrapper 能够编译设备端代码。
# 构建命令示例（Linux + NVIDIA GPU）：
#   cmake -S . -B build-cuda \
#         -DCMAKE_CXX_COMPILER=/path/to/kokkos/bin/nvcc_wrapper \
#         -DKokkos_ENABLE_CUDA=ON \
#         -DKokkos_ENABLE_CUDA_LAMBDA=ON \
#         -DKokkos_ARCH_AMPERE80=ON \
#         -DCMAKE_BUILD_TYPE=Release
kokkos_check(DEVICES "CUDA" RETURN_VALUE HAVE_CUDA)
if(HAVE_CUDA)
    enable_language(CUDA)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    set_target_properties(lulesh PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "Kokkos CUDA backend detected")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    if(CMAKE_CXX_COMPILER MATCHES "nvcc_wrapper")
        # ── Path A: nvcc_wrapper as CXX compiler (recommended) ──────────────
        # nvcc_wrapper intercepts every CXX compilation and internally renames
        # .cc files to .cu before passing them to nvcc, so lambdas receive
        # proper external-linkage names (no _INTERNAL_ prefix).
        #
        # IMPORTANT: do NOT set LANGUAGE CUDA on .cc source files.
        # Setting LANGUAGE CUDA causes CMake to use CMAKE_CUDA_COMPILER (nvcc
        # directly) instead of CMAKE_CXX_COMPILER (nvcc_wrapper).  nvcc called
        # directly on a .cc file gives lambdas an _INTERNAL_ prefix that its
        # own cudafe1 tool cannot demangle (template argument becomes '::').
        #
        # Instead, pass --expt-extended-lambda as a CXX flag.  nvcc_wrapper
        # forwards all CXX flags to nvcc, so the flag reaches nvcc correctly.
        # OpenMP::OpenMP_CXX supplies -fopenmp via the same CXX path, so
        # _OPENMP is defined without needing -Xcompiler=-fopenmp explicitly.
        message(STATUS "CUDA compiler: nvcc_wrapper — compiling all sources as CXX (nvcc_wrapper handles CUDA internally)")
        target_compile_options(lulesh PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:--expt-extended-lambda>
        )
        message(STATUS "CUDA extended lambda support enabled (--expt-extended-lambda via CXX flags)")
    else()
        # ── Path B: g++ + kokkos_launch_compiler ────────────────────────────
        # kokkos_launch_compiler intercepts CXX commands flagged with
        # -DKOKKOS_DEPENDENCE and redirects them to nvcc.  Setting
        # LANGUAGE CUDA here would make CMake call nvcc *directly* on .cc
        # files, which causes nvcc to give lambdas an _INTERNAL_ prefix that
        # its own cudafe1 tool cannot demangle (template arg becomes '::').
        # Solution: do NOT set LANGUAGE CUDA — let kokkos_launch_compiler
        # handle the routing.  --expt-extended-lambda must be supplied by
        # rebuilding Kokkos with -DKokkos_ENABLE_CUDA_LAMBDA=ON, or by
        # switching to nvcc_wrapper (recommended).
        message(STATUS "CUDA compiler: kokkos_launch_compiler (g++ → nvcc redirect)")
        message(WARNING
            "Using g++ with kokkos_launch_compiler.\n"
            "  LANGUAGE CUDA is NOT set on kernel sources to avoid _INTERNAL_ lambda\n"
            "  name issues with nvcc's cudafe1 tool.\n"
            "  KOKKOS_LAMBDA support requires Kokkos to be built with\n"
            "  -DKokkos_ENABLE_CUDA_LAMBDA=ON.\n"
            "  For a fully supported build, use nvcc_wrapper as CMAKE_CXX_COMPILER:\n"
            "    cmake ... -DCMAKE_CXX_COMPILER=/path/to/kokkos/bin/nvcc_wrapper")
    endif()
endif()

# --- 7. 状态打印 ---
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenMP CXX Flags: ${OpenMP_CXX_FLAGS}")
message(STATUS "Kokkos devices: ${Kokkos_DEVICES}")