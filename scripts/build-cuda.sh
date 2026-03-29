#!/usr/bin/env bash
# =============================================================================
# build-cuda.sh — 在 Linux + NVIDIA GPU 上重建 Kokkos 并编译 LULESH
#
# 使用场景：
#   当 Kokkos 安装时未启用 Kokkos_ENABLE_CUDA_LAMBDA，导致 KOKKOS_LAMBDA 宏
#   展开为 [=]（无 __device__ 属性），nvcc 拒绝在设备端调用该 lambda。
#   本脚本先重建 Kokkos（加入 CUDA_LAMBDA=ON），再重建 LULESH。
#
# 用法：
#   chmod +x scripts/build-cuda.sh
#   ./scripts/build-cuda.sh            # 使用默认路径
#   ./scripts/build-cuda.sh --help     # 显示参数说明
#
# 前置条件：
#   - NVIDIA GPU 驱动与 CUDA Toolkit 已安装（cuda >= 11.0）
#   - Kokkos 源码已下载（默认路径：~/kokkos）
#   - cmake >= 3.16，make，g++
# =============================================================================

set -euo pipefail  # 任何命令失败立即退出；未定义变量视为错误；管道失败也退出

# -----------------------------------------------------------------------------
# 0. 帮助信息
# -----------------------------------------------------------------------------
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<EOF
用法: $0 [选项]

选项:
  --kokkos-src DIR     Kokkos 源码目录         (默认: ~/kokkos)
  --kokkos-install DIR Kokkos 安装目标目录     (默认: ~/kokkos_cuda_install)
  --kokkos-build DIR   Kokkos 临时构建目录     (默认: /tmp/kokkos-build)
  --arch ARCH          GPU 架构 CMake 标识     (默认: AMPERE80)
                       常用值: VOLTA70 / TURING75 / AMPERE80 / HOPPER90
  --cuda-arch NUM      CUDA 架构号（数字）     (默认: 80)
                       与 --arch 对应: 70/75/80/90
  --lulesh-build DIR   LULESH 构建目录         (默认: ./build-cuda)
  --skip-kokkos        跳过 Kokkos 重建，直接构建 LULESH（Kokkos 已是正确版本时使用）
  -h, --help           显示此帮助

示例:
  # V100 GPU（Volta 架构）
  $0 --arch VOLTA70 --cuda-arch 70

  # A100 GPU（Ampere 架构，默认值）
  $0

  # 仅重建 LULESH（Kokkos 已重建完毕）
  $0 --skip-kokkos
EOF
    exit 0
fi

# -----------------------------------------------------------------------------
# 1. 参数解析（支持 --key value 风格覆盖默认值）
# -----------------------------------------------------------------------------
KOKKOS_SRC="/home/fzj/桌面/kokkos"
KOKKOS_INSTALL="${HOME}/kokkos_cuda_install"
KOKKOS_BUILD="/tmp/kokkos-build"
KOKKOS_ARCH="ADALOVE89"   # RTX 4070 对应架构
CUDA_ARCH="89"            # 数字形式 sm_89
LULESH_BUILD="./build-cuda"
SKIP_KOKKOS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kokkos-src)     KOKKOS_SRC="$2";     shift 2 ;;
        --kokkos-install) KOKKOS_INSTALL="$2"; shift 2 ;;
        --kokkos-build)   KOKKOS_BUILD="$2";   shift 2 ;;
        --arch)           KOKKOS_ARCH="$2";    shift 2 ;;
        --cuda-arch)      CUDA_ARCH="$2";      shift 2 ;;
        --lulesh-build)   LULESH_BUILD="$2";   shift 2 ;;
        --skip-kokkos)    SKIP_KOKKOS=1;       shift   ;;
        *) echo "[错误] 未知参数: $1"; exit 1 ;;
    esac
done

# nvcc_wrapper 路径由 Kokkos 源码提供，与源码目录绑定
NVCC_WRAPPER="${KOKKOS_SRC}/bin/nvcc_wrapper"

# -----------------------------------------------------------------------------
# 2. 路径校验
# -----------------------------------------------------------------------------
echo "============================================================"
echo " LULESH CUDA 构建脚本"
echo "============================================================"
echo " Kokkos 源码  : ${KOKKOS_SRC}"
echo " Kokkos 安装  : ${KOKKOS_INSTALL}"
echo " GPU 架构     : ${KOKKOS_ARCH} (sm_${CUDA_ARCH})"
echo " nvcc_wrapper : ${NVCC_WRAPPER}"
echo " LULESH 构建  : ${LULESH_BUILD}"
echo "============================================================"

if [[ $SKIP_KOKKOS -eq 0 ]]; then
    # 检查 Kokkos 源码目录是否存在
    if [[ ! -d "${KOKKOS_SRC}" ]]; then
        echo "[错误] Kokkos 源码目录不存在: ${KOKKOS_SRC}"
        echo "       请先克隆 Kokkos："
        echo "         git clone https://github.com/kokkos/kokkos.git ~/kokkos"
        echo "       或用 --kokkos-src 指定正确路径"
        exit 1
    fi

    # 检查 nvcc_wrapper 是否存在并可执行
    if [[ ! -x "${NVCC_WRAPPER}" ]]; then
        echo "[错误] nvcc_wrapper 不可执行: ${NVCC_WRAPPER}"
        echo "       请确认 Kokkos 源码包含 bin/nvcc_wrapper，或执行："
        echo "         chmod +x ${NVCC_WRAPPER}"
        exit 1
    fi
fi

# 检查 nvcc 是否在 PATH 中（CUDA Toolkit 是否安装）
if ! command -v nvcc &>/dev/null; then
    echo "[错误] 未找到 nvcc，请确认 CUDA Toolkit 已安装并加入 PATH"
    echo "       通常需要在 ~/.bashrc 中添加："
    echo "         export PATH=/usr/local/cuda/bin:\$PATH"
    exit 1
fi

echo "[信息] CUDA 版本: $(nvcc --version | grep 'release' | awk '{print $6}')"
echo "[信息] CMake 版本: $(cmake --version | head -1)"
echo ""

# -----------------------------------------------------------------------------
# 3. 重建 Kokkos（含 CUDA_LAMBDA=ON）
#
#    关键参数说明：
#    -DKokkos_ENABLE_CUDA=ON
#        启用 CUDA 后端，Kokkos::parallel_for 等将在 GPU 上执行
#
#    -DKokkos_ENABLE_CUDA_LAMBDA=ON
#        ★ 核心修复项 ★
#        使 KOKKOS_LAMBDA 宏展开为 [=] __host__ __device__
#        未启用时展开为 [=]（仅 host），nvcc 会报
#        "calling a constexpr __host__ function from __device__ function"
#
#    -DKokkos_ENABLE_OPENMP=ON
#        同时启用 OpenMP 后端（CPU 并行），两者可共存
#        注意：nvcc 编译 CUDA 文件时不自动传 -fopenmp，
#        CMakeLists.txt 中已通过 OpenMP::OpenMP_CXX 在 CXX 路径正确传递
#
#    -DKokkos_ARCH_AMPERE80=ON（或其他架构）
#        指定目标 GPU 的 PTX/SASS 架构，使 Kokkos 生成最优指令集
#        必须与实际 GPU 匹配，否则运行时报 "no kernel image" 错误
#
#    -DCMAKE_CXX_COMPILER=nvcc_wrapper
#        ★ 关键编译器选择 ★
#        nvcc_wrapper 是 Kokkos 提供的 nvcc 包装器：
#        它将 .cc/.cpp 文件内部重命名为 .cu 再交给 nvcc，
#        使 lambda 获得正确的外部链接名（无 _INTERNAL_ 前缀）。
#        如果直接用 nvcc 编译 .cc 文件，cudafe1 会给 lambda 加
#        _INTERNAL_ 前缀，导致模板参数解析失败。
# -----------------------------------------------------------------------------
if [[ $SKIP_KOKKOS -eq 0 ]]; then
    echo "------------------------------------------------------------"
    echo "步骤 1/4：配置 Kokkos（含 CUDA_LAMBDA=ON）"
    echo "------------------------------------------------------------"

    # 清空旧的 Kokkos 构建目录，避免缓存旧的 CMAKE_CACHE_VARIABLES
    rm -rf "${KOKKOS_BUILD}"

    cmake -S "${KOKKOS_SRC}" -B "${KOKKOS_BUILD}" \
        -DCMAKE_INSTALL_PREFIX="${KOKKOS_INSTALL}" \
        -DKokkos_ENABLE_CUDA=ON \
        -DKokkos_ENABLE_CUDA_LAMBDA=ON \
        -DKokkos_ENABLE_OPENMP=ON \
        "-DKokkos_ARCH_${KOKKOS_ARCH}=ON" \
        -DCMAKE_CXX_COMPILER="${NVCC_WRAPPER}" \
        -DCMAKE_BUILD_TYPE=Release

    echo ""
    echo "------------------------------------------------------------"
    echo "步骤 2/4：编译并安装 Kokkos"
    echo "------------------------------------------------------------"
    # -j$(nproc) 使用全部 CPU 核心并行编译
    cmake --build "${KOKKOS_BUILD}" -j"$(nproc)"

    # 将 Kokkos 安装到 KOKKOS_INSTALL 目录
    # （包含 lib/cmake/Kokkos/KokkosConfig.cmake，供 find_package 使用）
    cmake --install "${KOKKOS_BUILD}"

    echo ""
    echo "[完成] Kokkos 已安装至: ${KOKKOS_INSTALL}"
else
    echo "[跳过] Kokkos 重建（--skip-kokkos）"
    # 验证目标安装目录确实包含 Kokkos
    if [[ ! -f "${KOKKOS_INSTALL}/lib/cmake/Kokkos/KokkosConfig.cmake" ]]; then
        echo "[错误] ${KOKKOS_INSTALL} 中未找到 KokkosConfig.cmake"
        echo "       请先运行不带 --skip-kokkos 的完整构建"
        exit 1
    fi
fi

# -----------------------------------------------------------------------------
# 4. 重建 LULESH
#
#    参数说明：
#    -DCMAKE_PREFIX_PATH=KOKKOS_INSTALL
#        告知 CMake 的 find_package(Kokkos) 在此目录查找 KokkosConfig.cmake
#
#    -DCMAKE_CXX_COMPILER=nvcc_wrapper
#        同样使用 nvcc_wrapper 作为 C++ 编译器（而非 g++）
#        CMakeLists.txt 中通过 CMAKE_CXX_COMPILER MATCHES "nvcc_wrapper"
#        检测此路径，进入 Path A 分支：
#          - 不设置 LANGUAGE CUDA（避免 _INTERNAL_ lambda 名问题）
#          - --expt-extended-lambda 通过 CXX 路径传递给 nvcc_wrapper
#
#    -DCMAKE_CUDA_ARCHITECTURES=80
#        CMake CUDA 语言的架构参数，与 Kokkos_ARCH_AMPERE80 对应
#        （数字形式：70=Volta, 75=Turing, 80=Ampere, 90=Hopper）
#
#    -DCMAKE_BUILD_TYPE=Release
#        开启 -O3 优化，关闭调试符号
# -----------------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "步骤 3/4：配置 LULESH（指向新 Kokkos 安装）"
echo "------------------------------------------------------------"

# 清空旧的 LULESH 构建目录
rm -rf "${LULESH_BUILD}"
mkdir -p "${LULESH_BUILD}"

# LULESH 源码根目录（本脚本位于 scripts/ 子目录，../  即项目根）
LULESH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cmake -S "${LULESH_ROOT}" -B "${LULESH_BUILD}" \
    -DCMAKE_PREFIX_PATH="${KOKKOS_INSTALL}" \
    -DCMAKE_CXX_COMPILER="${NVCC_WRAPPER}" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DCMAKE_BUILD_TYPE=Release

echo ""
echo "------------------------------------------------------------"
echo "步骤 4/4：编译 LULESH（日志输出至 ${LULESH_BUILD}/make.log）"
echo "------------------------------------------------------------"

# 同时输出到终端和日志文件（tee）
# 2>&1 将 stderr（nvcc 警告/错误）合并到同一流
cmake --build "${LULESH_BUILD}" -j"$(nproc)" 2>&1 | tee "${LULESH_BUILD}/make.log"

# 检查最终二进制是否生成
BINARY="${LULESH_BUILD}/lulesh2.0"
if [[ ! -f "${BINARY}" ]]; then
    echo ""
    echo "[错误] 编译失败，未找到 ${BINARY}"
    echo "       详细错误见: ${LULESH_BUILD}/make.log"
    exit 1
fi

# -----------------------------------------------------------------------------
# 5. 正确性验证
#    参考值：Final Origin Energy = 8.104796e+04
#    -s 10 -i 50 ：10³ = 1000 个单元，迭代 50 步（轻量验证用例）
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " 构建成功！"
echo " 二进制路径 : ${BINARY}"
echo "============================================================"
echo ""
echo "正在运行正确性验证（-s 10 -i 50）..."
echo ""

"${BINARY}" -s 30 -i 100

echo ""
echo "期望输出: Final Origin Energy = 8.104796e+04"
echo ""
echo "性能基准运行（-s 45 -i 200）："
echo "  ${BINARY} -s 45 -i 200"
