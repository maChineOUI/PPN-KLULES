#!/usr/bin/env bash
# Build LULESH with Kokkos/OpenMP backend on macOS (Apple Silicon)
set -euo pipefail

BUILD_TYPE="${1:-Release}"
BUILD_DIR="build/${BUILD_TYPE,,}"   # lowercase: release / relwithdebinfo / debug

cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_PREFIX_PATH=/opt/homebrew/Cellar/kokkos/5.0.2 \
    -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
    -DOpenMP_CXX_LIB_NAMES="omp" \
    -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

cmake --build "$BUILD_DIR" -j"$(sysctl -n hw.ncpu)"
echo "Binary: $BUILD_DIR/lulesh2.0"
