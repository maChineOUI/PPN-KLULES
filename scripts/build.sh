#!/usr/bin/env bash
# LULESH Build Script: Compatible with macOS (Apple Silicon) and Linux (Ubuntu)
set -euo pipefail

# --- Parameters ---
# Argument 1: Build type (Release/Debug), default: Release
BUILD_TYPE="${1:-Release}"
# Argument 2: If "clean", delete the existing build directory
MODE="${2:-""}"

BUILD_DIR="build/${BUILD_TYPE,,}"

# --- 1. Automatic Cleaning Logic ---
if [ "$MODE" == "clean" ]; then
    echo "--- [Action] Cleaning build directory... ---"
    rm -rf "$BUILD_DIR"
fi

# Ensure build directory exists
mkdir -p "$BUILD_DIR"

# --- 2. OS Detection and CMake Configuration ---
OS_NAME="$(uname -s)"

if [ "$OS_NAME" = "Darwin" ]; then
    echo "--- [OS] Detected macOS ---"
    # Mac specific: Using Homebrew paths for Kokkos and OpenMP
    cmake -S . -B "$BUILD_DIR" \
        -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/kokkos \
        -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
        -DOpenMP_CXX_LIB_NAMES="omp" \
        -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    
    # Get CPU cores on Mac
    CPU_CORES=$(sysctl -n hw.ncpu)

else
    echo "--- [OS] Detected Linux/Ubuntu ---"
    # Linux specific: Use GCC 13 and automatic detection
    export CC=gcc-13
    export CXX=g++-13

    # On Linux, let CMake find OpenMP naturally (avoiding Mac paths)
    cmake -S . -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    
    # Get CPU cores on Linux
    CPU_CORES=$(nproc)
fi

# --- 3. Execution ---
echo "--- [Build] Starting build with $CPU_CORES cores ---"
cmake --build "$BUILD_DIR" -j"$CPU_CORES"

echo "---------------------------------------"
echo "Build Successful!"
echo "Binary: $BUILD_DIR/lulesh2.0"
echo "Tip: To reset cache, run: $0 $BUILD_TYPE clean"