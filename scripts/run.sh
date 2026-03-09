#!/usr/bin/env bash
# Run LULESH — common invocations
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build/release}"
BIN="$BUILD_DIR/lulesh2.0"

if [[ ! -x "$BIN" ]]; then
    echo "Binary not found: $BIN  (run scripts/build.sh first)" >&2
    exit 1
fi

# On Apple Silicon, P-cores and E-cores have very different speeds.
# A Kokkos barrier waits for the slowest thread, so mixing core types
# degrades performance. Default to P-core count only when detectable.
if [[ "$(uname -s)" == "Darwin" ]] && sysctl -n hw.perflevel0.physicalcpu &>/dev/null; then
    P_CORES=$(sysctl -n hw.perflevel0.physicalcpu)
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-$P_CORES}"
    export OMP_NUM_THREADS
    echo "--- [Info] Apple Silicon: using $OMP_NUM_THREADS P-core(s) (set OMP_NUM_THREADS to override) ---"
fi

MODE="${1:-verify}"

case "$MODE" in
    verify)
        # Correctness check: expect Final Origin Energy = 8.104796e+04
        OMP_PROC_BIND=close "$BIN" -s 10 -i 50
        ;;
    bench)
        # Performance benchmark
        OMP_PROC_BIND=close "$BIN" -s 45 -i 200
        ;;
    scale)
        # Scalability test: T=1,2,4 (all P-cores)
        MAX_T="${OMP_NUM_THREADS:-4}"
        for t in 1 2 "$MAX_T"; do
            echo "--- T=$t ---"
            OMP_NUM_THREADS=$t OMP_PROC_BIND=close "$BIN" -s 45 -i 100
        done
        ;;
    *)
        echo "Usage: $0 [verify|bench|scale]" >&2
        exit 1
        ;;
esac
