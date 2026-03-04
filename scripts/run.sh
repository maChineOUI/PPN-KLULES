#!/usr/bin/env bash
# Run LULESH — common invocations
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build/release}"
BIN="$BUILD_DIR/lulesh2.0"

if [[ ! -x "$BIN" ]]; then
    echo "Binary not found: $BIN  (run scripts/build.sh first)" >&2
    exit 1
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
    *)
        echo "Usage: $0 [verify|bench]" >&2
        exit 1
        ;;
esac
