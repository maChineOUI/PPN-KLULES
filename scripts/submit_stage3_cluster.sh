#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export STAGE2_SECTIONS="${STAGE2_SECTIONS:-hybrid}"
export MPI_BINDING_MODE="${MPI_BINDING_MODE:-core}"
export HYBRID_LAYOUT_MODE="${HYBRID_LAYOUT_MODE:-auto}"
export OMP_PLACES_MODE="${OMP_PLACES_MODE:-cores}"
export OMP_PROC_BIND_MODE="${OMP_PROC_BIND_MODE:-close}"

STAGE_LABEL=stage3 exec "$SCRIPT_DIR/submit_stage2_binding_test.sh" "$@"
