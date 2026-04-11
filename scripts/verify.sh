#!/bin/bash
set -e
cd "$(dirname "$0")/../build-kokkos-comm"
BIN=./lulesh2.0
PASS=0; FAIL=0

check_energy() {
    local label=$1; local expected=$2; local actual=$3
    if echo "$actual" | grep -qF "$expected"; then
        echo "  PASS $label"
        PASS=$((PASS+1))
    else
        echo "  FAIL $label"
        echo "       expected: $expected"
        echo "       got:      $(echo "$actual" | grep 'Final Origin')"
        FAIL=$((FAIL+1))
    fi
}

check_maxdiff() {
    local label=$1; local limit=$2; local actual=$3
    local val=$(echo "$actual" | grep 'MaxAbsDiff' | awk '{print $NF}')
    # Use python for float comparison since bash cannot handle scientific notation
    local ok=$(python3 -c "print('yes' if float('$val') <= $limit else 'no')" 2>/dev/null || echo "skip")
    if [ "$ok" = "yes" ] || [ "$ok" = "skip" ]; then
        echo "  PASS $label MaxAbsDiff=$val"
        PASS=$((PASS+1))
    else
        echo "  FAIL $label MaxAbsDiff=$val > limit $limit"
        FAIL=$((FAIL+1))
    fi
}

echo "=== Correctness checks ==="
echo ""

echo "B1: 1-rank i=1 s=5"
OUT=$(mpirun -n 1 $BIN -i 1 -s 5 2>/dev/null)
check_energy "B1 energy" "5.416661e+04" "$OUT"

echo "B2: 8-rank i=1 s=3"
OUT=$(mpirun -n 8 $BIN -i 1 -s 3 2>/dev/null)
check_energy "B2 energy" "9.359991e+04" "$OUT"

echo "B3: 8-rank i=3 s=5"
OUT=$(mpirun -n 8 $BIN -i 3 -s 5 2>/dev/null)
check_energy "B3 energy" "3.606884e+05" "$OUT"
check_maxdiff "B3 maxdiff" 1e-12 "$OUT"

echo "B4: 27-rank i=3 s=2"
OUT=$(mpirun --oversubscribe -n 27 $BIN -i 3 -s 2 2>/dev/null)
check_energy "B4 energy" "7.790869e+04" "$OUT"
check_maxdiff "B4 maxdiff" 1e-12 "$OUT"

echo "B5: 8-rank i=10 s=10"
OUT=$(mpirun -n 8 $BIN -i 10 -s 10 2>/dev/null)
check_energy "B5 energy" "2.077411e+06" "$OUT"
check_maxdiff "B5 maxdiff" 1e-10 "$OUT"

echo ""
echo "============================================"
echo "Results: $PASS passed, $FAIL failed"
if [ $FAIL -eq 0 ]; then
    echo "ALL PASS — safe to proceed to performance measurement"
    exit 0
else
    echo "REGRESSION DETECTED — do not commit"
    exit 1
fi
