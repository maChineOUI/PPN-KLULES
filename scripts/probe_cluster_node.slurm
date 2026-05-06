#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=00:10:00

set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi
fi

RESULTS_DIR="${RESULTS_DIR:?missing RESULTS_DIR}"
CASE_LABEL="${CASE_LABEL:?missing CASE_LABEL}"
BASELINE_EXEC="${BASELINE_EXEC:?missing BASELINE_EXEC}"
PROBLEM_SIZE="${PROBLEM_SIZE:-10}"
ITERATIONS="${ITERATIONS:-10}"

mkdir -p "$RESULTS_DIR"

LOG_PATH="$RESULTS_DIR/${CASE_LABEL}.log"
CSV_PATH="$RESULTS_DIR/${CASE_LABEL}.csv"

module purge
module use /home/software/modulefiles
module load openmpi/4.1.6

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

get_lscpu_field() {
  local pattern=$1
  local value
  value="$(lscpu | awk -F: -v pat="$pattern" '$1 ~ pat {gsub(/^[ \t]+/, "", $2); print $2; exit}')"
  if [[ -n "$value" ]]; then
    printf "%s" "$value"
  else
    printf "NA"
  fi
}

host_name="$(hostname)"
model_name="$(get_lscpu_field '^Model name$')"
threads_per_core="$(get_lscpu_field '^Thread\\(s\\) per core$')"
cores_per_socket="$(get_lscpu_field '^Core\\(s\\) per socket$')"
sockets="$(get_lscpu_field '^Socket\\(s\\)$')"
numa_nodes="$(get_lscpu_field '^NUMA node\\(s\\)$')"
cpu_count="$(get_lscpu_field '^CPU\\(s\\)$')"
allocated_cpu_count="$(nproc)"
node0_cpus="$(get_lscpu_field '^NUMA node0 CPU\\(s\\)$')"
node1_cpus="$(get_lscpu_field '^NUMA node1 CPU\\(s\\)$')"
node2_cpus="$(get_lscpu_field '^NUMA node2 CPU\\(s\\)$')"
node3_cpus="$(get_lscpu_field '^NUMA node3 CPU\\(s\\)$')"

{
  echo "=== hostname ==="
  echo "$host_name"
  echo
  echo "=== lscpu summary ==="
  lscpu | egrep 'Model name|Socket|Core|Thread|NUMA'
  echo
  echo "=== numactl --hardware ==="
  numactl --hardware || true
  echo
  echo "=== module list ==="
  module list 2>&1 || true
  echo
  echo "=== mpirun smoke ==="
} > "$LOG_PATH"

status="PASS"
if ! mpirun -np 1 "$BASELINE_EXEC" -s "$PROBLEM_SIZE" -i "$ITERATIONS" >> "$LOG_PATH" 2>&1; then
  status="RUN_FAIL"
fi

if grep -q 'libpmix.so.2' "$LOG_PATH"; then
  status="PMIX_FAIL"
fi

actual_ranks="$(grep "MPI tasks" "$LOG_PATH" | awk '{print $NF}' | tail -n 1 || true)"
elapsed_s="$(grep "Elapsed time" "$LOG_PATH" | awk '{print $4}' | tail -n 1 || true)"
fom="$(grep "^FOM" "$LOG_PATH" | awk '{print $3}' | tail -n 1 || true)"
energy="$(grep "Final Origin Energy" "$LOG_PATH" | awk '{print $NF}' | tail -n 1 || true)"
pmix_fail="no"
if grep -q 'libpmix.so.2' "$LOG_PATH"; then
  pmix_fail="yes"
fi

if grep -q 'Run completed:' "$LOG_PATH"; then
  smoke_ok="yes"
else
  smoke_ok="no"
  if [[ "$status" == "PASS" ]]; then
    status="NO_COMPLETION"
  fi
fi

printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
  "node" \
  "model_name" \
  "threads_per_core" \
  "cores_per_socket" \
  "sockets" \
  "numa_nodes" \
  "cpu_count" \
  "allocated_cpu_count" \
  "node0_cpus" \
  "node1_cpus" \
  "node2_cpus" \
  "node3_cpus" \
  "smoke_ok" \
  "status" \
  "actual_ranks" \
  "elapsed_s" \
  "fom" \
  "energy" \
  "pmix_fail" > "$CSV_PATH"

printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
  "$host_name" \
  "$model_name" \
  "$threads_per_core" \
  "$cores_per_socket" \
  "$sockets" \
  "$numa_nodes" \
  "$cpu_count" \
  "$allocated_cpu_count" \
  "$node0_cpus" \
  "$node1_cpus" \
  "$node2_cpus" \
  "$node3_cpus" \
  "$smoke_ok" \
  "$status" \
  "${actual_ranks:-NA}" \
  "${elapsed_s:-NA}" \
  "${fom:-NA}" \
  "${energy:-NA}" \
  "$pmix_fail" >> "$CSV_PATH"

echo "[$CASE_LABEL] host=$host_name status=$status smoke_ok=$smoke_ok"
