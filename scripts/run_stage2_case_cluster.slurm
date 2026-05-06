#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=00:20:00

set -euo pipefail

# Make sure the module function and module-dependent library paths are
# initialized in non-interactive batch shells as well.
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi
fi

module purge
module use /home/software/modulefiles
module load openmpi/4.1.6

CASE_LABEL="${CASE_LABEL:?missing CASE_LABEL}"
EXEC_PATH="${EXEC_PATH:?missing EXEC_PATH}"
MPI_RANKS="${MPI_RANKS:?missing MPI_RANKS}"
OMP_THREADS="${OMP_THREADS:?missing OMP_THREADS}"
PROBLEM_SIZE="${PROBLEM_SIZE:?missing PROBLEM_SIZE}"
ITERATIONS="${ITERATIONS:?missing ITERATIONS}"
RESULTS_DIR="${RESULTS_DIR:?missing RESULTS_DIR}"
MPI_BINDING_MODE="${MPI_BINDING_MODE:-none}"
VERSION_NAME="${VERSION_NAME:?missing VERSION_NAME}"
SECTION_NAME="${SECTION_NAME:?missing SECTION_NAME}"

mkdir -p "$RESULTS_DIR"

export OMP_NUM_THREADS="$OMP_THREADS"
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

srun_cmd=(srun --mpi=pmix -n "$MPI_RANKS" -c "$OMP_THREADS")
if [[ "$MPI_BINDING_MODE" == "core" ]]; then
  srun_cmd+=(--cpu-bind=cores)
elif [[ "$MPI_BINDING_MODE" == "none" ]]; then
  srun_cmd+=(--cpu-bind=none)
fi
srun_cmd+=("$EXEC_PATH" -s "$PROBLEM_SIZE" -i "$ITERATIONS")

status="PASS"
if ! "${srun_cmd[@]}" > "$RESULTS_DIR/${CASE_LABEL}.log" 2>&1; then
  status="RUN_FAIL"
fi

actual_ranks="$(grep "MPI tasks" "$RESULTS_DIR/${CASE_LABEL}.log" | awk '{print $NF}' | tail -n 1 || true)"
elapsed_s="$(grep "Elapsed time" "$RESULTS_DIR/${CASE_LABEL}.log" | awk '{print $4}' | tail -n 1 || true)"
fom="$(grep "^FOM" "$RESULTS_DIR/${CASE_LABEL}.log" | awk '{print $3}' | tail -n 1 || true)"
energy="$(grep "Final Origin Energy" "$RESULTS_DIR/${CASE_LABEL}.log" | awk '{print $NF}' | tail -n 1 || true)"
max_abs_diff="$(grep "MaxAbsDiff" "$RESULTS_DIR/${CASE_LABEL}.log" | awk '{print $3}' | tail -n 1 || true)"

actual_ranks="${actual_ranks:-NA}"
elapsed_s="${elapsed_s:-NA}"
fom="${fom:-NA}"
energy="${energy:-NA}"
max_abs_diff="${max_abs_diff:-NA}"

if [[ "$status" == "PASS" && "$actual_ranks" != "$MPI_RANKS" ]]; then
  status="RANK_ERR"
fi

cat > "$RESULTS_DIR/${CASE_LABEL}.metrics.csv" <<EOF
case_id,version,section,mpi_ranks,omp_threads,size,iters,elapsed_s,fom,energy,max_abs_diff,status
${CASE_LABEL},${VERSION_NAME},${SECTION_NAME},${MPI_RANKS},${OMP_THREADS},${PROBLEM_SIZE},${ITERATIONS},${elapsed_s},${fom},${energy},${max_abs_diff},${status}
EOF

echo "[$CASE_LABEL] status=$status elapsed=$elapsed_s fom=$fom energy=$energy"
