#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=00:40:00

set -euo pipefail

echo "[stage4] $(date '+%F %T') batch script start"
echo "[stage4] submit host: $(hostname)"

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi
fi

module purge
module use /home/software/modulefiles
module load openmpi/4.1.6
echo "[stage4] $(date '+%F %T') modules loaded"
module list 2>&1 || true

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
SCALING_MODE="${SCALING_MODE:?missing SCALING_MODE}"
NODE_COUNT="${NODE_COUNT:?missing NODE_COUNT}"
RANKS_PER_NODE="${RANKS_PER_NODE:?missing RANKS_PER_NODE}"
BASE_SIZE="${BASE_SIZE:?missing BASE_SIZE}"

mkdir -p "$RESULTS_DIR"

export OMP_NUM_THREADS="$OMP_THREADS"
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

echo "[stage4] case_label=$CASE_LABEL"
echo "[stage4] version=$VERSION_NAME section=$SECTION_NAME scaling=$SCALING_MODE"
echo "[stage4] nodes=$NODE_COUNT ranks_per_node=$RANKS_PER_NODE mpi_ranks=$MPI_RANKS omp_threads=$OMP_THREADS"
echo "[stage4] base_size=$BASE_SIZE actual_size=$PROBLEM_SIZE iterations=$ITERATIONS"
echo "[stage4] exec_path=$EXEC_PATH"
echo "[stage4] working_dir=$(pwd)"
echo "[stage4] slurm_job_id=${SLURM_JOB_ID:-NA} nodelist=${SLURM_JOB_NODELIST:-NA}"

mpirun_cmd=(mpirun -np "$MPI_RANKS")
if [[ "$MPI_BINDING_MODE" == "core" ]]; then
  mpirun_cmd+=(--map-by "ppr:${RANKS_PER_NODE}:node:PE=${OMP_THREADS}")
  mpirun_cmd+=(--bind-to core)
fi
mpirun_cmd+=("$EXEC_PATH" -s "$PROBLEM_SIZE" -i "$ITERATIONS")

printf '[stage4] mpirun command:'
printf ' %q' "${mpirun_cmd[@]}"
printf '\n'
echo "[stage4] $(date '+%F %T') launching mpirun"

status="PASS"
if ! "${mpirun_cmd[@]}" > "$RESULTS_DIR/${CASE_LABEL}.log" 2>&1; then
  status="RUN_FAIL"
fi
echo "[stage4] $(date '+%F %T') mpirun finished with status=$status"

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
case_id,version,section,scaling_mode,node_count,ranks_per_node,mpi_ranks,omp_threads,base_size,size,iters,elapsed_s,fom,energy,max_abs_diff,status
${CASE_LABEL},${VERSION_NAME},${SECTION_NAME},${SCALING_MODE},${NODE_COUNT},${RANKS_PER_NODE},${MPI_RANKS},${OMP_THREADS},${BASE_SIZE},${PROBLEM_SIZE},${ITERATIONS},${elapsed_s},${fom},${energy},${max_abs_diff},${status}
EOF

echo "[$CASE_LABEL] status=$status nodes=$NODE_COUNT np=$MPI_RANKS t=$OMP_THREADS s=$PROBLEM_SIZE elapsed=$elapsed_s fom=$fom"
echo "[stage4] $(date '+%F %T') batch script end"
