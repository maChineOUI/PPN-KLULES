#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=00:20:00

set -euo pipefail

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
MPI_BINDING_MODE="${MPI_BINDING_MODE:-core}"
MPI_REPORT_BINDINGS="${MPI_REPORT_BINDINGS:-no}"
VERSION_NAME="${VERSION_NAME:?missing VERSION_NAME}"
SECTION_NAME="${SECTION_NAME:?missing SECTION_NAME}"
HYBRID_LAYOUT_MODE="${HYBRID_LAYOUT_MODE:-auto}"
OMP_PLACES_MODE="${OMP_PLACES_MODE:-}"
OMP_PROC_BIND_MODE="${OMP_PROC_BIND_MODE:-}"
OMP_DISPLAY_ENV_MODE="${OMP_DISPLAY_ENV_MODE:-no}"
OMP_DISPLAY_AFFINITY_MODE="${OMP_DISPLAY_AFFINITY_MODE:-no}"

mkdir -p "$RESULTS_DIR"

export OMP_NUM_THREADS="$OMP_THREADS"
if [[ -n "$OMP_PLACES_MODE" ]]; then
  export OMP_PLACES="$OMP_PLACES_MODE"
elif [[ "$SECTION_NAME" == "hybrid" ]]; then
  export OMP_PLACES=cores
else
  export OMP_PLACES=threads
fi

if [[ -n "$OMP_PROC_BIND_MODE" ]]; then
  export OMP_PROC_BIND="$OMP_PROC_BIND_MODE"
elif [[ "$SECTION_NAME" == "hybrid" ]]; then
  export OMP_PROC_BIND=close
else
  export OMP_PROC_BIND=spread
fi

if [[ "$OMP_DISPLAY_ENV_MODE" == "yes" ]]; then
  export OMP_DISPLAY_ENV=TRUE
fi
if [[ "$OMP_DISPLAY_AFFINITY_MODE" == "yes" ]]; then
  export OMP_DISPLAY_AFFINITY=TRUE
fi

mpirun_cmd=(mpirun)
if [[ "$MPI_REPORT_BINDINGS" == "yes" ]]; then
  mpirun_cmd+=(--report-bindings)
fi

if [[ "$SECTION_NAME" == "hybrid" ]]; then
  case "$MPI_BINDING_MODE" in
    core)
      case "$HYBRID_LAYOUT_MODE" in
        auto)
          if [[ "$MPI_RANKS" == "8" ]]; then
            # On node60-class nodes this places two ranks per NUMA domain and
            # gives each rank OMP_THREADS cores.
            mpirun_cmd+=(--map-by "ppr:2:numa:PE=${OMP_THREADS}" --bind-to core)
          else
            mpirun_cmd+=(--map-by "slot:PE=${OMP_THREADS}" --bind-to core)
          fi
          ;;
        numa2x)
          mpirun_cmd+=(--map-by "ppr:2:numa:PE=${OMP_THREADS}" --bind-to core)
          ;;
        slotpe)
          mpirun_cmd+=(--map-by "slot:PE=${OMP_THREADS}" --bind-to core)
          ;;
        none)
          ;;
        *)
          echo "Unsupported HYBRID_LAYOUT_MODE=$HYBRID_LAYOUT_MODE" >&2
          exit 2
          ;;
      esac
      ;;
    hwthread)
      mpirun_cmd+=(--map-by "slot:PE=${OMP_THREADS}" --bind-to hwthread)
      ;;
    none)
      ;;
    *)
      echo "Unsupported MPI_BINDING_MODE=$MPI_BINDING_MODE" >&2
      exit 2
      ;;
  esac
else
  case "$MPI_BINDING_MODE" in
    core)
      mpirun_cmd+=(--bind-to core --map-by core)
      ;;
    hwthread)
      mpirun_cmd+=(--bind-to hwthread --map-by core)
      ;;
    none)
      ;;
    *)
      echo "Unsupported MPI_BINDING_MODE=$MPI_BINDING_MODE" >&2
      exit 2
      ;;
  esac
fi

mpirun_cmd+=(-np "$MPI_RANKS" "$EXEC_PATH" -s "$PROBLEM_SIZE" -i "$ITERATIONS")

status="PASS"
{
  echo "CASE_LABEL=$CASE_LABEL"
  echo "SECTION_NAME=$SECTION_NAME"
  echo "MPI_BINDING_MODE=$MPI_BINDING_MODE"
  echo "HYBRID_LAYOUT_MODE=$HYBRID_LAYOUT_MODE"
  echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
  echo "OMP_PLACES=$OMP_PLACES"
  echo "OMP_PROC_BIND=$OMP_PROC_BIND"
  echo "MPI_CMD=${mpirun_cmd[*]}"
  lscpu | egrep 'Model name|Socket|Core|Thread|CPU\\(s\\)|NUMA'
  echo
  "${mpirun_cmd[@]}"
} > "$RESULTS_DIR/${CASE_LABEL}.log" 2>&1 || status="RUN_FAIL"

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

echo "[$CASE_LABEL] binding=$MPI_BINDING_MODE hybrid_layout=$HYBRID_LAYOUT_MODE status=$status elapsed=$elapsed_s fom=$fom energy=$energy"
