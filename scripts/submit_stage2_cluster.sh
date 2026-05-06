#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STAGE_LABEL="${STAGE_LABEL:-stage2}"

ACCOUNT="${STAGE2_ACCOUNT:-phadcloud}"
PARTITION="${STAGE2_PARTITION:-INTEL_8581}"
TIME_LIMIT="${STAGE2_TIME_LIMIT:-00:20:00}"
NODELIST="${STAGE2_NODELIST:-}"
DEFAULT_EXCLUDE_NODES="node124"
EXCLUDE_NODES="${STAGE2_EXCLUDE_NODES:-$DEFAULT_EXCLUDE_NODES}"
SECTIONS="${STAGE2_SECTIONS:-all}"          # all | mpi | omp | hybrid
VERSIONS="${STAGE2_VERSIONS:-both}"         # both | baseline | kc

BASELINE_EXEC="${STAGE2_BASELINE_EXEC:-$HOME/build-lulesh-baseline/lulesh2.0}"
KC_EXEC="${STAGE2_KC_EXEC:-$HOME/build-cluster-kc/lulesh2.0}"

# Defaults below are tuned for the current single-node CPU plan on the
# currently usable nodes of the rented cluster:
#  - preferred stable node type: 120 physical cores per node
#  - topology on node60-class nodes: 2 sockets, 4 NUMA nodes, 30 cores/NUMA
#  - stages 2 and 3 are single-node only
#  - stages 2/3 keep MPI ranks to the legal LULESH cube counts 1, 8 and 27
#
# Theoretical 4-MPI NUMA layouts are not directly legal in LULESH, so the
# hybrid defaults use the closest practical single-node cases for a 120-core
# node. In particular, 8x15 matches 2 MPI ranks per NUMA node with 15 OpenMP
# threads per rank, fully covering 120 cores.
SIZES_STR="${STAGE2_SIZES:-20 40 80}"
ITERS="${STAGE2_ITERS:-20}"
MPI_RANKS_STR="${STAGE2_MPI_RANKS:-1 8 27}"
OMP_THREADS_STR="${STAGE2_OMP_THREADS:-1 15 30 60 120}"
HYBRID_CASES_STR="${STAGE2_HYBRID_CASES:-1:120 8:15 27:4}"
MPI_BINDING_MODE_DEFAULT="${MPI_BINDING_MODE:-none}"

read -r -a SIZES <<< "$SIZES_STR"
read -r -a MPI_RANKS <<< "$MPI_RANKS_STR"
read -r -a OMP_THREADS <<< "$OMP_THREADS_STR"
read -r -a HYBRID_CASES <<< "$HYBRID_CASES_STR"

have_section() {
  local name=$1
  [[ "$SECTIONS" == "all" || "$SECTIONS" == "$name" ]]
}

have_version() {
  local name=$1
  [[ "$VERSIONS" == "both" || "$VERSIONS" == "$name" ]]
}

sanitize_token() {
  local value=$1
  value="${value// /-}"
  value="${value//:/x}"
  value="${value//,/-}"
  echo "$value"
}

build_results_tag() {
  local tag="${STAGE_LABEL}-cluster"
  tag+="-sec$(sanitize_token "$SECTIONS")"
  tag+="-ver$(sanitize_token "$VERSIONS")"
  tag+="-s$(sanitize_token "$SIZES_STR")"
  tag+="-i${ITERS}"

  if have_section mpi; then
    tag+="-mpi$(sanitize_token "$MPI_RANKS_STR")"
  fi
  if have_section omp; then
    tag+="-omp$(sanitize_token "$OMP_THREADS_STR")"
  fi
  if have_section hybrid; then
    tag+="-hyb$(sanitize_token "$HYBRID_CASES_STR")"
  fi

  tag+="-${TIMESTAMP}"
  echo "$tag"
}

RESULTS_TAG="$(build_results_tag)"
RESULTS_DIR="${STAGE2_RESULTS_DIR:-$ROOT_DIR/reports/${RESULTS_TAG}}"
mkdir -p "$RESULTS_DIR"

MANIFEST="$RESULTS_DIR/manifest.csv"
SUBMITTED="$RESULTS_DIR/submitted_jobs.txt"

cat > "$MANIFEST" <<'EOF'
case_id,version,section,mpi_ranks,omp_threads,size,iters,ntasks,cpus_per_task,exec_path
EOF
: > "$SUBMITTED"

emit_case() {
  local version=$1
  local section=$2
  local ranks=$3
  local threads=$4
  local exec_path=$5
  local case_id="${version}_${section}_np${ranks}_t${threads}_s${SIZE}_i${ITERS}_${TIMESTAMP}"

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$case_id" "$version" "$section" "$ranks" "$threads" "$SIZE" "$ITERS" "$ranks" "$threads" "$exec_path" \
    >> "$MANIFEST"

  if [[ ! -x "$exec_path" ]]; then
    echo "skip $case_id: missing executable $exec_path" | tee -a "$SUBMITTED"
    return
  fi

  local binding_mode="$MPI_BINDING_MODE_DEFAULT"
  local -a placement_args=()

  if [[ -n "$NODELIST" ]]; then
    placement_args+=(--nodelist="$NODELIST")
  fi
  if [[ -n "$EXCLUDE_NODES" ]]; then
    placement_args+=(--exclude="$EXCLUDE_NODES")
  fi

  local job_id
  job_id=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --job-name="$case_id" \
    --time="$TIME_LIMIT" \
    --output="$RESULTS_DIR/${case_id}.slurm.out" \
    --error="$RESULTS_DIR/${case_id}.slurm.err" \
    --ntasks="$ranks" \
    --cpus-per-task="$threads" \
    "${placement_args[@]}" \
    --export=ALL,CASE_LABEL="$case_id",VERSION_NAME="$version",SECTION_NAME="$section",EXEC_PATH="$exec_path",MPI_RANKS="$ranks",OMP_THREADS="$threads",PROBLEM_SIZE="$SIZE",ITERATIONS="$ITERS",RESULTS_DIR="$RESULTS_DIR",MPI_BINDING_MODE="$binding_mode" \
    "$ROOT_DIR/scripts/run_stage2_case_cluster.slurm")

  echo "$job_id $case_id" | tee -a "$SUBMITTED"
}

echo "Stage label: $STAGE_LABEL"
echo "Cluster submission"
echo "Results dir: $RESULTS_DIR"
echo "Account/partition: $ACCOUNT / $PARTITION"
echo "Nodelist: ${NODELIST:-<auto>}"
echo "Exclude nodes: ${EXCLUDE_NODES:-<none>}"
echo "Sections: $SECTIONS"
echo "Versions: $VERSIONS"
echo "Sizes: $SIZES_STR"
echo "Iterations: $ITERS"
echo "MPI ranks: $MPI_RANKS_STR"
echo "OMP threads: $OMP_THREADS_STR"
echo "Hybrid cases: $HYBRID_CASES_STR"
echo "MPI binding mode: $MPI_BINDING_MODE_DEFAULT"
echo

for SIZE in "${SIZES[@]}"; do
  if have_section mpi; then
    for ranks in "${MPI_RANKS[@]}"; do
      if have_version baseline; then
        emit_case baseline mpi "$ranks" 1 "$BASELINE_EXEC"
      fi
      if have_version kc; then
        emit_case kc mpi "$ranks" 1 "$KC_EXEC"
      fi
    done
  fi

  if have_section omp; then
    for threads in "${OMP_THREADS[@]}"; do
      if have_version baseline; then
        emit_case baseline omp 1 "$threads" "$BASELINE_EXEC"
      fi
      if have_version kc; then
        emit_case kc omp 1 "$threads" "$KC_EXEC"
      fi
    done
  fi

  if have_section hybrid; then
    for hybrid_case in "${HYBRID_CASES[@]}"; do
      IFS=: read -r ranks threads <<< "$hybrid_case"
      if have_version baseline; then
        emit_case baseline hybrid "$ranks" "$threads" "$BASELINE_EXEC"
      fi
      if have_version kc; then
        emit_case kc hybrid "$ranks" "$threads" "$KC_EXEC"
      fi
    done
  fi
done

echo
echo "Submitted jobs list:"
echo "  $SUBMITTED"
echo "Metrics files will be written under:"
echo "  $RESULTS_DIR"
