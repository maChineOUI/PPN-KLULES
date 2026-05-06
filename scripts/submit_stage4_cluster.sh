#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STAGE_LABEL="${STAGE_LABEL:-stage4}"

ACCOUNT="${STAGE4_ACCOUNT:-phadcloud}"
PARTITION="${STAGE4_PARTITION:-INTEL_8581}"
TIME_LIMIT="${STAGE4_TIME_LIMIT:-01:00:00}"
NODELIST_RAW="${STAGE4_NODELIST:-auto}"
DEFAULT_EXCLUDE_NODES="node53,node96,node124,node125,node132,node134,node135,node136,node137,node138,node145,node146,node147,node148,node149,node153,node159,node161,node162,node163,node165,node168,node169,node170,node171,node175"
EXCLUDE_NODES_RAW="${STAGE4_EXCLUDE_NODES:-$DEFAULT_EXCLUDE_NODES}"
VERSIONS="${STAGE4_VERSIONS:-both}"          # both | baseline | kc
SCALING_MODES="${STAGE4_SCALING_MODES:-weak}" # weak | strong | weak strong

BASELINE_EXEC="${STAGE4_BASELINE_EXEC:-$HOME/build-lulesh-baseline/lulesh2.0}"
KC_EXEC="${STAGE4_KC_EXEC:-$HOME/build-cluster-kc/lulesh2.0}"

# Stage 4 uses the best single-node hybrid template discovered in stage 3 and
# expands it to 1 / 8 / 27 nodes.
BASE_SIZES_STR="${STAGE4_BASE_SIZES:-120}"
ITERS="${STAGE4_ITERS:-20}"
NODE_COUNTS_STR="${STAGE4_NODE_COUNTS:-1 8 27}"
PER_NODE_CASES_STR="${STAGE4_PER_NODE_CASES:-8:12}"
MPI_BINDING_MODE_DEFAULT="${MPI_BINDING_MODE:-none}"

read -r -a BASE_SIZES <<< "$BASE_SIZES_STR"
read -r -a NODE_COUNTS <<< "$NODE_COUNTS_STR"
read -r -a PER_NODE_CASES <<< "$PER_NODE_CASES_STR"
read -r -a SCALING_MODE_LIST <<< "$SCALING_MODES"

normalize_optional_list() {
  local value=$1
  case "$value" in
    ""|auto|AUTO|none|NONE)
      echo ""
      ;;
    *)
      echo "$value"
      ;;
  esac
}

NODELIST="$(normalize_optional_list "$NODELIST_RAW")"
EXCLUDE_NODES="$(normalize_optional_list "$EXCLUDE_NODES_RAW")"

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

cube_root_scale() {
  local nodes=$1
  case "$nodes" in
    1) echo 1 ;;
    8) echo 2 ;;
    27) echo 3 ;;
    *)
      echo "unsupported node count for strong scaling: $nodes" >&2
      return 1
      ;;
  esac
}

build_results_tag() {
  local tag="${STAGE_LABEL}-cluster"
  tag+="-ver$(sanitize_token "$VERSIONS")"
  tag+="-mode$(sanitize_token "$SCALING_MODES")"
  tag+="-bases$(sanitize_token "$BASE_SIZES_STR")"
  tag+="-i${ITERS}"
  tag+="-nodes$(sanitize_token "$NODE_COUNTS_STR")"
  tag+="-pernode$(sanitize_token "$PER_NODE_CASES_STR")"
  tag+="-${TIMESTAMP}"
  echo "$tag"
}

RESULTS_TAG="$(build_results_tag)"
RESULTS_DIR="${STAGE4_RESULTS_DIR:-$ROOT_DIR/reports/${RESULTS_TAG}}"
mkdir -p "$RESULTS_DIR"

MANIFEST="$RESULTS_DIR/manifest.csv"
SUBMITTED="$RESULTS_DIR/submitted_jobs.txt"

cat > "$MANIFEST" <<'EOF'
case_id,version,section,scaling_mode,node_count,ranks_per_node,mpi_ranks,omp_threads,base_size,size,iters,exec_path
EOF
: > "$SUBMITTED"

emit_case() {
  local version=$1
  local scaling_mode=$2
  local node_count=$3
  local ranks_per_node=$4
  local threads=$5
  local base_size=$6
  local actual_size=$7
  local exec_path=$8
  local total_ranks=$((node_count * ranks_per_node))
  local case_id="${version}_stage4_${scaling_mode}_n${node_count}_np${total_ranks}_rpn${ranks_per_node}_t${threads}_s${actual_size}_b${base_size}_i${ITERS}_${TIMESTAMP}"

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$case_id" "$version" "stage4" "$scaling_mode" "$node_count" "$ranks_per_node" "$total_ranks" "$threads" "$base_size" "$actual_size" "$ITERS" "$exec_path" \
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
    --nodes="$node_count" \
    --ntasks="$total_ranks" \
    --cpus-per-task="$threads" \
    --output="$RESULTS_DIR/${case_id}.slurm.out" \
    --error="$RESULTS_DIR/${case_id}.slurm.err" \
    "${placement_args[@]}" \
    --export=ALL,CASE_LABEL="$case_id",VERSION_NAME="$version",SECTION_NAME="stage4",EXEC_PATH="$exec_path",MPI_RANKS="$total_ranks",OMP_THREADS="$threads",PROBLEM_SIZE="$actual_size",BASE_SIZE="$base_size",ITERATIONS="$ITERS",RESULTS_DIR="$RESULTS_DIR",MPI_BINDING_MODE="$binding_mode",SCALING_MODE="$scaling_mode",NODE_COUNT="$node_count",RANKS_PER_NODE="$ranks_per_node" \
    "$ROOT_DIR/scripts/run_stage4_case_cluster.slurm")

  echo "$job_id $case_id" | tee -a "$SUBMITTED"
}

echo "Stage label: $STAGE_LABEL"
echo "Cluster submission"
echo "Results dir: $RESULTS_DIR"
echo "Account/partition: $ACCOUNT / $PARTITION"
echo "Nodelist: ${NODELIST:-<auto>}"
echo "Exclude nodes: ${EXCLUDE_NODES:-<none>}"
echo "Versions: $VERSIONS"
echo "Scaling modes: $SCALING_MODES"
echo "Base sizes: $BASE_SIZES_STR"
echo "Iterations: $ITERS"
echo "Node counts: $NODE_COUNTS_STR"
echo "Per-node cases: $PER_NODE_CASES_STR"
echo "MPI binding mode: $MPI_BINDING_MODE_DEFAULT"
echo

for BASE_SIZE in "${BASE_SIZES[@]}"; do
  for per_node_case in "${PER_NODE_CASES[@]}"; do
    IFS=: read -r ranks_per_node threads <<< "$per_node_case"
    for node_count in "${NODE_COUNTS[@]}"; do
      for scaling_mode in "${SCALING_MODE_LIST[@]}"; do
        actual_size="$BASE_SIZE"
        if [[ "$scaling_mode" == "strong" ]]; then
          scale="$(cube_root_scale "$node_count")"
          if (( BASE_SIZE % scale != 0 )); then
            echo "skip strong case base_size=$BASE_SIZE nodes=$node_count: base size not divisible by cube-root scale $scale" | tee -a "$SUBMITTED"
            continue
          fi
          actual_size=$((BASE_SIZE / scale))
        fi

        if have_version baseline; then
          emit_case baseline "$scaling_mode" "$node_count" "$ranks_per_node" "$threads" "$BASE_SIZE" "$actual_size" "$BASELINE_EXEC"
        fi
        if have_version kc; then
          emit_case kc "$scaling_mode" "$node_count" "$ranks_per_node" "$threads" "$BASE_SIZE" "$actual_size" "$KC_EXEC"
        fi
      done
    done
  done
done

echo
echo "Submitted jobs list:"
echo "  $SUBMITTED"
echo "Metrics files will be written under:"
echo "  $RESULTS_DIR"
