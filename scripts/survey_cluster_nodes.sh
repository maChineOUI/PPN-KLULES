#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MODE="${SURVEY_MODE:-submit}"

ACCOUNT="${SURVEY_ACCOUNT:-phadcloud}"
PARTITION="${SURVEY_PARTITION:-INTEL_8581}"
TIME_LIMIT="${SURVEY_TIME_LIMIT:-00:10:00}"
BASELINE_EXEC="${SURVEY_BASELINE_EXEC:-$HOME/build-lulesh-baseline/lulesh2.0}"
PROBLEM_SIZE="${SURVEY_PROBLEM_SIZE:-10}"
ITERATIONS="${SURVEY_ITERATIONS:-10}"
NODE_LIST_STR="${SURVEY_NODES:-}"

RESULTS_DIR="${SURVEY_RESULTS_DIR:-$ROOT_DIR/reports/node-survey-${PARTITION}-${TIMESTAMP}}"

aggregate_results() {
  local results_dir=$1
  local summary_csv="$results_dir/all_nodes.csv"
  local csv_count=0
  shopt -s nullglob
  local csv_files=("$results_dir"/probe_*.csv)
  shopt -u nullglob

  if [[ ${#csv_files[@]} -eq 0 ]]; then
    echo "No per-node CSV files found under $results_dir" >&2
    return 1
  fi

  : > "$summary_csv"
  local first=1
  local csv_file
  for csv_file in "${csv_files[@]}"; do
    if (( first )); then
      cat "$csv_file" >> "$summary_csv"
      first=0
    else
      tail -n +2 "$csv_file" >> "$summary_csv"
    fi
    ((csv_count += 1))
  done

  echo "Aggregated $csv_count node CSV files into:"
  echo "  $summary_csv"
  echo
  echo "Quick view:"
  column -s, -t "$summary_csv" || cat "$summary_csv"
}

if [[ "$MODE" == "collect" ]]; then
  aggregate_results "$RESULTS_DIR"
  exit 0
fi

mkdir -p "$RESULTS_DIR"

SUBMITTED="$RESULTS_DIR/submitted_jobs.txt"
SUMMARY="$RESULTS_DIR/summary.csv"
: > "$SUBMITTED"

if [[ -n "$NODE_LIST_STR" ]]; then
  read -r -a NODES <<< "$NODE_LIST_STR"
else
  mapfile -t NODES < <(scontrol show hostnames "$(sinfo -h -p "$PARTITION" -o "%N" | sort -u)")
fi

if [[ ${#NODES[@]} -eq 0 ]]; then
  echo "No nodes found for partition $PARTITION" >&2
  exit 1
fi

echo "Partition: $PARTITION"
echo "Nodes to probe (${#NODES[@]}): ${NODES[*]}"
echo "Results dir: $RESULTS_DIR"
echo

for node in "${NODES[@]}"; do
  case_label="probe_${node}_${TIMESTAMP}"
  job_id=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --job-name="$case_label" \
    --time="$TIME_LIMIT" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --nodelist="$node" \
    --output="$RESULTS_DIR/${case_label}.slurm.out" \
    --error="$RESULTS_DIR/${case_label}.slurm.err" \
    --export=ALL,RESULTS_DIR="$RESULTS_DIR",CASE_LABEL="$case_label",BASELINE_EXEC="$BASELINE_EXEC",PROBLEM_SIZE="$PROBLEM_SIZE",ITERATIONS="$ITERATIONS" \
    "$ROOT_DIR/scripts/probe_cluster_node.slurm")

  echo "$job_id,$node,$case_label" | tee -a "$SUBMITTED"
done

cat > "$SUMMARY" <<EOF
node,job_id,case_label,csv_path,log_path
EOF

while IFS=, read -r job_id node case_label; do
  printf "%s,%s,%s,%s,%s\n" \
    "$node" \
    "$job_id" \
    "$case_label" \
    "$RESULTS_DIR/${case_label}.csv" \
    "$RESULTS_DIR/${case_label}.log" >> "$SUMMARY"
done < "$SUBMITTED"

echo
echo "Submitted jobs:"
echo "  $SUBMITTED"
echo "Summary manifest:"
echo "  $SUMMARY"
echo
echo "Next steps:"
echo "  1. Watch jobs: squeue -u $USER"
echo "  2. After completion, aggregate everything with:"
echo "     SURVEY_MODE=collect SURVEY_RESULTS_DIR=$RESULTS_DIR bash $ROOT_DIR/scripts/survey_cluster_nodes.sh"
