#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ORIG_ROOT="${PROJECT_ROOT}/../lulesh/LULESH"

DEFAULT_NEW_BIN="${PROJECT_ROOT}/build/mpi/lulesh2.0"
DEFAULT_ORIG_BUILD_DIR="${ORIG_ROOT}/build-baseline"
DEFAULT_ORIG_BIN="${DEFAULT_ORIG_BUILD_DIR}/lulesh2.0"

NEW_BIN="${NEW_BIN:-${DEFAULT_NEW_BIN}}"
ORIG_BUILD_DIR="${ORIG_BUILD_DIR:-${DEFAULT_ORIG_BUILD_DIR}}"
ORIG_BIN="${ORIG_BIN:-${DEFAULT_ORIG_BIN}}"

NP="${NP:-8}"
INCLUDE_HYBRID=0
REBUILD_ORIG=0
MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS:-}"
ENERGY_ABS_TOL="${ENERGY_ABS_TOL:-1e-8}"
MAXREL_ABS_TOL="${MAXREL_ABS_TOL:-1e-10}"
EXTRA_CASES="${EXTRA_CASES:-}"

for arg in "$@"; do
  case "$arg" in
    --hybrid)
      INCLUDE_HYBRID=1
      ;;
    --rebuild-orig)
      REBUILD_ORIG=1
      ;;
    *)
      echo "Unknown option: $arg" >&2
      echo "Usage: bash scripts/compare-mpi-baseline.sh [--hybrid] [--rebuild-orig]" >&2
      echo "Env: NP=<ranks> MPI_EXTRA_ARGS='<extra mpirun args>'" >&2
      exit 1
      ;;
  esac
done

if [[ ! -x "${NEW_BIN}" ]]; then
  echo "New binary not found: ${NEW_BIN}" >&2
  echo "Build it first, for example:" >&2
  echo "  cmake -S . -B build/mpi -DCMAKE_BUILD_TYPE=Release -DKokkos_DIR=/usr/local/lib/cmake/Kokkos -DLULESH_ENABLE_MPI=ON -DCMAKE_CXX_COMPILER=/usr/bin/mpicxx" >&2
  echo "  cmake --build build/mpi -j8" >&2
  exit 1
fi

if [[ ! -d "${ORIG_ROOT}" ]]; then
  echo "Original LULESH source tree not found: ${ORIG_ROOT}" >&2
  exit 1
fi

build_original() {
  echo "[build] configuring original baseline in ${ORIG_BUILD_DIR}"
  cmake -S "${ORIG_ROOT}" -B "${ORIG_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_MPI=ON \
    -DWITH_OPENMP=ON \
    -DCMAKE_CXX_COMPILER=/usr/bin/mpicxx
  cmake --build "${ORIG_BUILD_DIR}" -j8
}

if [[ ! -x "${ORIG_BIN}" || "${REBUILD_ORIG}" -eq 1 ]]; then
  build_original
fi

REPORT_DIR="${PROJECT_ROOT}/reports/compare-mpi-$(date +%Y%m%d_%H%M%S)"
mkdir -p "${REPORT_DIR}"

echo "[info] reports will be stored in ${REPORT_DIR}"
echo "[info] original baseline binary: ${ORIG_BIN}"
echo "[info] refactored binary: ${NEW_BIN}"

run_case() {
  local label="$1"
  local threads="$2"
  local size="$3"
  local iters="$4"
  local orig_out="${REPORT_DIR}/${label}_orig.txt"
  local new_out="${REPORT_DIR}/${label}_new.txt"
  local -a mpi_args=()

  local -a env_vars=("OMP_NUM_THREADS=${threads}")
  if (( threads > 1 )); then
    env_vars+=("OMP_PROC_BIND=spread" "OMP_PLACES=threads")
  fi
  if [[ -n "${MPI_EXTRA_ARGS}" ]]; then
    read -r -a mpi_args <<< "${MPI_EXTRA_ARGS}"
  fi

  echo "[run] ${label} : original"
  env "${env_vars[@]}" mpirun "${mpi_args[@]}" -np "${NP}" "${ORIG_BIN}" -i "${iters}" -s "${size}" > "${orig_out}"

  echo "[run] ${label} : refactored"
  env "${env_vars[@]}" mpirun "${mpi_args[@]}" -np "${NP}" "${NEW_BIN}" -i "${iters}" -s "${size}" > "${new_out}"
}

extract_value() {
  local pattern="$1"
  local file="$2"
  awk -F= -v pat="${pattern}" '
    $0 ~ pat {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
      print $2
      exit
    }
  ' "${file}"
}

extract_scalar() {
  local pattern="$1"
  local file="$2"
  extract_value "${pattern}" "${file}" | awk '{print $1}'
}

requested_mpi_tasks_ok() {
  local reported_tasks="$1"
  awk -v reported="${reported_tasks}" -v expected="${NP}" '
    BEGIN {
      if (reported == expected) {
        print "yes"
      } else {
        print "no"
      }
    }
  '
}

metric_abs_diff() {
  local orig_val="$1"
  local new_val="$2"
  awk -v o="${orig_val}" -v n="${new_val}" '
    BEGIN {
      d = n - o
      if (d < 0) d = -d
      printf("%.6e", d)
    }
  '
}

speedup_ratio() {
  local orig_time="$1"
  local new_time="$2"
  awk -v o="${orig_time}" -v n="${new_time}" '
    BEGIN {
      if (n == 0) {
        print "inf"
      } else {
        printf("%.3f", o / n)
      }
    }
  '
}

case_status() {
  local orig_tasks="$1"
  local new_tasks="$2"
  local energy_diff="$3"
  local maxrel_diff="$4"
  awk -v ot="${orig_tasks}" -v nt="${new_tasks}" \
      -v expected="${NP}" \
      -v e="${energy_diff}" -v m="${maxrel_diff}" \
      -v etol="${ENERGY_ABS_TOL}" -v mtol="${MAXREL_ABS_TOL}" '
    BEGIN {
      if (ot != expected || nt != expected) {
        print "INVALID_MPI_BUILD"
      } else if (e <= etol && m <= mtol) {
        print "PASS"
      } else {
        print "CHECK"
      }
    }
  '
}

diagnose_case() {
  local orig_tasks="$1"
  local new_tasks="$2"
  local energy_diff="$3"
  local maxrel_diff="$4"
  awk -v ot="${orig_tasks}" -v nt="${new_tasks}" \
      -v expected="${NP}" \
      -v e="${energy_diff}" -v m="${maxrel_diff}" \
      -v etol="${ENERGY_ABS_TOL}" -v mtol="${MAXREL_ABS_TOL}" '
    BEGIN {
      if (ot != expected || nt != expected) {
        print "Reported MPI task count does not match requested NP; at least one binary is not running as an MPI build."
      } else if (e <= etol && m <= mtol) {
        print "Metrics match within tolerance."
      } else {
        print "MPI task counts look valid, but numerical metrics differ and need investigation."
      }
    }
  '
}

extract_key_metrics() {
  local file="$1"
  grep -E "Final Origin Energy|MaxRelDiff|Elapsed time|FOM" "${file}" || true
}

CASES=(
  "np${NP}_t1_s10_i10:1:10:10"
  "np${NP}_t1_s20_i20:1:20:20"
  "np${NP}_t1_s40_i20:1:40:20"
)

if (( INCLUDE_HYBRID == 1 )); then
  CASES+=("np${NP}_t4_s20_i20:4:20:20")
fi

if [[ -n "${EXTRA_CASES}" ]]; then
  IFS=';' read -r -a extra_case_array <<< "${EXTRA_CASES}"
  for extra_case in "${extra_case_array[@]}"; do
    if [[ -n "${extra_case}" ]]; then
      CASES+=("${extra_case}")
    fi
  done
fi

for case_spec in "${CASES[@]}"; do
  IFS=: read -r label threads size iters <<< "${case_spec}"
  run_case "${label}" "${threads}" "${size}" "${iters}"
done

SUMMARY_FILE="${REPORT_DIR}/summary.md"
KEY_FILE="${REPORT_DIR}/key-metrics.txt"
{
  echo "| case | status | orig MPI tasks | new MPI tasks | orig energy | new energy | energy abs diff | orig MaxRelDiff | new MaxRelDiff | MaxRelDiff abs diff | orig time (s) | new time (s) | speedup (orig/new) | orig FOM | new FOM |"
  echo "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"

  for case_spec in "${CASES[@]}"; do
    IFS=: read -r label _threads _size _iters <<< "${case_spec}"
    orig_out="${REPORT_DIR}/${label}_orig.txt"
    new_out="${REPORT_DIR}/${label}_new.txt"

    orig_tasks="$(extract_scalar "MPI tasks" "${orig_out}")"
    new_tasks="$(extract_scalar "MPI tasks" "${new_out}")"
    orig_energy="$(extract_scalar "Final Origin Energy" "${orig_out}")"
    new_energy="$(extract_scalar "Final Origin Energy" "${new_out}")"
    orig_maxrel="$(extract_scalar "MaxRelDiff" "${orig_out}")"
    new_maxrel="$(extract_scalar "MaxRelDiff" "${new_out}")"
    orig_time="$(extract_scalar "Elapsed time" "${orig_out}")"
    new_time="$(extract_scalar "Elapsed time" "${new_out}")"
    orig_fom="$(extract_scalar "FOM" "${orig_out}")"
    new_fom="$(extract_scalar "FOM" "${new_out}")"
    energy_diff="$(metric_abs_diff "${orig_energy}" "${new_energy}")"
    maxrel_diff="$(metric_abs_diff "${orig_maxrel}" "${new_maxrel}")"
    status="$(case_status "${orig_tasks}" "${new_tasks}" "${energy_diff}" "${maxrel_diff}")"

    printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" \
      "${label}" \
      "${status}" \
      "${orig_tasks}" \
      "${new_tasks}" \
      "${orig_energy}" \
      "${new_energy}" \
      "${energy_diff}" \
      "${orig_maxrel}" \
      "${new_maxrel}" \
      "${maxrel_diff}" \
      "${orig_time}" \
      "${new_time}" \
      "$(speedup_ratio "${orig_time}" "${new_time}")" \
      "${orig_fom}" \
      "${new_fom}"
  done
} | tee "${SUMMARY_FILE}"

{
  echo "Comparison report: ${REPORT_DIR}"
  echo "Original baseline binary: ${ORIG_BIN}"
  echo "Refactored binary: ${NEW_BIN}"
  echo "MPI ranks (NP): ${NP}"
  echo "MPI extra args: ${MPI_EXTRA_ARGS:-<none>}"
  echo "Energy abs tolerance: ${ENERGY_ABS_TOL}"
  echo "MaxRelDiff abs tolerance: ${MAXREL_ABS_TOL}"
  echo

  for case_spec in "${CASES[@]}"; do
    IFS=: read -r label threads size iters <<< "${case_spec}"
    orig_out="${REPORT_DIR}/${label}_orig.txt"
    new_out="${REPORT_DIR}/${label}_new.txt"

    orig_tasks="$(extract_scalar "MPI tasks" "${orig_out}")"
    new_tasks="$(extract_scalar "MPI tasks" "${new_out}")"
    orig_energy="$(extract_scalar "Final Origin Energy" "${orig_out}")"
    new_energy="$(extract_scalar "Final Origin Energy" "${new_out}")"
    orig_maxrel="$(extract_scalar "MaxRelDiff" "${orig_out}")"
    new_maxrel="$(extract_scalar "MaxRelDiff" "${new_out}")"
    orig_time="$(extract_scalar "Elapsed time" "${orig_out}")"
    new_time="$(extract_scalar "Elapsed time" "${new_out}")"
    orig_fom="$(extract_scalar "FOM" "${orig_out}")"
    new_fom="$(extract_scalar "FOM" "${new_out}")"
    energy_diff="$(metric_abs_diff "${orig_energy}" "${new_energy}")"
    maxrel_diff="$(metric_abs_diff "${orig_maxrel}" "${new_maxrel}")"
    status="$(case_status "${orig_tasks}" "${new_tasks}" "${energy_diff}" "${maxrel_diff}")"

    echo "[${label}]"
    echo "threads=${threads}, size=${size}, iterations=${iters}, status=${status}"
    echo "mpi tasks: original=${orig_tasks}, refactored=${new_tasks}, requested=${NP}"
    echo "orig: energy=${orig_energy}, maxrel=${orig_maxrel}, time=${orig_time}s, FOM=${orig_fom}"
    echo "new : energy=${new_energy}, maxrel=${new_maxrel}, time=${new_time}s, FOM=${new_fom}"
    echo "diff: energy_abs=${energy_diff}, maxrel_abs=${maxrel_diff}, speedup(orig/new)=$(speedup_ratio "${orig_time}" "${new_time}")"
    echo "diagnosis: $(diagnose_case "${orig_tasks}" "${new_tasks}" "${energy_diff}" "${maxrel_diff}")"
    echo "raw outputs:"
    echo "  ${orig_out}"
    echo "  ${new_out}"
    echo
    echo "key lines from original:"
    extract_key_metrics "${orig_out}"
    echo
    echo "key lines from refactored:"
    extract_key_metrics "${new_out}"
    echo
  done
} | tee "${KEY_FILE}"

echo
echo "[done] summary written to ${SUMMARY_FILE}"
echo "[done] key metrics written to ${KEY_FILE}"
