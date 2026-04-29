#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <commit> <version_label> <result_dir>" >&2
  exit 1
fi

COMMIT="$1"
LABEL="$2"
RESULT_DIR="$3"
ROOT="/home/h/ppn/PPN-KLULES"
WT="/tmp/cpu1239-${COMMIT}-$$"
KOKKOS_PREFIX="/tmp/kokkos-5.0.2-install"

mkdir -p "${RESULT_DIR}/raw"

git -C "${ROOT}" worktree add --detach "${WT}" "${COMMIT}" >/dev/null

cleanup() {
  git -C "${ROOT}" worktree remove --force "${WT}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

configure_build() {
  rm -rf "${WT}/build/release"
  case "${COMMIT}" in
    3003bb6)
      cmake -S "${WT}" -B "${WT}/build/release" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS='-include cstdio' >/dev/null
      ;;
    0b4c125|4d14004)
      cmake -S "${WT}" -B "${WT}/build/release" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="${KOKKOS_PREFIX}" >/dev/null
      ;;
    e3e7e98)
      cmake -S "${WT}" -B "${WT}/build/release" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="${KOKKOS_PREFIX}" \
        -DUSE_MPI=OFF >/dev/null
      ;;
    *)
      echo "unsupported commit: ${COMMIT}" >&2
      exit 1
      ;;
  esac

  cmake --build "${WT}/build/release" -j4 >/dev/null
}

extract_field() {
  local pattern="$1"
  local file="$2"
  grep -F "${pattern}" "${file}" | head -n1 | awk -F'= ' '{print $2}' | xargs
}

extract_first_number() {
  local pattern="$1"
  local file="$2"
  extract_field "${pattern}" "${file}" | awk '{print $1}'
}

append_correctness() {
  local raw_file="$1"
  local csv="${RESULT_DIR}/correctness.csv"
  local elapsed grind fom energy mad tad mrd
  elapsed="$(extract_first_number "Elapsed time" "${raw_file}")"
  grind="$(extract_first_number "Grind time (us/z/c)" "${raw_file}")"
  fom="$(extract_first_number "FOM" "${raw_file}")"
  energy="$(extract_field "Final Origin Energy" "${raw_file}")"
  mad="$(extract_field "MaxAbsDiff" "${raw_file}")"
  tad="$(extract_field "TotalAbsDiff" "${raw_file}")"
  mrd="$(extract_field "MaxRelDiff" "${raw_file}")"
  printf "%s,%s,1,10,50,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${COMMIT}" "${LABEL}" "${elapsed}" "${grind}" "${fom}" "${energy}" \
    "${mad}" "${tad}" "${mrd}" "$(basename "${raw_file}")" >> "${csv}"
}

append_scaling() {
  local kind="$1"
  local scale_case="$2"
  local threads="$3"
  local size="$4"
  local iters="$5"
  local run_id="$6"
  local raw_file="$7"
  local csv elapsed grind fom energy mad tad mrd
  csv="${RESULT_DIR}/${kind}_scaling.csv"
  elapsed="$(extract_first_number "Elapsed time" "${raw_file}")"
  grind="$(extract_first_number "Grind time (us/z/c)" "${raw_file}")"
  fom="$(extract_first_number "FOM" "${raw_file}")"
  energy="$(extract_field "Final Origin Energy" "${raw_file}")"
  mad="$(extract_field "MaxAbsDiff" "${raw_file}")"
  tad="$(extract_field "TotalAbsDiff" "${raw_file}")"
  mrd="$(extract_field "MaxRelDiff" "${raw_file}")"

  if [[ "${kind}" == "strong" ]]; then
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
      "${COMMIT}" "${LABEL}" "${scale_case}" "${threads}" "${size}" "${iters}" \
      "${run_id}" "${elapsed}" "${grind}" "${fom}" "${energy}" "${mad}" \
      "${tad}" "${mrd}" "$(basename "${raw_file}")" >> "${csv}"
  else
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
      "${COMMIT}" "${LABEL}" "${threads}" "${size}" "${iters}" "${run_id}" \
      "${elapsed}" "${grind}" "${fom}" "${energy}" "${mad}" "${tad}" \
      "${mrd}" "$(basename "${raw_file}")" >> "${csv}"
  fi
}

run_correctness() {
  local raw_file="${RESULT_DIR}/raw/correctness_t1_s10_i50.txt"
  OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=1 \
    "${WT}/build/release/lulesh2.0" -s 10 -i 50 > "${raw_file}" 2>&1
  append_correctness "${raw_file}"
}

run_strong_case() {
  local size="$1"
  local iters="$2"
  local threads="$3"
  local scale_case="s${size}_i${iters}"
  local run_id raw_file
  for run_id in 1 2 3 4 5; do
    raw_file="${RESULT_DIR}/raw/strong_${scale_case}_t${threads}_run${run_id}.txt"
    OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS="${threads}" \
      "${WT}/build/release/lulesh2.0" -s "${size}" -i "${iters}" > "${raw_file}" 2>&1
    append_scaling strong "${scale_case}" "${threads}" "${size}" "${iters}" "${run_id}" "${raw_file}"
  done
}

run_weak_case() {
  local threads="$1"
  local size="$2"
  local iters=100
  local run_id raw_file
  for run_id in 1 2 3 4 5; do
    raw_file="${RESULT_DIR}/raw/weak_t${threads}_s${size}_i${iters}_run${run_id}.txt"
    OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS="${threads}" \
      "${WT}/build/release/lulesh2.0" -s "${size}" -i "${iters}" > "${raw_file}" 2>&1
    append_scaling weak "" "${threads}" "${size}" "${iters}" "${run_id}" "${raw_file}"
  done
}

configure_build
run_correctness

for t in 1 2 4 8; do
  run_strong_case 30 100 "${t}"
done

for t in 1 2 4 8; do
  run_strong_case 45 200 "${t}"
done

run_weak_case 1 30
run_weak_case 2 38
run_weak_case 4 48
run_weak_case 8 60
