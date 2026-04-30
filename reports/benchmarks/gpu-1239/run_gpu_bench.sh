#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <commit> <version_label> <binary> <result_dir>" >&2
  exit 1
fi

COMMIT="$1"
LABEL="$2"
BINARY="$3"
RESULT_DIR="$4"

mkdir -p "${RESULT_DIR}/raw"

CORRECTNESS_CSV="${RESULT_DIR}/correctness.csv"
STRONG_CSV="${RESULT_DIR}/strong_scaling.csv"
WEAK_CSV="${RESULT_DIR}/weak_scaling.csv"
ENV_MD="${RESULT_DIR}/environment.md"

cat > "${CORRECTNESS_CSV}" <<'EOF'
commit,version_label,threads,size,iters,elapsed_s,grind_time_us_z_c,fom,final_energy,max_abs_diff,total_abs_diff,max_rel_diff,notes
EOF

cat > "${STRONG_CSV}" <<'EOF'
commit,version_label,scale_case,threads,size,iters,run_id,elapsed_s,grind_time_us_z_c,fom,final_energy,max_abs_diff,total_abs_diff,max_rel_diff,raw_file
EOF

cat > "${WEAK_CSV}" <<'EOF'
commit,version_label,threads,size,iters,run_id,elapsed_s,grind_time_us_z_c,fom,final_energy,max_abs_diff,total_abs_diff,max_rel_diff,raw_file
EOF

write_environment() {
  {
    echo "# Environment"
    echo
    echo "- date: $(date --iso-8601=seconds)"
    echo "- repo branch: $(git -C "$(dirname "$0")/../.." branch --show-current 2>/dev/null || echo unknown)"
    echo "- repo head: $(git -C "$(dirname "$0")/../.." rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "- benchmark commit label: ${COMMIT}"
    echo "- version label: ${LABEL}"
    echo "- binary: ${BINARY}"
    echo "- use_mpi: OFF"
    echo "- cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-0}"
    echo "- omp_num_threads: ${OMP_NUM_THREADS:-1}"
    echo "- omp_proc_bind: ${OMP_PROC_BIND:-close}"
    echo "- omp_places: ${OMP_PLACES:-cores}"
    echo "- nvcc: $(nvcc --version | grep 'release' | awk '{print $6}' || echo unavailable)"
    echo "- nvidia_smi: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo unavailable)"
  } > "${ENV_MD}"
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
    "${mad}" "${tad}" "${mrd}" "$(basename "${raw_file}")" >> "${CORRECTNESS_CSV}"
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
  if [[ "${kind}" == "strong" ]]; then
    csv="${STRONG_CSV}"
  else
    csv="${WEAK_CSV}"
  fi

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

run_case() {
  local raw_file="$1"
  shift
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
  OMP_PROC_BIND="${OMP_PROC_BIND:-close}" \
  OMP_PLACES="${OMP_PLACES:-cores}" \
    "${BINARY}" "$@" > "${raw_file}" 2>&1
}

run_correctness() {
  local raw_file="${RESULT_DIR}/raw/correctness_t1_s10_i50.txt"
  run_case "${raw_file}" -s 10 -i 50
  append_correctness "${raw_file}"
}

run_strong_case() {
  local size="$1"
  local iters="$2"
  local scale_case="s${size}_i${iters}"
  local run_id raw_file
  for run_id in 1 2 3 4 5; do
    raw_file="${RESULT_DIR}/raw/strong_${scale_case}_t1_run${run_id}.txt"
    run_case "${raw_file}" -s "${size}" -i "${iters}"
    append_scaling strong "${scale_case}" 1 "${size}" "${iters}" "${run_id}" "${raw_file}"
  done
}

run_weak_case() {
  local size="$1"
  local iters=100
  local run_id raw_file
  for run_id in 1 2 3 4 5; do
    raw_file="${RESULT_DIR}/raw/weak_t1_s${size}_i${iters}_run${run_id}.txt"
    run_case "${raw_file}" -s "${size}" -i "${iters}"
    append_scaling weak "" 1 "${size}" "${iters}" "${run_id}" "${raw_file}"
  done
}

write_environment

run_correctness
run_strong_case 30 100
run_strong_case 45 200
run_weak_case 30
run_weak_case 38
run_weak_case 48
run_weak_case 60
