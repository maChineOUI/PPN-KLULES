#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <version_label> <result_dir>" >&2
  exit 1
fi

LABEL="$1"
RESULT_DIR="$2"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="${ROOT}/build/bench-cpu-baseline"
BINARY="${BUILD_DIR}/lulesh2.0"
COMMIT="$(git -C "${ROOT}" rev-parse --short HEAD)"

mkdir -p "${RESULT_DIR}/raw"

CORRECTNESS_CSV="${RESULT_DIR}/correctness.csv"
STRONG_CSV="${RESULT_DIR}/strong_scaling.csv"
WEAK_CSV="${RESULT_DIR}/weak_scaling.csv"
AGG_CSV="${RESULT_DIR}/aggregate.csv"
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

cat > "${AGG_CSV}" <<'EOF'
commit,version_label,test_type,scale_case,threads,size,iters,runs,min_elapsed_s,max_elapsed_s,mean_elapsed_s,median_elapsed_s,stddev_elapsed_s,mean_fom,median_fom,speedup,efficiency
EOF

write_environment() {
  {
    echo "# Environment"
    echo
    echo "- date: $(date --iso-8601=seconds)"
    echo "- repo branch: $(git -C "${ROOT}" branch --show-current)"
    echo "- repo head: $(git -C "${ROOT}" rev-parse HEAD)"
    echo "- compiler: $(g++ --version | head -1)"
    echo "- cmake: $(cmake --version | head -1)"
    echo "- build_dir: ${BUILD_DIR}"
    echo "- omp_proc_bind: close"
    echo "- omp_places: cores"
    echo "- threads: 1,2,4,8"
  } > "${ENV_MD}"
}

configure_build() {
  rm -rf "${BUILD_DIR}"
  cmake -S "${ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_MPI=Off \
    -DWITH_OPENMP=On >/dev/null
  cmake --build "${BUILD_DIR}" -j"$(nproc)" >/dev/null
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

run_case() {
  local threads="$1"
  local raw_file="$2"
  shift 2
  OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS="${threads}" \
    "${BINARY}" "$@" > "${raw_file}" 2>&1
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

run_correctness() {
  local raw_file="${RESULT_DIR}/raw/correctness_t1_s10_i50.txt"
  run_case 1 "${raw_file}" -s 10 -i 50
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
    run_case "${threads}" "${raw_file}" -s "${size}" -i "${iters}"
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
    run_case "${threads}" "${raw_file}" -s "${size}" -i "${iters}"
    append_scaling weak "" "${threads}" "${size}" "${iters}" "${run_id}" "${raw_file}"
  done
}

write_aggregate() {
  python3 - <<'PY' "${COMMIT}" "${LABEL}" "${STRONG_CSV}" "${WEAK_CSV}" "${AGG_CSV}"
import csv
import statistics
import sys

commit, label, strong_csv, weak_csv, out_csv = sys.argv[1:6]

def median(values):
    return statistics.median(values)

def mean(values):
    return statistics.mean(values)

def stddev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0

with open(strong_csv, newline='') as f:
    strong_rows = list(csv.DictReader(f))
with open(weak_csv, newline='') as f:
    weak_rows = list(csv.DictReader(f))

strong_groups = {}
for row in strong_rows:
    key = (row["scale_case"], row["threads"], row["size"], row["iters"])
    strong_groups.setdefault(key, []).append(row)

strong_baseline = {}
for key, rows in strong_groups.items():
    scale_case, threads, size, iters = key
    if threads == "1":
        strong_baseline[scale_case] = median([float(r["elapsed_s"]) for r in rows])

with open(out_csv, "a", newline='') as f:
    w = csv.writer(f)

    for key in sorted(strong_groups.keys(), key=lambda x: (x[0], int(x[1]))):
        scale_case, threads, size, iters = key
        rows = strong_groups[key]
        elapsed = [float(r["elapsed_s"]) for r in rows]
        fom = [float(r["fom"]) for r in rows]
        med = median(elapsed)
        baseline = strong_baseline[scale_case]
        speedup = baseline / med
        efficiency = speedup / int(threads)
        w.writerow([
            commit, label, "strong", scale_case, threads, size, iters, len(rows),
            f"{min(elapsed):.6f}", f"{max(elapsed):.6f}", f"{mean(elapsed):.6f}",
            f"{med:.6f}", f"{stddev(elapsed):.6f}", f"{mean(fom):.6f}",
            f"{median(fom):.6f}", f"{speedup:.6f}", f"{efficiency:.6f}",
        ])

    weak_groups = {}
    for row in weak_rows:
        key = (row["threads"], row["size"], row["iters"])
        weak_groups.setdefault(key, []).append(row)

    weak_baseline = None
    for key, rows in weak_groups.items():
        threads, size, iters = key
        if threads == "1":
            weak_baseline = median([float(r["elapsed_s"]) for r in rows])
            break

    for key in sorted(weak_groups.keys(), key=lambda x: int(x[0])):
        threads, size, iters = key
        rows = weak_groups[key]
        elapsed = [float(r["elapsed_s"]) for r in rows]
        fom = [float(r["fom"]) for r in rows]
        med = median(elapsed)
        speedup = weak_baseline / med if weak_baseline else 0.0
        efficiency = speedup / int(threads) if int(threads) else 0.0
        w.writerow([
            commit, label, "weak", f"t{threads}_s{size}_i{iters}", threads, size, iters, len(rows),
            f"{min(elapsed):.6f}", f"{max(elapsed):.6f}", f"{mean(elapsed):.6f}",
            f"{med:.6f}", f"{stddev(elapsed):.6f}", f"{mean(fom):.6f}",
            f"{median(fom):.6f}", f"{speedup:.6f}", f"{efficiency:.6f}",
        ])
PY
}

write_environment
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

write_aggregate
