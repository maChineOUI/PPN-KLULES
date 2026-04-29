# CPU Baseline Study for Original LULESH 2.0

This benchmark set mirrors the `PPN-KLULES/benchmarks/cpu-1239` layout so the
baseline CPU results can be compared directly against versions 1, 2, 3, and 9.

Conventions:

- CPU-only runs
- `WITH_MPI=Off`
- `WITH_OPENMP=On`
- `OMP_PROC_BIND=close`
- `OMP_PLACES=cores`
- Threads: `1,2,4,8`
- Strong scaling cases:
  - `s30_i100`
  - `s45_i200`
- Weak scaling cases:
  - `t1_s30_i100`
  - `t2_s38_i100`
  - `t4_s48_i100`
  - `t8_s60_i100`

Files per result directory:

- `correctness.csv`
- `strong_scaling.csv`
- `weak_scaling.csv`
- `aggregate.csv`
- `environment.md`
- `raw/`
