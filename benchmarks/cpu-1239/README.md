# CPU Scaling Study for Versions 1, 2, 3, 9

This benchmark set records CPU-only correctness, strong scaling, and weak scaling
results for the following commits:

- `3003bb6`
- `0b4c125`
- `4d14004`
- `e3e7e98`

Per-version results are stored in each commit directory:

- `correctness.csv`
- `strong_scaling.csv`
- `weak_scaling.csv`
- `aggregate.csv`
- `raw/`

Conventions:

- CPU-only runs
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
- Each measured case keeps all raw stdout/stderr in `raw/`
