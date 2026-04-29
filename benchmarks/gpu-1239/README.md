# GPU Single-Card Study for Version 9

This benchmark set mirrors the CPU `cpu-1239` layout for a single-GPU,
non-MPI run of `v9_kokkoscomm_gpu`.

Conventions:

- `USE_MPI=OFF`
- Single GPU only (`CUDA_VISIBLE_DEVICES=0` by default)
- Host OpenMP pinned to `OMP_NUM_THREADS=1`
- Correctness case:
  - `-s 10 -i 50`
- Strong-scaling comparison cases:
  - `s30_i100`
  - `s45_i200`
- Weak-scaling workload points reused from CPU:
  - `t1_s30_i100`
  - `t2_s38_i100`
  - `t4_s48_i100`
  - `t8_s60_i100`

Files per result directory:

- `correctness.csv`
- `strong_scaling.csv`
- `weak_scaling.csv`
- `environment.md`
- `raw/`

Run with:

```bash
./benchmarks/gpu-1239/run_gpu_bench.sh <commit> <version_label> <binary> <result_dir>
```
