# MPI Baseline vs Kokkos Refactor Comparison

Date: 2026-04-09

## Scope

This report compares the original MPI-enabled LULESH baseline against the refactored Kokkos version with MPI communication reintroduced.

Comparison goals:

1. Validate numerical correctness after reintroducing MPI communication.
2. Measure runtime and FOM differences on the local machine.

## Executables

- Original baseline: `/tmp/lulesh-orig-mpi/lulesh2.0`
- Refactored Kokkos version: `/home/fzj/桌面/ppn/PPN-KLULES/build/mpi/lulesh2.0`

## Reproducible Commands

Pure MPI and hybrid `np=8` suite:

```bash
cd /home/fzj/桌面/ppn/PPN-KLULES
NP=8 bash scripts/compare-mpi-baseline.sh --hybrid
```

Pure MPI `np=27` suite:

```bash
cd /home/fzj/桌面/ppn/PPN-KLULES
MPI_EXTRA_ARGS="--oversubscribe" NP=27 bash scripts/compare-mpi-baseline.sh
```

The helper script `scripts/compare-mpi-baseline.sh` was updated to accept:

- `NP=<ranks>`
- `MPI_EXTRA_ARGS='<extra mpirun args>'`

This makes it possible to run `np=27` on a machine where Open MPI exposes fewer slots than requested.

## Raw Result Directories

- `np=8`: `/home/fzj/桌面/ppn/PPN-KLULES/reports/compare-mpi-20260409_164119`
- `np=27`: `/home/fzj/桌面/ppn/PPN-KLULES/reports/compare-mpi-20260409_164550`

## Correctness Summary

Across all benchmark cases in this report:

- `Final Origin Energy` matches exactly to printed precision between the original and refactored versions.
- `MaxRelDiff` remains at the same level as the baseline.
- The `np=27` cases confirm correctness not only for boundary ranks but also for internal-rank communication paths.

Representative examples:

| case | orig energy | new energy | orig MaxRelDiff | new MaxRelDiff |
|---|---:|---:|---:|---:|
| `np8_t1_s20_i20` | `1.038309e+07` | `1.038309e+07` | `9.285741e-13` | `9.285741e-13` |
| `np8_t1_s40_i20` | `8.306471e+07` | `8.306471e+07` | `6.088817e-13` | `6.088817e-13` |
| `np27_t1_s20_i20` | `3.504293e+07` | `3.504293e+07` | `9.250613e-13` | `9.250613e-13` |
| `np27_t1_s40_i20` | `2.803434e+08` | `2.803434e+08` | `7.065527e-13` | `7.065527e-13` |

Conclusion on correctness:

- The reintroduced MPI communication in the Kokkos refactor is numerically consistent with the original LULESH baseline for both `np=8` and `np=27`.

## Performance Results

### Pure MPI, `np=8`, `OMP_NUM_THREADS=1`

| case | orig time (s) | new time (s) | time ratio `(orig/new)` | orig FOM | new FOM |
|---|---:|---:|---:|---:|---:|
| `np8_t1_s10_i10` | `0.0086` | `0.01` | `0.860` | `9310.948` | `5445.5168` |
| `np8_t1_s20_i20` | `0.14` | `0.13` | `1.077` | `9430.0087` | `10093.401` |
| `np8_t1_s40_i20` | `1.1` | `1.02` | `1.078` | `9646.4737` | `10066.827` |

Interpretation:

- At the smallest case (`s=10`), the original baseline is slightly faster.
- Once the local work size increases (`s=20`, `s=40`), the Kokkos refactor becomes slightly faster, by about `1.08x`.

### Pure MPI, `np=27`, `OMP_NUM_THREADS=1`

| case | orig time (s) | new time (s) | time ratio `(orig/new)` | orig FOM | new FOM |
|---|---:|---:|---:|---:|---:|
| `np27_t1_s10_i10` | `0.021` | `0.03` | `0.700` | `12691.385` | `10057.574` |
| `np27_t1_s20_i20` | `0.36` | `0.23` | `1.565` | `11974.942` | `18870.13` |
| `np27_t1_s40_i20` | `3.2` | `2.13` | `1.502` | `10864.687` | `16249.34` |

Interpretation:

- At very small size (`s=10`), the original baseline still has lower overhead.
- For more meaningful cases (`s=20`, `s=40`), the Kokkos refactor is clearly faster, by about `1.50x` to `1.57x`.
- Because `np=27` covers internal ranks, these results are especially important for validating both correctness and communication efficiency behavior.

### Hybrid Case, `np=8`, `OMP_NUM_THREADS=4`

| case | orig time (s) | new time (s) | time ratio `(orig/new)` | orig FOM | new FOM |
|---|---:|---:|---:|---:|---:|
| `np8_t4_s20_i20` | `1.1e+02` | `39.99` | `2.751` | `11.771846` | `32.00753` |

Interpretation:

- On this machine, the Kokkos refactor is much faster than the original baseline in this hybrid configuration.
- However, this case is strongly affected by rank/thread binding and local scheduling.
- It should be treated as a secondary result, not the primary headline number.

## Main Conclusions

1. Numerical correctness is validated.
   The Kokkos version with restored MPI communication matches the original LULESH baseline in all tested cases, including `np=27` internal-rank communication.

2. Pure MPI performance is competitive and often better.
   For non-trivial problem sizes, the refactored Kokkos version outperforms the original baseline:
   - around `1.08x` at `np=8`
   - around `1.50x` to `1.57x` at `np=27`

3. Small cases are dominated by overhead.
   For `s=10`, the original baseline remains faster. This is expected and does not indicate a communication bug.

4. Hybrid results should be interpreted carefully.
   The current hybrid numbers are valid as measured results on the local machine, but they are sensitive to placement and thread binding. They should not be used as the only efficiency claim.

## Recommended Citation in a Progress Report

Suggested short conclusion:

> The refactored Kokkos LULESH with restored MPI communication reproduces the original LULESH results exactly to printed precision. In pure-MPI tests, it matches or exceeds the original baseline performance for moderate and large cases, reaching about `1.5x` speedup at `np=27` for the larger benchmark sizes on the local test platform.
