# CPU Phase 2 Summary

Date: 2026-04-30  
Cluster: Zen  
Current deployed/tested commit on cluster: `e3e7e98` (`lulesh-KC`)

## Scope

This summary consolidates the CPU results obtained so far for:

- baseline LULESH
- Kokkos migrated LULESH

It separates:

- exploratory runs performed manually on the login node
- representative validation runs performed on `compute` nodes through Slurm

All reported cases below completed successfully and produced numerically consistent results.

## Numeric correctness

Across all validated cases:

- `Final Origin Energy` matches between baseline and Kokkos
- `MaxAbsDiff` remains very small
- no compute-node smoke test produced stderr output

## Exploratory results on login node

### Pure MPI

| Version | Config | Energy | MaxAbsDiff | Elapsed (s) | FOM (z/s) |
|---|---|---:|---:|---:|---:|
| baseline | `1x1`, `s=10`, `i=10` | `2.596764e+05` | `4.092726e-12` | `0.0051` | `1976.6795` |
| kokkos | `1x1`, `s=10`, `i=10` | `2.596764e+05` | `4.092726e-12` | `0.01` | `1980.1553` |
| baseline | `8x1`, `s=10`, `i=10` | `2.077411e+06` | `5.093170e-11` | `0.0061` | `13113.61` |
| kokkos | `8x1`, `s=10`, `i=10` | `2.077411e+06` | `5.093170e-11` | `0.01` | `13194.784` |
| baseline | `27x1`, `s=10`, `i=10` | `7.011263e+06` | `4.365575e-11` | `0.0084` | `32269.746` |
| kokkos | `27x1`, `s=10`, `i=10` | `7.011263e+06` | `4.365575e-11` | `0.01` | `32430.325` |

### Pure OpenMP

| Version | Config | Energy | MaxAbsDiff | Elapsed (s) | FOM (z/s) |
|---|---|---:|---:|---:|---:|
| baseline | `1x8`, `s=10`, `i=10` | `2.596764e+05` | `4.547474e-13` | `0.093` | `107.05497` |
| kokkos | `1x8`, `s=10`, `i=10` | `2.596764e+05` | `4.092726e-12` | `0.02` | `472.22788` |
| baseline | `1x32`, `s=10`, `i=10` | `2.596764e+05` | `4.547474e-13` | `0.47` | `21.268872` |
| kokkos | `1x32`, `s=10`, `i=10` | `2.596764e+05` | `4.092726e-12` | `0.09` | `109.01424` |

### Hybrid MPI + OpenMP

| Version | Config | Energy | MaxAbsDiff | Elapsed (s) | FOM (z/s) |
|---|---|---:|---:|---:|---:|
| baseline | `8x8`, `s=10`, `i=10` | `2.077411e+06` | `5.093170e-11` | `0.095` | `845.78521` |
| kokkos | `8x8`, `s=10`, `i=10` | `2.077411e+06` | `5.093170e-11` | `0.02` | `3592.8605` |

## Representative validation on compute nodes

These runs were submitted through Slurm on the `compute` partition and confirm that the chosen Kokkos version also runs correctly on compute nodes.

| Mode | Config | Slurm job | Energy | MaxAbsDiff | Elapsed (s) | FOM (z/s) | stderr |
|---|---|---:|---:|---:|---:|---:|---|
| pure MPI | `8x1`, `s=10`, `i=10` | `1671605` | `2.077411e+06` | `5.093170e-11` | `0.01` | `13698.63` | empty |
| pure OpenMP | `1x8`, `s=10`, `i=10` | `1671606` | `2.596764e+05` | `4.092726e-12` | `0.04` | `237.05051` | empty |
| hybrid | `8x8`, `s=10`, `i=10` | `1671607` | `2.077411e+06` | `5.093170e-11` | `0.04` | `1869.1457` | empty |

## First formal compute-node dataset (`s=20`, `i=20`)

These runs extend the initial representative validation and provide a first small formal CPU dataset on compute nodes. At this stage, the comparison is complete for `MPI 8x1`, `OMP 1x8`, and `OMP 1x32`; the hybrid line currently includes only the Kokkos result, and `MPI 27x1` remains blocked by a launcher/binding issue.

| Mode | Config | Version | Slurm job | Energy | MaxAbsDiff | Elapsed (s) | FOM (z/s) | stderr |
|---|---|---|---:|---:|---:|---:|---:|---|
| pure MPI | `8x1`, `s=20`, `i=20` | baseline | `1671626` | `1.038309e+07` | `9.313226e-10` | `0.088` | `14475.077` | empty |
| pure MPI | `8x1`, `s=20`, `i=20` | kokkos | `1671621` | `1.038309e+07` | `1.164153e-10` | `0.09` | `13814.242` | empty |
| pure MPI | `27x1`, `s=20`, `i=20` | kokkos | `1671624` | — | — | — | — | blocked by MPI launcher / binding issue |
| pure OpenMP | `1x8`, `s=20`, `i=20` | baseline | `1671627` | `1.297886e+06` | `8.731149e-11` | `0.49` | `327.54358` | empty |
| pure OpenMP | `1x8`, `s=20`, `i=20` | kokkos | `1671622` | `1.297886e+06` | `1.309672e-10` | `0.17` | `914.69961` | empty |
| pure OpenMP | `1x32`, `s=20`, `i=20` | baseline | `1671628` | `1.297886e+06` | `8.731149e-11` | `1.8` | `86.946949` | empty |
| pure OpenMP | `1x32`, `s=20`, `i=20` | kokkos | `1671625` | `1.297886e+06` | `1.309672e-10` | `0.43` | `373.31783` | empty |
| hybrid | `8x8`, `s=20`, `i=20` | baseline | `1671629` | `1.038309e+07` | `1.164153e-10` | `0.5` | `2566.0465` | empty |
| hybrid | `8x8`, `s=20`, `i=20` | kokkos | `1671623` | `1.038309e+07` | `9.313226e-10` | `0.18` | `7198.1667` | empty |

## Current interpretation

- The current cluster deployment (`e3e7e98`) is usable for CPU testing.
- Pure MPI is numerically correct on `1`, `8`, and `27` ranks in manual exploratory runs.
- Representative pure MPI, pure OpenMP, and hybrid cases all run correctly on compute nodes under Slurm.
- A first formal compute-node CPU dataset at `s=20`, `i=20` has now been obtained for representative MPI, OpenMP, and hybrid configurations.
- At `MPI 8x1`, baseline and Kokkos are numerically consistent and show very similar performance.
- At `OMP 1x8` and `OMP 1x32`, baseline and Kokkos are numerically consistent, with Kokkos showing more favorable timings on the tested cases.
- At `hybrid 8x8`, baseline and Kokkos are numerically consistent, with Kokkos showing more favorable timings on the tested case.
- At this stage, the only missing formal Kokkos point in the planned small matrix is `MPI 27x1` at `s=20`, `i=20`, currently blocked by an MPI launcher / binding issue rather than by a numerical failure.
- On the small exploratory OpenMP and hybrid cases tested so far, the Kokkos version shows more favorable timings than baseline.

## Automated OpenMP dataset on compute nodes

The OpenMP line was automated separately on compute nodes, avoiding the MPI launcher sensitivity observed in the multi-rank campaign.

### OpenMP results (`s=20`, `i=20`)

| Version | Threads | Energy | MaxAbsDiff | Elapsed (s) | FOM (z/s) |
|---|---:|---:|---:|---:|---:|
| baseline | `1` | `1.297886e+06` | `1.309672e-10` | `0.084` | `1914.5448` |
| kokkos | `1` | `1.297886e+06` | `1.309672e-10` | `0.10` | `1555.3392` |
| baseline | `8` | `1.297886e+06` | `8.731149e-11` | `0.5` | `322.86629` |
| kokkos | `8` | `1.297886e+06` | `1.309672e-10` | `0.18` | `906.1963` |
| baseline | `32` | `1.297886e+06` | `8.731149e-11` | `1.9` | `86.049419` |
| kokkos | `32` | `1.297886e+06` | `1.309672e-10` | `0.45` | `359.09367` |

### OpenMP results (`s=40`, `i=20`)

| Version | Threads | Energy | MaxAbsDiff | Elapsed (s) | FOM (z/s) |
|---|---:|---:|---:|---:|---:|
| baseline | `1` | `1.038309e+07` | `9.313226e-10` | `0.64` | `1990.7655` |
| kokkos | `1` | `1.038309e+07` | `9.313226e-10` | `0.80` | `1597.8858` |
| baseline | `8` | `1.038309e+07` | `1.164153e-10` | `1.1` | `1157.1218` |
| kokkos | `8` | `1.038309e+07` | `9.313226e-10` | `0.85` | `1500.3247` |
| baseline | `32` | `1.038309e+07` | `1.164153e-10` | `2.5` | `516.65463` |
| kokkos | `32` | `1.038309e+07` | `9.313226e-10` | `1.17` | `1097.8698` |

### OpenMP results (`s=80`, `i=20`)

| Version | Threads | Energy | MaxAbsDiff | Elapsed (s) | FOM (z/s) |
|---|---:|---:|---:|---:|---:|
| baseline | `1` | `8.306471e+07` | `2.793968e-09` | `4.9` | `2075.7873` |
| kokkos | `1` | `8.306471e+07` | `2.793968e-09` | `7.25` | `1412.3489` |
| baseline | `8` | `8.306471e+07` | `2.793968e-09` | `5.6` | `1833.0694` |
| kokkos | `8` | `8.306471e+07` | `2.793968e-09` | `6.29` | `1627.3478` |
| baseline | `32` | `8.306471e+07` | `2.793968e-09` | `7.0` | `1462.504` |
| kokkos | `32` | `8.306471e+07` | `2.793968e-09` | `6.63` | `1543.5954` |

## OpenMP interpretation

- The OpenMP automation path is stable on compute nodes.
- At both `s=20` and `s=40`, baseline and Kokkos remain numerically consistent for `1`, `8`, and `32` threads.
- At `1` thread, baseline is slightly faster than Kokkos.
- At `8` and `32` threads, Kokkos is clearly faster than baseline on the tested cases.
- For both codes, performance decreases when going from `8` to `32` threads on these cases, suggesting that higher thread counts are not beneficial for these problem sizes.
- At `s=80`, baseline and Kokkos also remain numerically consistent for `1`, `8`, and `32` threads.
- At `s=80`, baseline is faster at `1` and `8` threads, while Kokkos becomes slightly faster at `32` threads.
- Overall, the OpenMP comparison is therefore scale-dependent: Kokkos is clearly favorable at `s=20` and `s=40` for higher thread counts, while at `s=80` the advantage becomes smaller and only remains visible at `32` threads.

## Automated MPI dataset on compute nodes

The MPI line was partially automated on compute nodes for the stable `1x1` and `8x1` configurations. The automation path is usable for `1x1` across all tested sizes and for baseline `8x1`, but some Kokkos `8x1` cases remain sensitive to MPI launcher / binding behavior. For those cases, valid compute-node results were recovered with manual single-case Slurm runs.

### MPI results (`1x1`)

| Version | Size | Energy | MaxAbsDiff | Elapsed (s) | FOM (z/s) |
|---|---:|---:|---:|---:|---:|
| baseline | `20` | `1.297886e+06` | `1.309672e-10` | `0.083` | `1933.0468` |
| kokkos | `20` | `1.297886e+06` | `8.731149e-11` | `0.09` | `1873.4266` |
| baseline | `40` | `1.038309e+07` | `9.313226e-10` | `0.64` | `2004.428` |
| kokkos | `40` | `1.038309e+07` | `1.164153e-10` | `0.63` | `2032.2428` |
| baseline | `80` | `8.306471e+07` | `2.793968e-09` | `4.9` | `2101.7155` |
| kokkos | `80` | `8.306471e+07` | `2.793968e-09` | `5.36` | `1910.9583` |

### MPI results (`8x1`)

| Version | Size | Source | Energy | MaxAbsDiff | Elapsed (s) | FOM (z/s) |
|---|---:|---|---:|---:|---:|---:|
| baseline | `20` | automated | `1.038309e+07` | `9.313226e-10` | `0.089` | `14404.522` |
| kokkos | `20` | manual retry | `1.038309e+07` | `1.164153e-10` | `0.09` | `13875.339` |
| baseline | `40` | automated | `8.306471e+07` | `2.793968e-09` | `0.8` | `12861.832` |
| kokkos | `40` | automated | `8.306471e+07` | `2.793968e-09` | `0.65` | `15795.936` |
| baseline | `80` | automated | `6.645177e+08` | `1.490116e-08` | `6.6` | `12449.323` |
| kokkos | `80` | manual retry | `6.645177e+08` | `7.450581e-09` | `6.71` | `12205.522` |

## MPI interpretation

- The MPI automation path is partially usable on compute nodes for `1x1` and baseline `8x1`.
- For Kokkos `8x1`, some automated runs remain sensitive to MPI launcher / binding behavior; however, the missing `s=20` and `s=80` points were successfully recovered with manual single-case Slurm runs.
- Across the collected `1x1` and `8x1` datasets, baseline and Kokkos remain numerically consistent.
- At `1x1`, baseline and Kokkos show very similar performance, with no strong systematic advantage across the tested sizes.
- At `8x1`, Kokkos is very close to baseline at `s=20`, clearly faster at `s=40`, and slightly slower at `s=80` on the recovered manual result.
- The main unresolved MPI point remains `27x1`, which is still blocked by launcher / binding instability rather than by a numerical issue.

## Suggested next CPU campaign

Recommended next formal CPU matrix on compute nodes:

- pure MPI: `1x1`, `8x1`, `27x1`
- pure OpenMP: `1x1`, `1x8`, `1x32`
- hybrid: `8x8`, optionally `27x2`

Use initially:

- `s=20`
- `i=20`

and only increase problem size after confirming runtime and cost remain reasonable.
