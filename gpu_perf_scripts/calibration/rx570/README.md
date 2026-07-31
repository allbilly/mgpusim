# RX 570 (gfx803 / Polaris 20) timing calibration

This directory contains the reproducible RX 570 hardware and simulator
calibration. All ten workloads use the same optimized gfx803 HIP code objects
on hardware and in MGPUSim.

## Measurement basis

The host cannot pin `power_dpm_force_performance_level=high`, so the recorded
HIP-event microseconds remain diagnostic. The hardware file records both:

- `cold`: the first timed launch, including workload-dependent ROCm/KFD
  first-use costs;
- `steady`: the median of three independent processes, each averaging 100
  launches after 10 warmups. This is the comparison used for the GPU model.

Cold-minus-steady ranges from about 10 to 71 µs and is not represented by a
single simulator constant. `compare.py` therefore uses steady timing by
default and exposes cold timing only as an explicit diagnostic.

The harness still does not collect `s_memtime` cycle-counter deltas. A pinned
clock or a cycle-counter probe is required before treating the absolute
microseconds as reference-quality.

## Rebuild exact code objects

```bash
./regen_hsaco.sh
```

This compiles every native source with `-O3 --offload-arch=gfx803`, installs
the result as `kernels_gfx803.hsaco` in its benchmark package, and records
compiler identity and hashes in `hsaco_manifest.txt`.

## Hardware measurement

```bash
./build_and_run.sh
./build_and_run.sh --warmup 20 --iters 100
```

The script generates deterministic k-means and pagerank fixtures, builds the
exact-HSACO harness in the ROCm container, checks clock policy, and runs it on
the RX 570.

## Simulator measurement

```bash
SIM_JOBS=4 ./run_sim.sh sim_out | tee sim_results.txt
./compare.py sim_results.txt
./compare.py --hw-mode cold sim_results.txt
```

`SIM_JOBS` controls independent simulator processes, not simulated GPU
parallelism. All runs use:

```text
-timing -arch gcn3 -gpu rx570 -verify
```

The `gcn3` name is intentional: this repository classifies gfx803 as
`arch.GCN3`, even though Polaris is marketed as GCN 4.

## Cross-size holdouts

Use matched size sweeps to check scaling separately from the default-size
calibration points:

```bash
./run_size_sweeps.py --mode sim --jobs 4 --output sim_size_sweep.csv
./run_size_sweeps.py --mode hardware --jobs 1 --trials 3 \
  --output hw_size_sweep.csv
./compare_size_sweeps.py hw_size_sweep.csv sim_size_sweep.csv
```

The sweep uses four launch-compatible sizes for each of eight scalable
families plus a ReLU midpoint at 131K. It writes every point and its status
even if an individual run fails, so a failed holdout cannot silently
disappear. Both committed CSV files contain all 33 matched points. Hardware
mode checks `/dev/kfd` access, runs three independent processes per point by
default, and records their median.

The sizes and work transforms are fixed in `run_size_sweeps.py`. Do not tune
the timing model from simulator-only curves. Collect the corresponding exact
gfx803 hardware rows first, then judge both point error and the reported
hardware/simulator slope ratio.

## Current steady-state result

| Benchmark | HW steady (µs) | Sim (µs) | Error |
|---|---:|---:|---:|
| vectoradd | 9.299 | 7.340 | 21.1% |
| relu | 6.997 | 6.375 | 8.9% |
| matrixmult | 52.396 | 49.350 | 5.8% |
| matrixtranspose | 22.579 | 25.402 | 12.5% |
| bitonicsort | 348.697 | 321.321 | 7.9% |
| aes | 18.595 | 16.892 | 9.2% |
| fir | 7.875 | 7.566 | 3.9% |
| kmeans | 30.045 | 27.345 | 9.0% |
| pagerank | 19.176 | 22.334 | 16.5% |
| nw | 146.999 | 139.076 | 5.4% |

Canonical-size MARE is **10.0%** across all ten remeasured benchmarks. The
anti-overfit result is **7.0% MARE across 33 matched size points**; all eight
swept families have family MARE below 10%. The maximum point error is the
65K vectoradd holdout at 21.1%. All points verify.

## Main model corrections

- Exact gfx803 code objects are embedded for all ten benchmarks.
- Missing gfx803 scalar operations and `v_mad_u16` are implemented.
- L2 bank service is restored to 16 requests/cycle; the earlier low cap
  created a superlinear large-vector queue bottleneck.
- GDDR5 exposes four interleaved banks per 32-bit controller so unrelated
  random misses can overlap; cache and store serialization costs are modeled
  separately.
- A shared L2-to-DRAM token bucket preserves a 625 KiB short/random burst and
  limits sustained cache-line issue; the large vectoradd point remains within
  9% across the repeated hardware median.
- A write-filtered L1-to-L2 token bucket preserves the first 192 KiB of dense
  stores, then limits sustained write ingress to three lines per five cycles.
  Reads and responses remain unrestricted.
- Full-line write serialization is charged once per wave instruction rather
  than once per generated cache-line request. This preserves scalar-store
  timing without penalizing a wide store 16 times; transpose family MARE falls
  from 28.8% to 8.8% while the 33-point MARE falls from 8.6% to 7.0%.
- Sparse read coalescing no longer pays an extra per-line stall on top of the
  actual generated requests and memory latency.
- Wide vector loads and stores use their actual cache, memory, and LDS timing;
  cross-size evidence rejected the synthetic per-line penalties previously
  fitted to one matrix-multiply point.
- Vector XOR/AND/OR use a separate one-cycle timing class, and fully utilized
  stores have an independent 48-cycle per-instruction issue cost.
- Dependent FMA execution uses a twelve-cycle effective occupancy; all four
  matrix-multiply holdouts are within 6.2%, while FIR and k-means remain within
  range.
- GPU-side first/subsequent/post-kernel dispatch costs are calibrated for the
  RX 570's multi-launch workloads. Host cold-start costs remain excluded.

See `progress_rx570.md` at the repository root for the debugging evidence and
parameter sweeps.

## Open

- Add an `s_memtime`/`s_memrealtime` probe for clock-independent latency data.
- Re-measure with the GPU clock pinned; current auto-clock microseconds remain
  diagnostic despite three-process medians.
- Add size sweeps for k-means and pagerank, and investigate the 65K
  vectoradd and canonical pagerank residuals without benchmark-specific timing.
