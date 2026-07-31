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
  launches after 10 warmups plus at least 50 ms of GPU-active warmup. This
  minimum duration is needed for the unpinned card to leave its idle clocks.
  This is the comparison used for the GPU model.

Cold-minus-steady ranges from about 10 to 72 µs and is not represented by a
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
./build_and_run.sh --warmup 10 --warmup-us 50000 --iters 100
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

The sweep uses four launch-compatible sizes for seven families, six k-means
sizes spanning its four-workgroup-per-CU transition, eight matrix-transpose
sizes spanning its L2-capacity transition, plus a ReLU midpoint at 131K.
K-means inputs are deterministic prefixes of the same random stream, while
PageRank gets a separately generated deterministic CSR graph at each node
count. It writes every point and its status
even if an individual run fails, so a failed holdout cannot silently
disappear. Both committed CSV files contain all 47 matched points. Hardware
mode checks `/dev/kfd` access, runs three independent processes per point by
default, warms the GPU for at least 50 ms per process, and records their
median. `--warmup-ms 0` explicitly disables the duration floor.

The sizes and work transforms are fixed in `run_size_sweeps.py`. Do not tune
the timing model from simulator-only curves. Collect the corresponding exact
gfx803 hardware rows first, then judge both point error and the reported
hardware/simulator slope ratio.

## Current steady-state result

| Benchmark | HW steady (µs) | Sim (µs) | Error |
|---|---:|---:|---:|
| vectoradd | 6.765 | 7.996 | 18.2% |
| relu | 7.155 | 7.256 | 1.4% |
| matrixmult | 51.803 | 49.485 | 4.5% |
| matrixtranspose | 16.989 | 14.154 | 16.7% |
| bitonicsort | 347.854 | 323.743 | 6.9% |
| aes | 18.314 | 16.925 | 7.6% |
| fir | 7.699 | 7.646 | 0.7% |
| kmeans | 29.615 | 28.473 | 3.9% |
| pagerank | 19.455 | 22.168 | 13.9% |
| nw | 146.510 | 144.844 | 1.1% |

Canonical-size MARE is **7.5%** across all ten remeasured benchmarks. The
anti-overfit result is **7.4% MARE across 47 matched size points**; eight of
the ten swept families have family MARE below 10%. The maximum point error
is the 16,384-point k-means holdout at 34.3%. All points verify. This newly
exposed residual is intentionally retained: generic cache-capacity and
in-flight-request corrections either did not affect it or regressed
independent vector and transpose holdouts.

## Main model corrections

- Exact gfx803 code objects are embedded for all ten benchmarks.
- Missing gfx803 scalar operations and `v_mad_u16` are implemented.
- L2 bank service is restored to 16 requests/cycle; the earlier low cap
  created a superlinear large-vector queue bottleneck.
- GDDR5 exposes four interleaved banks per 32-bit controller so unrelated
  random misses can overlap; cache and store serialization costs are modeled
  separately.
- A shared L2-to-DRAM token bucket preserves a 1 MiB short/random burst and
  limits sustained cache-line issue; the large vectoradd point remains within
  11% across the clock-warmed hardware median.
- A write-filtered L1-to-L2 token bucket preserves the first 768 KiB of dense
  stores, then limits sustained write ingress to one line per two cycles. The
  burst represents cache-resident dirty capacity and reproduces the measured
  transpose transition without a benchmark or size rule.
  Reads and responses remain unrestricted.
- Full-line write serialization is charged once per wave instruction rather
  than once per generated cache-line request. This preserves scalar-store
  timing without penalizing a wide store 16 times.
- Sparse read coalescing no longer pays an extra per-line stall on top of the
  actual generated requests and memory latency.
- Wide vector loads and stores use their actual cache, memory, and LDS timing;
  cross-size evidence rejected the synthetic per-line penalties previously
  fitted to one matrix-multiply point.
- Vector XOR/AND/OR use a separate one-cycle timing class, and fully utilized
  stores have an independent 90-cycle per-instruction issue cost.
- Dependent FMA execution uses a twelve-cycle effective occupancy; all four
  matrix-multiply holdouts are within 4.9%, while FIR and k-means remain within
  range.
- GPU-side first/subsequent/post-kernel dispatch costs are calibrated for the
  RX 570's multi-launch workloads. Host cold-start costs remain excluded.

See `progress_rx570.md` at the repository root for the debugging evidence and
parameter sweeps.

## Open

- Add an `s_memtime`/`s_memrealtime` probe for clock-independent latency data.
- Re-measure with the GPU clock pinned; current auto-clock microseconds remain
  diagnostic despite three-process medians.
- Investigate the k-means scaling slope and the remaining transpose-384
  residual without benchmark-specific timing.
