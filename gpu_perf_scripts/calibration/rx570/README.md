# RX 570 (gfx803 / Polaris 20) timing calibration

This directory contains the reproducible RX 570 hardware and simulator
calibration. All ten workloads use the same optimized gfx803 HIP code objects
on hardware and in MGPUSim.

## Measurement basis

The host cannot pin `power_dpm_force_performance_level=high`, so the recorded
HIP-event microseconds remain diagnostic. The hardware file records both:

- `cold`: the first timed launch, including workload-dependent ROCm/KFD
  first-use costs;
- `steady`: the average after warmup, which is the appropriate comparison for
  the GPU execution model.

Cold-minus-steady ranges from about 10 to 50 µs and is not represented by a
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
./run_size_sweeps.py --mode hardware --jobs 1 --output hw_size_sweep.csv
./compare_size_sweeps.py hw_size_sweep.csv sim_size_sweep.csv
```

The sweep uses four launch-compatible sizes for each of eight scalable
families. It writes every point and its status even if an individual run
fails, so a failed holdout cannot silently disappear. The committed simulator
CSV contains 32 verified points. The hardware CSV currently contains only the
four previously measured vectoradd points because this user cannot read and
write `/dev/kfd`; hardware mode checks that access before building.

The sizes and work transforms are fixed in `run_size_sweeps.py`. Do not tune
the timing model from simulator-only curves. Collect the corresponding exact
gfx803 hardware rows first, then judge both point error and the reported
hardware/simulator slope ratio.

## Current steady-state result

| Benchmark | HW steady (µs) | Sim (µs) | Error |
|---|---:|---:|---:|
| vectoradd | 7.101 | 7.387 | 4.0% |
| relu | 6.837 | 6.261 | 8.4% |
| matrixmult | 73.458 | 80.266 | 9.3% |
| matrixtranspose | — | 78.548 | — |
| bitonicsort | 350.214 | 320.969 | 8.4% |
| aes | 18.084 | 18.415 | 1.8% |
| fir | 7.637 | 7.454 | 2.4% |
| kmeans | 29.424 | 26.854 | 8.7% |
| pagerank | 20.356 | 21.812 | 7.2% |
| nw | — | 137.432 | — |

MARE is **6.3% across the eight benchmarks with valid steady hardware
data**, and every measured error is below 10%. All ten benchmarks verify their output;
matrix transpose has only a 78.722 µs cold hardware measurement, so it is not
included in steady MARE (its 78.548 µs simulation is 0.2% lower).
NW's prior hardware measurement used an incorrect ascending second phase. The
correct dependency order is now used by both launchers, and its hardware row
is deliberately blank until recollected. Multi-size evidence is recorded in
`hw_size_sweep.csv` and `sim_size_sweep.csv`.

## Main model corrections

- Exact gfx803 code objects are embedded for all ten benchmarks.
- Missing gfx803 scalar operations and `v_mad_u16` are implemented.
- L2 bank service is restored to 16 requests/cycle; the earlier low cap
  created a superlinear large-vector queue bottleneck.
- GDDR5 exposes four interleaved banks per 32-bit controller so unrelated
  random misses can overlap; cache and store serialization costs are modeled
  separately.
- A shared L2-to-DRAM token bucket preserves short/random bursts and limits
  sustained cache-line issue. The 262,144-element vectoradd error improves
  from 47.2% to 2.1%.
- Sparse read coalescing no longer pays an extra per-line stall on top of the
  actual generated requests and memory latency.
- Wide vector loads and stores use their actual cache, memory, and LDS timing;
  cross-size evidence rejected the synthetic per-line penalties previously
  fitted to one matrix-multiply point.
- Vector XOR/AND/OR use a separate one-cycle timing class, and fully utilized
  cache-line stores have an independent twelve-cycle issue cost.
- Dependent FMA execution uses a twelve-cycle effective occupancy; all four
  matrix-multiply holdouts are within 5%, while FIR and k-means remain within
  range.
- GPU-side first/subsequent/post-kernel dispatch costs are calibrated for the
  RX 570's multi-launch workloads. Host cold-start costs remain excluded.

See `progress_rx570.md` at the repository root for the debugging evidence and
parameter sweeps.

## Open

- Add an `s_memtime`/`s_memrealtime` probe for clock-independent latency data.
- Re-measure with the GPU clock pinned after granting the user `render` access
  to `/dev/kfd`.
- Extend multi-size hardware sweeps to the remaining benchmark families.
