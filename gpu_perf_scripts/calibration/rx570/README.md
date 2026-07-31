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

## Current steady-state result

| Benchmark | HW steady (µs) | Sim (µs) | Error |
|---|---:|---:|---:|
| vectoradd | 7.101 | 7.898 | 11.2% |
| relu | 6.837 | 5.902 | 13.7% |
| matrixmult | 73.458 | 74.388 | 1.3% |
| matrixtranspose | — | 72.515 | — |
| bitonicsort | 350.214 | 355.086 | 1.4% |
| aes | 18.084 | 20.243 | 11.9% |
| fir | 7.637 | 5.814 | 23.9% |
| kmeans | 29.424 | 25.982 | 11.7% |
| pagerank | 20.356 | 24.983 | 22.7% |
| nw | 195.281 | 170.430 | 12.7% |

MARE is **12.3% across the nine benchmarks with steady hardware data**. Every
one of the ten benchmarks verifies its output; matrix transpose has only a
78.722 µs cold hardware measurement, so it is not included in steady MARE.
The largest remaining errors are FIR (23.9%) and pagerank (22.7%); the
original 38–59% vector/matrix errors are gone.

## Main model corrections

- Exact gfx803 code objects are embedded for all ten benchmarks.
- Missing gfx803 scalar operations and `v_mad_u16` are implemented.
- L2 bank service is restored to 16 requests/cycle; the earlier low cap
  created a superlinear large-vector queue bottleneck.
- GDDR5 uses one serialized pipeline per 32-bit controller, 160 ns modeled
  latency, and a calibrated 128 GB/s aggregate line-service ceiling.
- Sparse read coalescing no longer pays an extra per-line stall on top of the
  actual generated requests and memory latency.
- A separate 126-cycle wide-read serialization cost models the matrix
  kernel's `flat_load_dwordx4` path.
- GPU-side first/subsequent/post-kernel dispatch costs are calibrated for the
  RX 570's multi-launch workloads. Host cold-start costs remain excluded.

See `progress_rx570.md` at the repository root for the debugging evidence and
parameter sweeps.

## Open

- Add an `s_memtime`/`s_memrealtime` probe for clock-independent latency data.
- Re-measure with the GPU clock pinned after granting the user `render` access
  to `/dev/kfd`.
- Investigate FIR underprediction and pagerank random-load overlap without
  benchmark-specific timing rules.
