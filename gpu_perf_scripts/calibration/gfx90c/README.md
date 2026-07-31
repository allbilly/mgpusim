# gfx90c timing calibration

This directory keeps hardware and MGPUSim measurements reproducible for the
ten benchmark calibration set.

`hw_ground_truth.txt` is the accepted legacy pinned table. It was collected
with warmed, many-iteration native HIP kernels before the exact-HSACO
single-shot harness below existed. `compare.py` therefore reports a calibration
score, not fully matched validation. Fresh pinned exact-HSACO targets are
required for that; matrix multiplication additionally needs replacement
because its source and HSACO were corrected after the legacy measurement.

## Rebuild and verify code objects

Both paths use the embedded `kernels_gfx90c.hsaco` files. Regenerate all code
objects with the pinned ROCm image and optimization level, then verify their
digests and exported kernel symbols:

```bash
./regen_hsaco.sh
./verify_hsaco.sh
```

Compiler identity, binary hashes, sources, destinations, and exported entry
points are recorded in `hsaco_manifest.txt`.

## Hardware measurement

Reference results require the GPU performance policy to be `high`. The runner
refuses an unpinned clock because `hipGetDeviceProperties().clockRate` reports
the maximum clock, not the clock actually used by a launch.

**Thermal safety (Renoir iGPU):** the integrated Radeon on Renoir has limited
thermal headroom and can hard-lockup under sustained load. On 2026-07-30, four
back-to-back full-suite containers caused an amdgpu hard lockup after the 4th
run's cache_latency jumped 30%. To avoid this:
- Run one benchmark at a time: `./build_and_run.sh --only cache_latency`
- Insert 30–60 s cooling pauses between runs
- Monitor edge temp: `cat /sys/class/hwmon/hwmon4/temp1_input` (millidegrees)
- Do not run multiple containers in parallel

After pinning the clock with the host's normal privileged administration
workflow:

```bash
./build_and_run.sh
```

The build first regenerates deterministic binary fixtures from the same Go
matrix generator and random seeds used by the simulator. The PageRank CSR
graph, initial rank vector, K-means features, and initial centroids therefore
match on both measurement paths. Generated fixtures live under `build/`.

The default is one cold launch sequence, matching one MGPUSim sample run.
Steady-state behavior can be measured explicitly:

```bash
./build_and_run.sh --warmup 1 --iters 100
```

K-means can additionally report its swap and compute kernels separately:

```bash
./build_and_run.sh --only kmeans --components --warmup 1 --iters 100
```

The exact-HSACO harness supports application size sweeps without recompiling:

```bash
./build/isca10_bench --only relu --relu-length 32768
./build/isca10_bench --only matrixmult --matrix-size 64
./build/isca10_bench --only matrixtranspose --transpose-width 256
./build/isca10_bench --only aes --aes-length 2048
./build/isca10_bench --only fir --fir-length 8192 --fir-taps 32
./build/isca10_bench --only kmeans --points 2048 --features 16 --clusters 5
./build/isca10_bench --only pagerank --pagerank-nodes 256
./build/isca10_bench --only nw --nw-length 64
```

PageRank hardware fixtures are generated for 128, 256, and 512 nodes at
sparsity 0.5, using the same deterministic CSR generator as the simulator.

Set `ALLOW_UNPINNED_CLOCK=1` only for diagnostics; do not compare that output
with `hw_ground_truth.txt`.

## Simulator measurement

Run the full verified suite:

```bash
set -o pipefail
SIM_JOBS=4 ./run_sim.sh sim_out | tee sim_results.txt
./compare.py sim_results.txt
```

`SIM_JOBS` controls independent benchmark processes, not simulated GPU
parallelism. Each benchmark writes `<name>.rc` for its overall status and
`<name>.sim.rc` for the simulator process status if that process ran. The
runner finishes all launched jobs and exits nonzero if any build, simulation,
or metric extraction fails. Restrict a sweep with a comma-separated `ONLY`
list:

```bash
ONLY=matrixmult,matrixtranspose SIM_JOBS=2 \
  ./run_sim.sh sim_out_lds
```

The comparison reports `HW/Sim`, signed error as `(HW/Sim - 1) × 100`, and
mean absolute relative error. Positive error means the simulator is too fast;
negative error means it is too slow. The corrected matrix-multiplication
kernel is displayed but excluded from aggregates because its checked-in
hardware target predates the source and HSACO correction.

## Model parameters

The gfx90c platform configuration models:

- seven active CUs in the physical eight-CU floorplan;
- cold and subsequent command-processor launch costs with large-grid
  amortization;
- opcode-class VALU issue intervals and dependent-result latencies;
- pipelined LDS issue, 32 four-byte banks, bank conflicts, and barrier release;
- cache-line utilization penalties for sparse loads/stores and a separate
  locality penalty for non-adjacent wide stores;
- a shared vector-memory transaction-group issue path;
- the calibrated L1/L2 and dual-channel banked-DDR hierarchy.

Keep one variable family per sweep. Re-run the complete suite before accepting
a parameter because compute, LDS, and memory bottlenecks can compensate for
one another in a single benchmark.
