# gfx90c timing calibration

This directory keeps hardware and MGPUSim measurements reproducible for the
ten benchmark calibration set.

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

Set `ALLOW_UNPINNED_CLOCK=1` only for diagnostics; do not compare that output
with `hw_ground_truth.txt`.

## Simulator measurement

Run the full verified suite:

```bash
SIM_JOBS=4 ./run_sim.sh sim_out | tee sim_results.txt
./compare.py sim_results.txt
```

`SIM_JOBS` controls independent benchmark processes, not simulated GPU
parallelism. Restrict a sweep with a comma-separated `ONLY` list:

```bash
ONLY=matrixmult,matrixtranspose SIM_JOBS=2 \
  ./run_sim.sh sim_out_lds
```

The comparison reports `HW/Sim`, signed error as `(HW/Sim - 1) × 100`, and
mean absolute relative error. Positive error means the simulator is too fast;
negative error means it is too slow.

## Model parameters

The gfx90c platform configuration models:

- seven active CUs in the physical eight-CU floorplan;
- cold and subsequent command-processor launch costs with large-grid
  amortization;
- opcode-class VALU issue intervals and dependent-result latencies;
- pipelined LDS issue, 32 four-byte banks, bank conflicts, and barrier release;
- cache-line utilization penalties for sparse loads/stores and a separate
  locality penalty for non-adjacent wide stores;
- a shared, eight-transaction-per-cycle vector-memory request pipeline;
- the calibrated L1/L2 and dual-channel banked-DDR hierarchy.

Keep one variable family per sweep. Re-run the complete suite before accepting
a parameter because compute, LDS, and memory bottlenecks can compensate for
one another in a single benchmark.
