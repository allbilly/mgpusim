# gfx90c microbenchmark results

This file records the targeted measurements used after the initial ISCA-10
calibration. It includes negative results so that later work does not repeat
knob sweeps that have already been falsified.

## Measurement status

- GPU: Renoir integrated Radeon, gfx90c, 7 active CUs.
- Reference application results in `hw_ground_truth.txt` were measured with
  the performance policy pinned to `high` (1600 MHz).
- The microbenchmark measurements below were collected with the policy at
  `auto`. During the long pointer chase, repeated reads of
  `pp_dpm_sclk` showed `200Mhz *` for the entire kernel.
- Therefore the microbenchmark hardware numbers are diagnostic curves and
  component ratios. They are not replacements for the pinned application
  ground truth.
- Runs were sequential and the GPU edge temperature remained 44-47 C.

## K-means feature-count sweep

Commit `96d12f4b` parameterized the exact-HSACO hardware harness with
`--points`, `--features`, and `--clusters`. The default 4096x16 input still
uses the exact fixture. Non-default sizes use deterministic random values;
the values do not affect the kernels' control flow.

Hardware command pattern:

```bash
ALLOW_UNPINNED_CLOCK=1 \
  ./build_and_run.sh --only kmeans --components \
  --points 4096 --features 16 --clusters 5 --warmup 2 --iters 10
```

At 4096 points and 5 clusters:

| Features | HW total (us) | HW swap (us) | HW compute (us) | Sim swap (us) |
|---------:|--------------:|-------------:|----------------:|--------------:|
| 1        | 51.880        | 16.010       | 39.264          | 4.544         |
| 2        | 71.328        | 21.780       | 52.629          | 5.002         |
| 4        | 112.050       | 31.933       | 80.672          | 8.394         |
| 8        | 199.261       | 52.505       | 143.496         | 24.533        |
| 16       | 376.700       | 124.082      | 257.265         | 29.455        |
| 32       | 745.085       | 240.537      | 502.595         | not run       |

The absolute HW/simulator ratio is invalid at different clocks. The curve
shape is still useful:

- Hardware swap time grows smoothly with feature count.
- The simulator has an artificial jump at 8 features. The low-utilization
  coalescing penalty affects the 4- and 8-feature patterns; at 16 features
  each cache-line request has one lane and the penalty is intentionally not
  applied.
- At the default point, simulator swap is 29.455 us and full K-means is
  59.096 us. Thus the simulated compute phase plus the second dispatch costs
  about 29.6 us.
- Hardware at the same unpinned state spends roughly twice as long in compute
  as swap (257.3 vs 124.1 us). Any future change must report the two kernels
  separately; fitting only their sum can alter the wrong phase.

## K-means swap profile

The default swap kernel disassembly has a 16-iteration loop:

```text
global_load_dword
s_waitcnt vmcnt(0)
global_store_dword
```

For 64 lanes and 16 features, the load touches 64 different 64-byte lines.
The store writes four contiguous full cache lines.

Simulator profile at 4096x16:

- swap kernel time: 29.455 us;
- L1V hit rate: approximately 5%;
- L2 read hits: approximately 69,560;
- L2 request latency: approximately 111 ns;
- L1V request latency including miss service: approximately 203-218 ns;
- VMem CPI contribution: approximately 14-15 cycles/instruction, the largest
  measured stack component.

Earlier work established that L1V thrashing is real for this layout: all
resident waves map their input lines to the same 64 sets of the 16 KiB,
4-way cache. One wave has high reuse, while many concurrent waves evict one
another.

## Scalar cache-latency probe

The repository pointer chase uses one scalar thread and `s_load_dword`.
Consequently it validates the scalar/L2 hierarchy but not vector coalescing.

Hardware, 256 KiB, 64-byte randomized nodes:

| Accesses | Warm-up kernels | Time (us) | ns/access | Observed sclk |
|---------:|----------------:|----------:|----------:|--------------:|
| 4,096    | 0               | 7,158.814 | 1,747.757 | 200 MHz       |
| 4,096    | 1               | 6,971.572 | 1,702.044 | 200 MHz       |
| 262,144  | 1               | 311,833.375 | 1,189.550 | 200 MHz     |

The one-lap cold/warm difference is only 2.6%. Repeated laps within one kernel
lower the average as the 256 KiB working set becomes resident.

Simulator, 256 KiB, 4,096 accesses:

| Initialization | Time (us) | ns/access | Cache observation |
|----------------|----------:|----------:|-------------------|
| DMA bypasses L2 | 2,102.325 | 513.263 | first lap is almost all DRAM misses |
| DMA warms L2    | 446.553   | 109.022 | 4,100 L2 hits, no L2 read misses |

The first-lap result must not be compared with a many-lap hardware average.
Also, off-core memory does not necessarily scale with shader clock, so
converting a 200 MHz DRAM miss to 1600 MHz GPU cycles is not a valid absolute
calibration. A vector pointer chase and a pinned 1600 MHz hardware run remain
the required measurements for final L2 fitting.

## Vector cache-latency probe

Commit `47755a8d` added a one-wave vector probe in which each active lane
follows a disjoint randomized cycle of 64-byte-spaced nodes. The checked-in
gfx90c HSACO is a loadable code object shared by MGPUSim and the HIP harness.
Its dependent loop was verified to contain `global_load_dword` followed by
`s_waitcnt vmcnt(0)`; metadata reports a 32-byte kernarg segment, zero spills,
and zero private memory.

Times below use 256 and 1,024 dependent iterations. The slope is
`(time_1024 - time_256) / 768`, which removes most fixed dispatch and cold-start
cost.

At a 16 KiB total working set:

| Active lanes | Sim 256 (us) | Sim 1024 (us) | Sim slope (ns) | HW 256 (us) | HW 1024 (us) | HW slope (ns) |
|-------------:|-------------:|--------------:|---------------:|------------:|-------------:|--------------:|
| 1            | 38.368       | 109.731       | 92.9           | 457.254     | 1,426.796    | 1,262.4       |
| 8            | 39.185       | 142.595       | 134.6          | 354.789     | 1,324.216    | 1,262.3       |
| 64           | 57.130       | 214.810       | 205.3          | 429.581     | 1,676.100    | 1,623.1       |

The hardware policy was still `auto`, and sysfs showed 200 MHz before and
after the sweep. For this L1-resident test only, converting the slopes to
shader-clock cycles gives an informative diagnostic: the 64-lane wave is
approximately 329 simulated cycles versus 325 observed cycles. The simulator
undercharges partially active waves, but the fully active vector-L1 path used
by the default K-means kernels is already close. This falsifies a broad
full-wave VMEM latency reduction as the K-means fix.

At a 256 KiB total working set:

| Active lanes | Sim 256 (us) | Sim 1024 (us) | Sim slope (ns) | HW 256 (us) | HW 1024 (us) | HW slope (ns) |
|-------------:|-------------:|--------------:|---------------:|------------:|-------------:|--------------:|
| 1            | 148.251      | 558.775       | 534.5          | 539.458     | 2,111.968    | 2,047.5       |
| 8            | 144.285      | 351.015       | 269.2          | 626.161     | 2,444.148    | 2,367.2       |
| 64           | 83.718       | 225.318       | 184.4          | 766.972     | 3,019.284    | 2,932.7       |

The simulator and hardware curves have opposite active-lane trends for this
larger footprint. However, off-core service time does not scale directly with
the observed shader clock, so these unpinned numbers cannot set an absolute L2
latency. Their safe conclusion is directional: reducing general L2/global-load
latency to fit matmul is not supported by this probe and would also worsen
PageRank and FIR.

## K-means producer/consumer isolation

Commit `e8939d33` added `-preinitialize-swap` to the simulator sample and
`--preinitialize-swap` to the exact-HSACO hardware harness. This uploads the
same feature-major data and skips `kmeans_kernel_swap`, leaving the compute
kernel and benchmark verification unchanged.

Paired default-size results:

| Path | Simulator (us) | HW auto/200 MHz (us) |
|------|---------------:|---------------------:|
| swap + compute | 59.096 | 377.704 |
| swap only | 29.455 | 123.976 |
| compute immediately after swap, by subtraction | 29.641 | 253.728 |
| compute with preinitialized feature-major input | 27.836 | 259.321 |
| repeated compute component | not measured | 258.467 |

Within each same-clock pair, hardware compute is about 2.2% faster immediately
after the producer, while simulator compute is about 6.5% slower. The direction
is wrong, consistent with the model's write-around L1V policy failing to retain
the producer's contiguous full-line stores. However, the effect is small:
cross-kernel residency can explain only a few percent of compute time, not the
full K-means error. The earlier write-through experiment was not a valid fix
because that implementation also waits for lower-level write acknowledgement
and made the producer slower.

A follow-up candidate moved the initial cluster upload before the swap kernel,
matching the hardware harness order and avoiding one conservative driver
flush. It worsened K-means from 59.096 to 61.034 us and was reverted. The key
reason is that `transposeFeatures()` calls `verifySwap()`, whose device-to-host
copy already flushes and invalidates the feature buffer before compute. The
later cluster upload is therefore a redundant flush, not the primary loss of
producer residency. The driver's sticky `markAllBuffersDirty` behavior is
over-conservative, but changing upload order does not repair this benchmark.

The accepted fix combines both necessary changes: upload the initial clusters
before swap and defer `verifySwap()` until after compute. This removes every
host transfer between the producer and consumer while preserving the same
verification afterward. K-means improved from 59.096 to 50.473 us, and both
the default one-iteration case and a three-iteration case passed CPU/RMSE
verification. The remaining error is +28.7%, down from +50.7%.

## Matrix-multiplication size sweep

Commit `3fe47656` parameterized the exact-HSACO harness with
`--matrix-size`. The same tiled kernel and 8x8 work-group were measured at
three square sizes:

| Matrix size | Work-groups | Sim (us) | HW auto/200 MHz (us) | Sim growth | HW growth |
|------------:|------------:|---------:|---------------------:|-----------:|----------:|
| 32          | 1           | 15.866   | 145.255              | -          | -         |
| 64          | 4           | 22.685   | 267.803              | 1.43x      | 1.84x     |
| 128         | 16          | 45.141   | 518.458              | 1.99x      | 1.94x     |

Absolute comparison remains invalid while hardware is unpinned, but the
saturated 64-to-128 growth agrees closely. Combined with the vector pointer
chase and the disassembly result that scratch is only 1.2% of instructions,
this provides no support for changing general L2 latency, LDS throughput, or
matmul-specific execution timing. The remaining pinned-reference error is
+15.4%, already below 20%; a pinned-clock LDS/VALU component probe is required
before another timing-model change.

The benchmark verifier also contained an unrelated loop-variable error that
checked only part of one row. Commit `91808957` now checks every output element
and adds a regression test, ensuring future size/ISA variants cannot produce a
false pass.

## Accepted general fixes

Two benchmark-driven fixes improved the K-means/matmul state:

1. `fe4e4f8c` fixed per-die dispatch for a non-divisible CU count. The old
   algorithm assigned work to only 4 of 7 CUs. Full K-means improved from
   69.385 to 59.742 us; matrix multiplication improved from 50.728 to
   45.141 us.
2. `4683ba25` corrected the modeled dual-channel DDR4 peak bandwidth from
   128 GB/s to 42.7 GB/s while preserving approximately 400 ns round-trip
   latency. Full K-means moved to 59.096 us with no suite regression.
3. `b2885f92` moved K-means diagnostic verification after the consumer and
   uploaded the initial clusters before the producer. The old host transfers
   forced a cache flush between the exact swap/compute pair. K-means improved
   from 59.096 to 50.473 us without changing kernel code or results.

Relative to the state at the start of this investigation:

- matrix multiplication: 50.728 -> 45.141 us (HW 39.107);
- K-means: 69.385 -> 50.473 us (HW 39.220).

## Rejected candidates

### L2 bank latency 128 -> 64 cycles

Selected results:

| Benchmark | 128-cycle sim (us) | 64-cycle sim (us) | HW (us) |
|-----------|-------------------:|------------------:|--------:|
| matrixmult | 45.141 | 40.196 | 39.107 |
| kmeans | 59.096 | 50.992 | 39.220 |
| pagerank | 112.948 | 96.415 | 130.638 |
| fir | 8.338 | 7.896 | 12.003 |

Matmul became close and K-means improved, but full-suite MARE worsened from
17.9% to 18.8%. PageRank and FIR, already too fast, became severe outliers.
The candidate was reverted.

### Vector transaction pipeline width 1 -> 4

GCN documentation describes forming up to four coalesced addresses per cycle,
but changing this internal pipeline alone did not model that end-to-end path:

- K-means: 59.096 -> 59.632 us;
- matrix multiplication: 45.141 -> 44.920 us.

The candidate was reverted. A future bandwidth change must include the whole
CU-to-cache path and be validated with a vector fan-out microbenchmark.

### L1V MSHR entries 128 -> 512

A 64-line strided load appears capable of consuming half the configured MSHRs,
but increasing both MSHRs and maximum concurrent cache transactions had no
effect:

- swap: 29.455 -> 29.529 us;
- full K-means: 59.096 -> 59.369 us;
- matrix multiplication and NW were unchanged.

Therefore MSHR capacity is not the controlling queue. The candidate and its
temporary configuration API were removed.

### L1V write-around -> write-through/write-allocate

Although write allocation could preserve the producer's transposed lines for
the compute kernel, the current cache implementation also waits for lower-level
write acknowledgement. Results were neutral or slightly worse:

- swap: 29.455 -> 29.838 us;
- full K-means: 59.096 -> 59.524 us.

The candidate and its temporary configuration API were removed. Store-buffer
retirement must be measured separately from cache allocation policy.

## Next discriminating experiment

The vector and producer/consumer probes rule out broad full-wave L1V and L2
latency reductions. The accepted uninterrupted kernel sequence recovers much
of the K-means error; further work would require decoupling write allocation
from lower-level write acknowledgement and validating the remaining small
consumer reuse benefit. Matmul should remain a separate
instruction-mix/local-memory investigation; its scratch traffic is only about
1.2% of dynamic instructions and the vector probe does not support lowering
general L2 latency.
