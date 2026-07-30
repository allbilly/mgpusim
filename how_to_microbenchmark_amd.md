# How to Microbenchmark an AMD GPU for MGPUSim Calibration

This guide describes a repeatable workflow for measuring an AMD GPU and using
the results to improve MGPUSim. It is aimed at gfx9-class hardware such as
gfx90c, but most of the method applies to newer AMD architectures.

The central rule is to change one architectural pressure at a time. A full
application time can show that the simulator is wrong; a small paired
microbenchmark can show which model is wrong.

## 1. Keep the comparison controlled

Use the same:

- HSACO/code object on hardware and in MGPUSim;
- kernel symbol, arguments, grid, work-group size, and dynamic LDS size;
- initialized input bytes and allocation sizes;
- warm-up and iteration policy;
- GPU clock policy.

Do not compare a freshly compiled native kernel with an old embedded simulator
binary. Compiler changes can alter occupancy, spilling, instruction selection,
and cache policy.

Record the environment with every result:

```bash
rocminfo | rg 'Name:|Marketing Name|gfx'
hipcc --version
uname -a
cat /sys/class/drm/card*/device/power_dpm_force_performance_level
cat /sys/class/drm/card*/device/pp_dpm_sclk
```

On a dedicated test host, pin the performance level before collecting reference
times:

```bash
sudo sh -c 'echo high > /sys/class/drm/card1/device/power_dpm_force_performance_level'
```

The correct `cardN` is host-specific. Resolve it from `rocminfo` and the PCI
device path; never assume that `card0` is the compute GPU. If the clock cannot
be pinned, label the run diagnostic rather than reference-quality. Prefer
cycle counts and ratios between paired kernels for an unpinned run.

The repository's gfx90c application harness enforces this distinction:

```bash
gpu_perf_scripts/calibration/gfx90c/build_and_run.sh \
  --warmup 20 --iters 200

# Diagnostic only when pinning is unavailable:
ALLOW_UNPINNED_CLOCK=1 \
  gpu_perf_scripts/calibration/gfx90c/build_and_run.sh \
  --warmup 20 --iters 200
```

## 2. Measure device work, not host overhead

For application-sized kernels, bracket a batch of launches with HIP events.
Warm up before recording the start event, synchronize the stop event, and
divide the batch time by the iteration count:

```cpp
hipEvent_t start, stop;
hipEventCreate(&start);
hipEventCreate(&stop);

for (int i = 0; i < warmups; ++i)
  launch();
hipDeviceSynchronize();

hipEventRecord(start);
for (int i = 0; i < iterations; ++i)
  launch();
hipEventRecord(stop);
hipEventSynchronize(stop);

float milliseconds = 0;
hipEventElapsedTime(&milliseconds, start, stop);
double microseconds_per_iteration =
    milliseconds * 1000.0 / iterations;
```

For very short kernels, also measure an empty kernel with the same launch
geometry. Do not blindly subtract it: launch and execution can overlap in a
batch. Treat it as evidence for a launch-overhead floor.

For dependent-latency probes, device cycle counters are more robust than HIP
events. On GCN, use `s_memtime` or `s_memrealtime`, put an explicit
`s_waitcnt` around the memory dependency, and store the delta. Inspect the
generated ISA to ensure the compiler did not hoist, remove, vectorize, or
replace the intended operation.

Report distributions, not only one mean. A practical default is:

- 20–100 untimed warm-ups;
- at least 100 timed batches;
- median as the headline result;
- p10/p90 or median absolute deviation as stability information;
- randomized test order when comparing several variants.

## 3. Preserve and inspect the exact HSACO

Compile for the real target and save intermediate files:

```bash
hipcc -O3 --offload-arch=gfx90c --save-temps -c probe.cpp
cp probe-hip-amdgcn-amd-amdhsa-gfx90c.o probe_gfx90c.hsaco

/opt/rocm/llvm/bin/llvm-objdump \
  --disassemble --mcpu=gfx90c probe_gfx90c.hsaco > probe_gfx90c.disasm

/opt/rocm/llvm/bin/llvm-readobj \
  --notes probe_gfx90c.hsaco > probe_gfx90c.metadata
```

Check at least:

- the expected memory opcode (`global_*`, `flat_*`, `buffer_*`, or `ds_*`);
- `s_waitcnt` placement;
- loop trip count and unrolling;
- VGPR and SGPR counts;
- `group_segment_fixed_size` (LDS);
- `private_segment_fixed_size` (scratch/spill);
- wave size and work-group geometry.

Embed or load that exact code object on the simulator side. Keep the source,
HSACO, disassembly, metadata, input generator, and raw results together.

## 4. Establish cache latency and capacity

A pointer chase exposes dependent-load latency:

```cpp
index = array[index];
```

Build one randomized Hamiltonian cycle over cache-line-spaced nodes. Random
order defeats prefetching, while one node per cache line prevents intra-line
spatial reuse. Sweep powers of two across the hierarchy:

```text
4 KiB, 8 KiB, 16 KiB, 32 KiB, 64 KiB, ...
512 KiB, 1 MiB, 2 MiB, 4 MiB, ...
```

Use enough complete laps to amortize the cold first lap. The repository already
contains a scalar pointer-chase implementation:

```bash
/home/fedora/.local/go/bin/go run ./amd/samples/cache_latency \
  -timing -gpu gfx90c -arch gcn5 \
  -array-bytes 16384 -cacheline-bytes 64
```

The scalar probe is useful for hierarchy latency, but it does not characterize
wave coalescing. Add a vector-load version before changing VMEM behavior.

### Vector stride and coalescing sweep

Launch one wave and have each active lane access:

```text
base + lane_id * stride
```

Sweep:

- active lanes: 1, 2, 4, 8, 16, 32, 64;
- stride: 4, 8, 16, 32, 64, 128, 256 bytes;
- load width: 4, 8, and 16 bytes;
- aligned and deliberately misaligned bases.

Record latency, generated memory transactions, and cache hit rate in both
hardware tooling and MGPUSim. This isolates line formation and request issue
width from raw cache latency.

## 5. Measure cross-kernel reuse for K-means

K-means first transposes its feature matrix, then immediately consumes that
matrix. A pointer chase cannot determine whether those producer writes are
resident for the consumer.

Use a two-kernel probe:

1. producer: each lane writes a coalesced line into buffer `B`;
2. optional interference kernel: touches a configurable unrelated footprint;
3. consumer: reads `B` with the same stride and wave geometry as the K-means
   compute kernel and reduces to a checksum.

Sweep `B` across 16 KiB, 64 KiB, 128 KiB, 256 KiB, 512 KiB, 1 MiB, and 2 MiB.
For each size collect four variants:

```text
cold consumer
producer -> consumer
producer -> interference -> consumer
producer -> explicit cache flush -> consumer
```

Also compare:

- host-to-device initialized input versus GPU-produced input;
- one producer/consumer pair versus repeated pairs;
- the exact K-means feature layout versus a contiguous control layout.

Useful conclusions:

- Faster `producer -> consumer` than cold consumer indicates retained cache
  state across launches.
- A sharp loss after an interference footprint estimates effective capacity.
- A difference between host-initialized and GPU-produced data indicates DMA
  cache-fill or coherence policy.
- A layout-only difference points to coalescing or cache-bank pressure rather
  than launch overhead.

Do not fix K-means by naming the benchmark in the timing model. Represent the
measured general mechanism: DMA fill policy, write allocation, dirty-line
retention, eviction, or cross-kernel cache state.

## 6. Measure private scratch for matrix multiplication

On gfx9, compiler spills and private arrays commonly appear as `buffer_load_*`
and `buffer_store_*` instructions using the private-segment resource. Create
paired kernels that are identical except for scratch use:

- `control`: arithmetic and global accesses, no private segment;
- `scratch`: dynamically index a volatile private array to force a nonzero
  `private_segment_fixed_size`.

Use a runtime-derived index so the compiler cannot scalarize the array. Verify
the spill in metadata and disassembly.

Measure three complementary cases:

### Dependent scratch latency

Each iteration loads a private value whose address or next operation depends on
the preceding result. This exposes completion latency and `waitcnt` handling.

### Independent scratch throughput

Issue several independent scratch loads or stores before consuming the results.
Sweep the number of active lanes and independent operations. This exposes
transaction formation, pipeline width, and request concurrency.

### Matrix-multiply decomposition

Compare:

- ALU-only inner loop;
- LDS-only tiled load/barrier loop;
- global + LDS + ALU without forced spill;
- the same kernel with forced spill;
- the production matrix-multiply HSACO.

The paired delta is more useful than either absolute time:

```text
scratch cost = time(scratch variant) - time(control variant)
```

Sweep work-group count so the test also reveals whether private-segment traffic
incorrectly limits occupancy or serializes across waves.

If hardware and simulator disagree only for dependent scratch, adjust scratch
latency/response completion. If they disagree only at many active lanes, adjust
private-address mapping, coalescing, transaction issue width, or concurrency.
If both scratch-free and spilled matmul disagree, investigate VALU, LDS,
barriers, or occupancy before changing scratch.

## 7. Use counters as supporting evidence

Query the counters that the installed ROCm stack actually exposes:

```bash
rocprofv3 --list-avail
```

Then collect kernel dispatch and relevant cache/memory counters. Counter names
vary by GPU and ROCm release, so do not copy an MI300 counter list into a gfx90c
script without checking availability.

In MGPUSim, enable the closest corresponding reports:

```bash
/home/fedora/.local/go/bin/go run ./amd/samples/cache_latency \
  -timing -gpu gfx90c -arch gcn5 \
  -report-cache-hit-rate \
  -report-cache-latency \
  -report-dram-transaction-count \
  -report-cpi-stack
```

Compare invariants before timing:

- retired instruction counts;
- cache-line request counts;
- read/write transaction counts;
- hit/miss direction;
- occupancy and number of resident waves.

A timing knob should not compensate for an incorrect transaction count.

## 8. Turn evidence into a simulator change

Use this decision sequence:

1. Confirm exact inputs, HSACO, launch geometry, and outputs.
2. Match instruction and memory-transaction counts.
3. Match cache residency and hit/miss transitions.
4. Match dependency latency and issue throughput separately.
5. Only then tune fixed launch overhead.

Prefer a parameter or mechanism that predicts an entire microbenchmark curve.
Reject a change that improves one point but breaks the curve shape.

For every candidate model change, run:

- the microbenchmark variant that motivated it;
- its paired control;
- K-means and matrix multiplication;
- the full calibration suite.

Track signed error consistently:

```text
error = hardware / simulator - 1
```

Negative error means the simulator is slower. A useful summary is median or
mean absolute relative error across the full suite, accompanied by per-test
errors so improvements cannot hide regressions.

## 9. Commit by milestone

Keep the investigation bisectable:

1. benchmark source, exact HSACO, disassembly, and harness;
2. raw measurements and interpretation;
3. general timing-model change plus unit tests;
4. application and full-suite validation;
5. final calibration table and documentation.

Do not commit generated build directories, temporary metric databases, or
unrelated user files. Each commit message should say what evidence the
milestone adds, not claim accuracy before validation.

## 10. Minimum result record

For each run, save:

```text
date and host
GPU marketing name and gfx target
ROCm/compiler version
clock policy and observed clock state
git commit
HSACO hash
kernel symbol and launch geometry
input/working-set size
warm-up and iteration counts
median and spread
relevant counters
simulator configuration
simulated time
hardware/simulator ratio
```

This metadata is what makes a microbenchmark a calibration experiment rather
than a one-off timing number.
