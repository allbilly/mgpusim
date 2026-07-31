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
wave coalescing. The gfx90c benchmark also includes a dependent vector-memory
variant. It gives every active lane a disjoint randomized cache-line cycle:

```bash
/home/fedora/.local/go/bin/go run ./amd/samples/cache_latency \
  -timing -gpu gfx90c -arch gcn5 -verify \
  -array-bytes 262144 -num-accesses 4096 -active-lanes 8
```

Sweep `-active-lanes 1`, `8`, and `64` first. Even the one-lane vector case
uses `global_load_dword`; `-active-lanes 0` selects the original scalar probe.
The checked-in gfx90c HSACO is loadable by both MGPUSim and the HIP hardware
harness, and its metadata/disassembly must be rechecked after recompilation:

```bash
llvm-objdump --disassemble-symbols=vector_pointer_chase_kernel \
  --mcpu=gfx90c kernels_gfx90c.hsaco
llvm-readelf --notes kernels_gfx90c.hsaco
```

The expected vector ABI is 32 kernarg bytes, with no private-memory use or
spills. Reject a generated binary if the dependent loop is scalarized to an
`s_load_*` instruction.

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

Also sweep the unchanged production kernel at square sizes 32, 64, and 128.
This separates a one-work-group dispatch-dominated point from the 4- and
16-work-group scaling regime. Compare the curve shape and marginal cost rather
than fitting a timing parameter to one matrix size.

## 7. Run geometric application-size sweeps

A default application input gives one calibration point. A geometric size
sweep separates fixed launch cost from steady-state throughput and shows where
occupancy, cache capacity, or memory bandwidth becomes limiting.

Choose legal input sizes that roughly double each step. Respect kernel
constraints such as tile multiples and available deterministic fixtures. The
gfx90c exact-HSACO hardware harness currently supports:

| Benchmark | Suggested size axis | Example geometric sizes |
|-----------|---------------------|-------------------------|
| ReLU | elements | 16,384, 32,768, 65,536, 131,072 |
| matrix multiplication | square width | 32, 64, 128 |
| matrix transpose | square width | 128, 256, 512 |
| AES | input bytes | 1,024, 2,048, 4,096, 8,192 |
| FIR | output length, with taps fixed | 2,048, 4,096, 8,192, 16,384 |
| FIR | taps, with length fixed | 1, 2, 4, 8, 16, 32, 64 |
| K-means | features, with points and clusters fixed | 1, 2, 4, 8, 16, 32 |
| PageRank | nodes at fixed density | 128, 256, 512 |
| Needleman-Wunsch | sequence length | 64, 128, 256 |

Change only one axis per sweep. For example, a FIR tap sweep should hold output
length constant, and a K-means feature sweep should hold points and clusters
constant. Otherwise a changed curve cannot be assigned to a particular
architectural pressure.

### Hardware procedure

Pin the GPU performance policy and confirm the observed shader clock before and
after the sweep. Run a minimum of 20 warm-up launches followed by at least 100
timed launches per size. Repeat the complete sweep at least ten times, randomize
the size order within each repetition, and report the median plus p10/p90 or
median absolute deviation.

The following commands use the repository's exact-HSACO harness:

```bash
# Run from the repository root. This builds the harness and fixtures once.
gpu_perf_scripts/calibration/gfx90c/build_and_run.sh \
  --warmup 20 --iters 200

repo_root="$(pwd)"
harness="$repo_root/gpu_perf_scripts/calibration/gfx90c/build/isca10_bench"

# Run after build_and_run.sh has built the harness and generated fixtures.
for repetition in $(seq 1 10); do
  for n in $(shuf -e 16384 32768 65536 131072); do
    echo "repetition=$repetition relu_length=$n"
    MGPUSIM_ROOT="$repo_root" "$harness" \
      --only relu --relu-length "$n" \
      --warmup 20 --iters 200
  done
done

# Put these loops inside the same repetition loop for a reference sweep.
for n in $(shuf -e 32 64 128); do
  MGPUSIM_ROOT="$repo_root" "$harness" \
    --only matrixmult --matrix-size "$n" \
    --warmup 20 --iters 200
done

for n in $(shuf -e 128 256 512); do
  MGPUSIM_ROOT="$repo_root" "$harness" \
    --only matrixtranspose --transpose-width "$n" \
    --warmup 20 --iters 200
done

for n in $(shuf -e 1024 2048 4096 8192); do
  MGPUSIM_ROOT="$repo_root" "$harness" \
    --only aes --aes-length "$n" \
    --warmup 20 --iters 200
done
```

The direct invocation requires a native ROCm runtime. If ROCm is available only
through the repository's container workflow, pass each size to
`build_and_run.sh` instead. The remaining size controls follow the same
pattern:

```bash
MGPUSIM_ROOT="$repo_root" "$harness" --only fir \
  --fir-length 8192 --fir-taps 32 --warmup 20 --iters 200
MGPUSIM_ROOT="$repo_root" "$harness" --only kmeans \
  --points 4096 --features 8 --clusters 5 --warmup 20 --iters 200
MGPUSIM_ROOT="$repo_root" "$harness" --only pagerank \
  --pagerank-nodes 256 --warmup 20 --iters 200
MGPUSIM_ROOT="$repo_root" "$harness" --only nw \
  --nw-length 128 --warmup 20 --iters 200
```

PageRank hardware fixtures exist for 128, 256, and 512 nodes at density 0.5.
Matrix size must be a multiple of 32, transpose width a multiple of 64, and
Needleman-Wunsch length a multiple of 64. AES length must be a positive
multiple of 1,024 bytes. Use FIR lengths that are multiples of its 256-thread
block size.

If clock pinning is unavailable, set `ALLOW_UNPINNED_CLOCK=1` only to build or
collect diagnostic curves. Record the observed clock for every repetition.
Do not compare those absolute times with pinned 1600 MHz application targets.

Run the matching legal sizes in MGPUSim with the same HSACO, input generator,
launch geometry, and verification enabled. Existing sample flags include:

```bash
/home/fedora/.local/go/bin/go run ./amd/samples/relu \
  -timing -gpu gfx90c -arch gcn5 -disable-rtm -verify -length 65536
/home/fedora/.local/go/bin/go run ./amd/samples/matrixmultiplication \
  -timing -gpu gfx90c -arch gcn5 -disable-rtm -verify \
  -x 64 -y 64 -z 64
/home/fedora/.local/go/bin/go run ./amd/samples/pagerank \
  -timing -gpu gfx90c -arch gcn5 -disable-rtm -verify \
  -node 256 -sparsity 0.5 -iterations 2
/home/fedora/.local/go/bin/go run ./amd/samples/nw \
  -timing -gpu gfx90c -arch gcn5 -disable-rtm -verify -length 128
```

### Interpret the curve

Use architectural work, not merely the command-line size, on the horizontal
axis. Examples include elements for ReLU, matrix elements for transpose,
multiply-accumulates for matrix multiplication and FIR, graph edges for
PageRank, and dynamic-programming cells for Needleman-Wunsch.

Within a region with unchanged behavior, fit:

```text
time(W) = intercept + slope * W
slope = (time(W2) - time(W1)) / (W2 - W1)
```

- The intercept estimates fixed dispatch and completion cost. It is evidence
  for an overhead floor, not a value to subtract blindly from every kernel.
- The large-size slope estimates steady-state cost per work unit. Compare
  hardware and simulator slopes even when an unpinned hardware clock makes
  their absolute times unsuitable for calibration.
- A knee is a sustained slope change. It can indicate occupancy saturation, a
  cache-capacity boundary, TLB pressure, or transition to DRAM bandwidth.
- A single step can instead come from a rounded work-group count or another
  launch-geometry boundary. Record work-group count, waves, bytes, and useful
  work beside every point before interpreting it as a cache or throughput knee.

After finding a knee with geometric sizes, add two or three closely spaced
points on each side. Confirm the proposed cause with transaction counts, cache
hit rates, occupancy, or CPI-stack changes. A useful result table is:

| Size | Useful work | Work-groups | Bytes | HW median | HW spread | Sim time | HW growth | Sim growth | Verified |
|-----:|------------:|------------:|------:|----------:|----------:|---------:|----------:|-----------:|:--------:|

Do not accept a model change because it matches only the default point. It
should predict the intercept, the saturated slope, and any measured knees
without breaking the paired control or the full calibration suite.

## 8. Use counters as supporting evidence

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

## 9. Turn evidence into a simulator change

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

## 10. Commit by milestone

Keep the investigation bisectable:

1. benchmark source, exact HSACO, disassembly, and harness;
2. raw measurements and interpretation;
3. general timing-model change plus unit tests;
4. application and full-suite validation;
5. final calibration table and documentation.

Do not commit generated build directories, temporary metric databases, or
unrelated user files. Each commit message should say what evidence the
milestone adds, not claim accuracy before validation.

## 11. Minimum result record

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
