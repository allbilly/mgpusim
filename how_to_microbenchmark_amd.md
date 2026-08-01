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

### Exact-HSACO load width, fanout, and wait sweep

The checked-in gfx90c VMEM probe turns that design into twelve stable kernel
symbols. It crosses 4-, 8-, and 16-byte global loads with four dependency
modes:

- `serial`: each load drains through `vmcnt(0)` before its value is consumed;
- `independent2`, `independent4`, and `independent8`: exactly 2, 4, or 8
  source-level independent loads precede one drain, exposing the overlap
  curve rather than only one independent endpoint.

This design follows the transferable parts of *Dissecting the NVIDIA
Blackwell Architecture with Microbenchmarks*
([Jarmusch et al.](https://arxiv.org/html/2507.10789v1)): distinguish a true
dependent chain from independent completion, time repeated operations, sweep
warp pressure and access shape, report robust central values such as medians,
and audit the final machine code. For AMD, use `s_memrealtime`/`s_memtime`
rather than NVIDIA's `%clock64` when an in-kernel cycle measurement is needed.
The paper's NVIDIA-specific cache, scheduler, instruction, and clock values do
not transfer to gfx90c. It also does not justify blindly subtracting an empty
launch from event-timed kernels.

Fanout uses the matrix-like interleaved lane mapping. At fanout 8, lanes
`0,8,...,56` share one address, lanes `1,9,...,57` share another, and so on.
The host passes the already-computed address-group count and the kernel derives
its power-of-two mask with one subtraction, avoiding a runtime integer divide
that would contaminate the loop.

Use 8 KiB and 64 KiB for the first capacity comparison. Eight KiB is safely
inside the 16 KiB L1V; 64 KiB exceeds L1V but remains below the current
128 KiB DMA-through-L2 limit. A 256 KiB initialization would bypass modeled
L2 and silently change cache state relative to warmed hardware.

One footprint lap is not a capacity measurement: it is compulsory traversal,
and the verified simulator sweep had zero L1V hits at every one-lap size.
Use the one-lap formula only to align coverage. For an L1 comparison, run at
least four laps, retain per-point cache counters, and fit the post-first-lap
repeat slope separately. Compare footprints at the same repeat counts; do not
infer an L1 knee from equal-lap raw times whose dynamic load counts differ.

Run one hardware point with:

```bash
gpu_perf_scripts/calibration/gfx90c/build_and_run.sh \
  --only vmemloadshape --vmem-width-dwords 4 \
  --vmem-mode serial --vmem-alias-lanes 8 \
  --vmem-array-bytes 65536 --vmem-repeats 512 \
  --vmem-workgroups 1 --warmup 20 --iters 1000
```

Run the identical simulator point with:

```bash
go run ./amd/samples/vmem_load_shape \
  -timing -arch gcn5 -gpu gfx90c -disable-rtm -verify \
  -width-dwords 4 -mode serial -alias-lanes 8 \
  -array-bytes 65536 -repeats 512 -workgroups 1 \
  -report-cache-hit-rate -report-cache-latency \
  -report-dram-transaction-count -report-cpi-stack
```

The optional wide-return model is deliberately excluded from the gfx90c
default. Exercise it only as an explicit calibration variable, keeping every
other simulator argument fixed:

```bash
for lane_dwords_per_cycle in 0 4 8 16 32 64; do
  /home/fedora/.local/go/bin/go run ./amd/samples/vmem_load_shape \
    -timing -arch gcn5 -gpu gfx90c -disable-rtm -verify \
    -vmem-load-return-lane-dwords-per-cycle "$lane_dwords_per_cycle" \
    -width-dwords 4 -mode serial -alias-lanes 8 \
    -array-bytes 8192 -repeats 64 -workgroups 1
done
```

Repeat the candidate sweep for widths 1, 2, and 4, all four dependency modes, and
the production matrix and K-means size sweeps. A useful value must improve the
wide, sparse, wait-heavy matrix path without materially moving the narrow,
high-occupancy K-means path. Do not enable a value in `gfx90c.MakeBuilder`
until pinned hardware width and work-group slopes select it and the default
zero setting has passed the complete regression suite.

If the total-return model moves narrow dword workloads, test the stricter
wave-owned, wide-only candidate separately:

```bash
for lane_dwords_per_cycle in 0 1 2 4; do
  /home/fedora/.local/go/bin/go run ./amd/samples/vmem_load_shape \
    -timing -arch gcn5 -gpu gfx90c -disable-rtm -verify \
    -vmem-wide-load-return-lane-dwords-per-cycle "$lane_dwords_per_cycle" \
    -width-dwords 4 -mode independent4 -alias-lanes 8 \
    -array-bytes 8192 -repeats 64 -workgroups 1
done
```

This second candidate charges only dwords beyond the first returned dword per
active lane and shares that extra service between loads from the same wave.
Different waves still overlap. Therefore dword/x2/x4 contribute 0/64/192
extra lane-dwords for a full wave, K-means-like dword traffic should remain
bit-identical, and equal-load-count serial/independent probes should receive
nearly the same absolute added body cost. Never set both experimental flags in
one run.

If the matched width sweep shows that x2 and x4 require different linear
extra-dword bandwidths, exercise the packed-width CU candidate separately:

```bash
for units_per_cycle in 0 1 2; do
  /home/fedora/.local/go/bin/go run ./amd/samples/vmem_load_shape \
    -timing -arch gcn5 -gpu gfx90c -disable-rtm -verify \
    -vmem-cu-wide-return-units-per-cycle "$units_per_cycle" \
    -width-dwords 4 -mode independent4 -alias-lanes 8 \
    -array-bytes 8192 -repeats 64 -workgroups 1
done
```

This third default-off candidate assigns a returned wide load
`ceil(2*A*(w-1)/w)` work units, where `A` is the active-lane count and `w` is
the uniform dword width. A full wave therefore contributes 0/64/86/96 units
for dword/x2/x3/x4. It is a phenomenological calibration hypothesis, not an
AMD architectural claim. Run it at G1/G7/G16/G28 in all four dependency modes:
a useful queue topology must preserve the G1 width fit, leave dword traffic
bit-identical, create the independent-load occupancy knee, and avoid creating
a serial-load knee. Never combine it with either older return-model flag.

Sweep cross-wave concurrency independently from the work budget:

```bash
for concurrent_waves in 1 2 3 4; do
  /home/fedora/.local/go/bin/go run ./amd/samples/vmem_load_shape \
    -timing -arch gcn5 -gpu gfx90c -disable-rtm -verify \
    -vmem-cu-wide-return-units-per-cycle 1 \
    -vmem-cu-wide-return-concurrent-waves "$concurrent_waves" \
    -width-dwords 4 -mode independent4 -alias-lanes 8 \
    -array-bytes 8192 -repeats 64 -workgroups 28
done
```

Repeat every P at G1/G7/G16/G28 for all four modes. P is a topology probe, not
a continuous fitting knob. Reject the whole static-P hypothesis if the P
needed to keep serial flat removes the independent knee, or if the P that
creates an independent knee also slows serial.

If static P fails, test dynamic burst classification without using benchmark
names or work-group thresholds:

```bash
for window in 2 4 8; do
  for burst_slots in 1 2; do
    /home/fedora/.local/go/bin/go run ./amd/samples/vmem_load_shape \
      -timing -arch gcn5 -gpu gfx90c -disable-rtm -verify \
      -vmem-cu-wide-return-units-per-cycle 1 \
      -vmem-cu-wide-return-burst-concurrent-waves "$burst_slots" \
      -width-dwords 4 -mode "independent${window}" -alias-lanes 8 \
      -array-bytes 8192 -repeats 64 -workgroups 28
  done
done
```

The burst candidate counts only issued, unretired modeled-wide loads. A wave
with one such load receives independent service; a wave with two or more
competes for Q burst slots. Dword traffic is excluded. Q=1 and Q=2 must bracket
the window2/4/8 occupancy surface before testing an intermittent second grant:

```bash
for assist_interval in 2 3 4 6; do
  /home/fedora/.local/go/bin/go run ./amd/samples/vmem_load_shape \
    -timing -arch gcn5 -gpu gfx90c -disable-rtm -verify \
    -vmem-cu-wide-return-units-per-cycle 1 \
    -vmem-cu-wide-return-burst-concurrent-waves 1 \
    -vmem-cu-wide-return-burst-assist-interval "$assist_interval" \
    -width-dwords 4 -mode independent4 -alias-lanes 8 \
    -array-bytes 8192 -repeats 64 -workgroups 28
done
```

The assist cadence advances only on ticks with at least two ready burst waves;
interval 1 is exactly Q=2 and zero disables it. Reject this refinement if one
interval cannot fit both occupancy points and the full window surface, or if
serial, dword, G1, or G7 controls move. Do not select an interval from the
independent4 curve alone.

#### Guarded dependency-window hardware sweep

Use the checked-in driver for the missing independent2/8 G16/G28 controls. It
runs matched zero-trip and R64 points serially with warm-up 20. It uses 10,000
timed iterations for R0 and 6,500 for R64 so the prior guarded timings predict
75--150 ms windows, plus 5 ms telemetry, at least 10 active samples, a 50 C start limit,
a 58 C abort limit, and at least 30 seconds between batches. It also rejects
any concurrent `Vgfx9_compute_unit_tb`, `verilator_bin`, `pytest`, or
`miaow_gcn4` command. Telemetry sampling is independently paced, so a slow
`/proc` guard scan cannot reduce the requested sample density; policy, thermal,
and telemetry failures still abort the container directly. The harness emits
one `CLOCK_MONOTONIC` start/end marker around the primary timed iteration loop;
the collector requires that marker and applies the 1600 MHz/sample-count check
only to its bounded interval. This excludes HIP initialization and verification
activity without weakening whole-run thermal, policy, or process guards. The
default is a one-batch directional plan and does not touch hardware or create
directories:

```bash
python3 gpu_perf_scripts/calibration/gfx90c/run_vmem_dependency_sweep.py
```

Review the eight printed commands, ensure the required `high` policy and
1600 MHz DPM selection are already in force, then opt in explicitly:

```bash
python3 gpu_perf_scripts/calibration/gfx90c/run_vmem_dependency_sweep.py \
  --execute --output-root /tmp/gfx90c-vmem-window-pilot-<unique-id>
```

Only after the directional surface is coherent, collect nine batches per
point with a different, nonexistent output root:

```bash
python3 gpu_perf_scripts/calibration/gfx90c/run_vmem_dependency_sweep.py \
  --execute --production \
  --output-root /tmp/gfx90c-vmem-window-production-<unique-id>
```

The driver stops on the first rejected collector invocation. It never kills a
conflicting process. Every point retains the collector's `manifest.json`, raw
stdout/stderr, telemetry, and summary in its own directory; the parent
`sweep_manifest.json` records progress and the first failed point. Never merge
a partial run with a later run, and never promote the one-batch pilot as a
production measurement. The driver also waits 30 seconds after each successful
point; this is required because each point uses a fresh collector process, so
the collector's own per-batch cooldown cannot span point boundaries.

#### Zero-trip baseline and repeat slope

An explicit repeat count of zero is a matched launch control. It uses the same
checked-in HSACO, selected kernel symbol, kernarg ABI, register allocation,
work-group geometry, prologue, epilogue, and final output store as the measured
point, but the initial scalar branch skips every global load. The harness fully
checks that every lane stores exact zero, and `verify_hsaco.sh` checks that the
branch target is after all loads and before the output store. This is a
*zero-trip control*, not a generic empty kernel: retaining the target's
non-load instructions is intentional.

```bash
# Hardware, using the same warm-up/iteration protocol as the nonzero point.
gpu_perf_scripts/calibration/gfx90c/build_and_run.sh \
  --only vmemloadshape --vmem-width-dwords 4 \
  --vmem-mode serial --vmem-alias-lanes 8 \
  --vmem-array-bytes 8192 --vmem-repeats 0 \
  --vmem-workgroups 1 --warmup 20 --iters 1000

# Simulator. The explicit `-repeats 0` is distinct from omitting the flag,
# which retains the default of 1024 loads per lane.
go run ./amd/samples/vmem_load_shape \
  -timing -arch gcn5 -gpu gfx90c -disable-rtm -verify \
  -width-dwords 4 -mode serial -alias-lanes 8 \
  -array-bytes 8192 -repeats 0 -workgroups 1
```

Measure a separate zero-trip baseline for every dependency symbol and
work-group count. Keep width, aliasing, footprint, work-groups, warm-ups,
iteration count, launch order, clock, and thermal limits identical to the
corresponding nonzero point. The baseline may depend on symbol metadata and
geometry; it is not one universal number for the suite.

For each platform `x`, report both the raw time and

```text
B_x(mode, workgroups) = median T_x(repeats=0)
D_x(repeats) = T_x(repeats) - B_x
```

Only interpret `D_hw / D_sim` when the difference is well above the combined
bootstrap uncertainty of the two measurements. In particular, do not divide a
small launch-dominated point by a noisy baseline. Preserve raw time as the
absolute-model metric; subtraction is a diagnostic decomposition.

The primary body-cost comparison is a robust slope from an equal-repeat sweep,
for example `repeats = 0, 4, 16, 64, 256, 512, 1024` (all values are valid for
both modes):

```text
T_x(repeats) = intercept_x + beta_x * repeats
slope ratio = beta_hw / beta_sim
```

Fit only a visibly linear regime and bootstrap the slope ratio. Compare serial
minus independent at the same repeat count as a paired dependency-overlap
contrast. Compare 8 KiB and 64 KiB at the *same* repeat count before assigning
a difference to the cache path; the equal-footprint-lap recipe below changes
repeat count with footprint and therefore cannot isolate capacity by itself.

Finally, align queue phase for raw absolute comparisons. The hardware helper
after warm-up measures a long batch of steady queued launches, whereas the
ordinary simulator sample contains one dispatcher-first launch. A zero-trip
subtraction within each protocol can diagnose body cost, but it does not make
those two raw totals equivalent. Report cold single-shot and steady queued
results separately until the simulator records a matched multi-launch batch.

For equal footprint coverage, calculate one lap as:

```text
vector_elements = array_bytes / (4 * width_dwords)
address_groups = 64 / alias_lanes
repeats_per_lap = vector_elements / address_groups
```

Use the same lap count at every width/fanout point, rounding repeats upward to
a multiple of the selected independent window. The minimal discriminating
matrix is:

- widths 1, 2, 4 dwords at fanout 8;
- fanouts 1 and 8 at dwordx4;
- all four dependency modes;
- both 8 KiB and 64 KiB footprints;
- one work-group for the cache comparison.

Then hold dwordx4, fanout 8, footprint, and repeat count fixed while sweeping
`1, 7, 16, 28` work-groups. This separates a per-wave load-return discrepancy
from sparse-grid/CU-fill behavior. Report raw launch time, the provided
`ns_per_wave_load` normalization, cache transactions, hit rates, and CPI stack.
Here `repeats` counts load instructions per lane: each independent loop trip
contains 2, 4, or 8 repeats/loads, and the normalization uses that total load
count. Do not interpret the normalization as latency once multiple waves
overlap.

The independent endpoint is deliberately a bound, not a transcription of the
production loop. The authoritative matrix HSACO serializes its A loads and
mixes serialized B loads with an initial two-load window. Use `independent2`
to anchor that window and the 4/8 points to test whether additional overlap
saturates before fitting a model term.
Always rerun `verify_hsaco.sh`: it checks width-specific opcodes, static load
counts, load-before-drain ordering, wave64 metadata, and absence of spills.

For store locality, avoid mapping one 16-byte lane store to each distant line:
that creates partial-line transactions and measures read-modify-write or
write-combine cost as well as stride. The checked-in gfx90c probe groups four
adjacent lanes into each complete 64-byte line and varies only the distance
between the 16 full lines emitted by one wave instruction:

```bash
/home/fedora/.local/go/bin/go run ./amd/samples/store_stride \
  -timing -arch gcn5 -gpu gfx90c -disable-rtm -verify \
  -stride-lines 64 -allocation-stride-lines 128 -repeats 32 \
  -report-cache-hit-rate

ALLOW_UNPINNED_CLOCK=1 \
  gpu_perf_scripts/calibration/gfx90c/build_and_run.sh \
  --only storestride --store-stride-lines 64 \
  --store-allocation-stride-lines 128 --store-repeats 32
```

Sweep line distances `1, 2, 4, 8, 16, 32, 63, 64, 65, 128` while keeping the
allocation stride at 128. This holds allocation size and initialization policy
constant across the threshold. Use unpinned hardware results only for
diagnostic curve shape; fit the timing model only after repeating the sweep at
a pinned clock.

Then run a stream-length/footprint sweep at strides `1`, `2`, and `64`, using
repeat counts `1, 2, 4, 8, 16, 32, 64`. Keep the allocation stride at 128 and
use one work-group. Each repeat adds one wave-wide store instruction, 1024
useful bytes, and 122,944 allocated bytes. This is not a pure loop-latency
test: dynamic instruction count, touched footprint, and allocation size all
grow together. In particular, check whether initialization or DMA behavior
changes when the allocation crosses an implementation threshold.

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

The checked-in gfx90c `scratch_spill` probe uses the corrected production
matrix-multiply body for both variants. `control4` compiles with private size
0, no spills, and no MUBUF traffic; `scratch4` matches production metadata with
private size 20 and four VGPR spills, and disassembles to four
`buffer_store_dword` plus four `buffer_load_dword` instructions at offsets
0/4/8/12. Run the exact pair with:

```bash
/home/fedora/.local/go/bin/go run ./amd/samples/scratch_spill \
  -timing -arch gcn5 -gpu gfx90c -disable-rtm -verify \
  -variant scratch4 -workgroups 16 \
  -report-cache-hit-rate -report-dram-transaction-count

ALLOW_UNPINNED_CLOCK=1 \
  gpu_perf_scripts/calibration/gfx90c/build_and_run.sh \
  --only scratchspill --scratch-variant scratch4 \
  --scratch-workgroups 16 --warmup 20 --iters 1000
```

Repeat both commands with `control4`, and sweep work-group counts
`1, 4, 7, 14, 16, 28, 56`. Sixteen matches corrected N=128 matmul; multiples
of seven expose one, two, four, and eight waves per physical CU. Randomize the
hardware variant order and use `(T_scratch4 - T_control4) / workgroups` as the
paired concurrency curve. This delta includes the compiler's production-like
VGPR/occupancy consequence as well as scratch requests; it is not a literal
single-instruction latency. Fit cycle parameters only from pinned-clock data.

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

### Single-shot and power-state controls

A warmed throughput sweep and a canonical one-launch measurement answer
different questions. When the reference is a one-launch test, add a matching
single-shot control rather than silently replacing it with a warmed batch:

- run at least ten deterministic randomized rounds over the tested sizes;
- use a fresh process/context when that matches the reference protocol;
- select `--warmup 0 --iters 1` and persist the exact round order;
- record every return code, temperature, policy, and clock sample immediately
  before and after the run;
- report median, MAD, CV, and size-normalized medians by order position.

A pre/post `pp_dpm_sclk` sample does not establish the clock used while the
kernel executed. If the size curve is flat or non-monotonic, launch phase,
initialization/cache state, grid thresholds, noise, or an in-kernel power
transient may dominate; the curve alone does not identify which one. Do not
fit timing-model throughput to it. Repeat with a pinned clock and a warmed
protocol, or collect continuous telemetry/device cycles. Also call the
protocol *single-shot*, not automatically cache-cold; GPU memset or
host-to-device initialization before timing may warm the cache.

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
  hardware and simulator slopes quantitatively only with a pinned clock or
  continuous evidence that the in-kernel clock remained stable. With only
  pre/post clock samples, report normalized hardware curve shape as diagnostic
  evidence and do not use its slope to tune cycle parameters.
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
