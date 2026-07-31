# gfx90c microbenchmark results

This file records the targeted measurements used after the initial ISCA-10
calibration. It includes negative results so that later work does not repeat
knob sweeps that have already been falsified.

## Measurement status

- GPU: Renoir integrated Radeon, gfx90c, 7 active CUs.
- Reference application results in `hw_ground_truth.txt` were measured with
  the performance policy pinned to `high` (1600 MHz).
- Those legacy targets used warmed, many-iteration native HIP kernels and
  predate the exact-HSACO single-shot harness. They remain the accepted
  calibration table, but are not fully matched validation references.
- The diagnostic hardware measurements below were collected with the policy at
  `auto`. During the long pointer chase, repeated reads of
  `pp_dpm_sclk` showed `200 MHz *` for the entire kernel.
- Therefore the microbenchmark hardware numbers are diagnostic curves and
  component ratios. They are not replacements for the pinned application
  ground truth.
- Runs were sequential and the GPU edge temperature remained 44-47 C.

## Verified application size sweeps

These sweeps change one input axis at a time while preserving the exact kernel
binary and launch rules. Simulator outputs were verified. Hardware used the
`auto` clock policy, so only within-hardware curve shape and growth are
interpretable.

### K-means feature and point sweeps

The feature sweep holds points at 4,096 and clusters at 5:

| Features | Sim (us) | Sim growth | HW auto (us) | HW growth |
|---------:|---------:|-----------:|-------------:|----------:|
| 4        | 16.914   | -          | 107.590      | -         |
| 8        | 36.213   | 2.14x      | 189.685      | 1.76x     |
| 16       | 40.439   | 1.12x      | 372.376      | 1.96x     |
| 32       | 90.147   | 2.23x      | 740.660      | 1.99x     |

The point sweep holds features at 16 and clusters at 5:

| Points | Sim (us) | Sim growth | HW auto (us) | HW growth |
|-------:|---------:|-----------:|-------------:|----------:|
| 1,024  | 21.975   | -          | 204.677      | -         |
| 2,048  | 29.538   | 1.34x      | 275.841      | 1.35x     |
| 4,096  | 40.439   | 1.37x      | 372.376      | 1.35x     |
| 8,192  | 82.660   | 2.04x      | 626.903      | 1.68x     |

The default pinned 1600 MHz target is 39.220 us at 4,096 points, 16 features,
and 5 clusters. The 372.376 us `auto` result at that same point demonstrates
why the diagnostic hardware column is unusable for absolute calibration.
Non-default sizes have no pinned reference. The useful observations are curve
shape:

- Hardware feature growth is close to 2x from 8 through 32 features. The
  simulator has a plateau from 8 to 16 followed by a larger 16-to-32 step.
- Fixed launch and setup cost is visible at the smaller point counts. The
  simulator enters an approximately linear large-input regime more sharply
  between 4,096 and 8,192 points than the diagnostic hardware curve.
- A one-point timing fit would hide both effects. Feature count and point count
  must remain independent calibration axes.

### ReLU and AES size controls

The ReLU simulator sweep was collected with verification enabled at commit
`93b490af`, and AES at `65f6178f`. Both use the same timing settings restored
by `17117fab` after an unsuccessful phase-sensitivity experiment, so these
times describe the current timing configuration.

| ReLU elements | Work-groups | Sim (us) | Sim growth |
|--------------:|------------:|---------:|-----------:|
| 16,384        | 256         | 3.100    | -          |
| 32,768        | 512         | 8.534    | 2.753x     |
| 65,536        | 1,024       | 14.572   | 1.708x     |
| 131,072       | 2,048       | 27.250   | 1.870x     |

The last three points admit the empirical large-grid fit
`T = 2.195 + 0.000190826 N` us with `R^2 = 0.99986`. The 16,384-element point
deviates; without counters its cause is unestablished, so it is excluded from
that fit.

| AES bytes | Work-groups | Sim (us) | Sim growth |
|----------:|------------:|---------:|-----------:|
| 1,024     | 1           | 15.447   | -          |
| 2,048     | 2           | 15.455   | 1.001x     |
| 4,096     | 4           | 15.476   | 1.001x     |
| 8,192     | 8           | 16.733   | 1.081x     |

The 1-4 KiB AES points are almost entirely on the modeled first-dispatch
floor. They constrain a small-grid intercept, not AES throughput. The 8 KiB
point is the first tested size with a clear incremental-work contribution.

Randomized unpinned hardware controls tested whether grouped execution order
caused the unusual hardware curves. Each benchmark used 12 deterministic
shuffled rounds (seed `20260731`), one fresh serialized process per size, and
`--warmup 0 --iters 1`. This means single-shot/no prior target launch, not
necessarily cache-cold: the harness initializes device memory before timing.

| ReLU elements | HW auto median (us) | CV | MAD (us) |
|--------------:|--------------------:|---:|---------:|
| 16,384        | 443.054             | 2.13% | 3.505 |
| 32,768        | 98.010              | 9.61% | 3.431 |
| 65,536        | 115.993             | 11.35% | 7.364 |
| 131,072       | 155.443             | 56.29% | 0.862 |

The large 131,072-element CV comes from one 524.640 us outlier; its small MAD
shows why both robust spread and CV are reported. Randomization rules out
grouped order as the sole cause of the 16,384-element anomaly: that size stayed
in the 420-458 us range in every order position, while 32,768 elements stayed
in the 90-125 us range.

| AES bytes | HW auto median (us) | CV | MAD (us) |
|----------:|--------------------:|---:|---------:|
| 1,024     | 411.644             | 2.86% | 7.138 |
| 2,048     | 394.057             | 4.38% | 3.131 |
| 4,096     | 411.399             | 2.49% | 6.267 |
| 8,192     | 400.629             | 2.85% | 9.188 |

All 96 randomized runs returned zero. Immediately before and after each run,
the policy was `auto` with 200 MHz active; temperatures were 45-47 C for ReLU
and 44-49 C for AES. These snapshots do not observe the in-kernel clock. The
ReLU curve is physically non-monotonic at the small end and the AES medians
are flat and non-monotonic. These unpinned single-shot curves are unsuitable
for throughput fitting and do not identify whether the cause is launch phase,
initialization/cache state, grid thresholds, measurement noise, or an
in-kernel power transient. They must not be compared with the pinned targets.

### Corrected matrix-multiplication sweep

The corrected kernel uses global Y when indexing matrix A, and every output
element passes the strengthened verifier:

| Matrix size | Work-groups | Sim (us) | Sim growth | HW auto (us) | HW growth |
|------------:|------------:|---------:|-----------:|-------------:|----------:|
| 32          | 1           | 14.444   | -          | 142.344      | -         |
| 64          | 4           | 20.036   | 1.39x      | 264.337      | 1.86x     |
| 128         | 16          | 42.229   | 2.11x      | 517.936      | 1.96x     |

The 64-to-128 growth is close once both paths have enough work-groups to
amortize fixed launch cost. The 32-to-64 difference shows why the
one-work-group point must not set a saturated throughput parameter.

The checked-in 39.107 us pinned target for N=128 predates the matrix-A indexing
fix. It is stale for absolute validation of the corrected kernel. A replacement
pinned measurement cannot currently be reacquired without sudo access to clock
control, so no corrected absolute HW/simulator error is reported here.

### Matrix-transpose width sweep

The final committed configuration and a uniform single-shot hardware sweep
produced the following curves. Every hardware point is the median of 15
serialized launches with no prior transpose launch (`--warmup 0 --iters 1`).
The harness performs a GPU `hipMemset` before timing, so its input cache state
is not guaranteed cold and does not match the simulator's default H2D-bypass
state. All 75 runs returned zero; samples immediately before and after every
run reported the diagnostic `auto` policy with 200 MHz active, and temperature
readings were 44-47 C. Absolute hardware times therefore remain unsuitable
for pinned-target calibration.

| Width | Sim (us) | Sim growth | HW auto median (us) | HW growth | HW CV |
|------:|---------:|-----------:|--------------------:|----------:|------:|
| 256   | 35.498   | -          | 115.497             | -         | 15.85% |
| 320   | 54.129   | 1.525x     | 146.225             | 1.266x    | 8.44% |
| 384   | 73.945   | 1.366x     | 166.914             | 1.141x    | 5.37% |
| 448   | 99.317   | 1.343x     | 191.460             | 1.147x    | 13.33% |
| 512   | 129.067  | 1.300x     | 222.970             | 1.165x    | 7.09% |

Both curves are smooth fixed-overhead-plus-area trends. An affine
`T = a + bN^2` fit has `R^2 = 0.9997` for simulation and `R^2 = 0.9890` for
hardware. The earlier apparent width-512 knee came from combining points that
used different warm-up/iteration protocols. The uniform single-shot sweep does
not reproduce or support an architectural knee, so the earlier comparison
must not be used to tune the model.

Counter-enabled default-DMA simulator runs at widths 320 and 384 show no
observable L2 traffic discontinuity: L2 read misses, write misses, and DRAM
transactions grow by approximately the 1.44x area ratio (read misses exactly;
the other counts within 0.1%), while L2 read hits remain 48 at both sizes.
Routing H2D through L2 makes the input warm and improves the two points from
54.129/73.945 us to 37.956/57.847 us, but the curve remains smooth. These
results establish no L2 transition under either tested simulator initialization
policy; they do not establish equivalent hardware cache state. The
experimental DMA change was reverted.

The same default-DMA runs show a modest L1V-TLB miss-fraction increase from
6.72% at width 320 to 7.68% at width 384, while L2-TLB misses per element stay
approximately constant. This is not a sharp modeled TLB wall, but the
application sweep is not a substitute for a dedicated translation-capacity
probe.

The default width-512 point passes its numerical comparison with the legacy
pinned target. However, the 64-line far-stride tier remains a canonical-point
fit rather than a mechanism validated by this application sweep. The dedicated
full-line store-stride probe below exercises the tier directly, but does not
support interpreting it as write combining, DRAM locality, or another
transaction-level effect.

### Full-line store-stride and stream-length sweeps

At commit `f5426c7a`, the exact-HSACO probe used one wave64 work-group, 32
dynamic wave-wide store instructions, and 16 complete 64-byte lines per
instruction. The primary sweep varied actual line spacing while keeping the
allocation stride at 128 lines. This holds the 3.752 MiB allocation and
initialization path constant. Every simulator point passed full output and gap
verification.

Hardware used ten deterministic randomized rounds, one fresh serialized
process per point, and `--warmup 0 --iters 1`. All 100 runs returned zero.
Pre/post samples reported policy `auto`, active 200 MHz, and 45-50 C; those
samples do not observe the in-kernel clock. The hardware values therefore
describe diagnostic curve shape only. `HW paired delta` is the median over
rounds of `(T_stride - T_stride1) / 32`, rather than a difference of medians.

| Stride (lines) | Spacing (bytes) | Sim (us) | Sim delta (ns/repeat) | HW auto median (us) | MAD (us) | CV | HW paired delta (ns/repeat) |
|---------------:|----------------:|---------:|----------------------:|--------------------:|---------:|---:|----------------------------:|
| 1   | 64    | 5.402  | 0.000    | 116.644 | 0.751 | 0.81% | 0.000   |
| 2   | 128   | 75.970 | 2205.250 | 117.371 | 0.902 | 1.28% | 33.656  |
| 4   | 256   | 75.970 | 2205.250 | 119.976 | 1.202 | 1.47% | 108.797 |
| 8   | 512   | 75.970 | 2205.250 | 120.652 | 1.147 | 0.97% | 100.344 |
| 16  | 1024  | 76.078 | 2208.625 | 119.250 | 0.806 | 1.96% | 87.672  |
| 32  | 2048  | 75.970 | 2205.250 | 120.627 | 0.912 | 3.30% | 123.672 |
| 63  | 4032  | 76.078 | 2208.625 | 124.450 | 1.378 | 1.47% | 233.094 |
| 64  | 4096  | 85.060 | 2489.313 | 123.758 | 0.752 | 3.53% | 231.984 |
| 65  | 4160  | 85.060 | 2489.313 | 124.985 | 1.328 | 4.19% | 249.688 |
| 128 | 8192  | 85.060 | 2489.313 | 126.363 | 2.730 | 2.31% | 321.703 |

The simulator produces an exact near-stride plateau at 2-63 lines and an
additional step at 64 lines. The hardware diagnostic curve grows gradually;
63 lines is slower than 64, while 65 and 128 continue the gradual trend. It
does not reproduce a sustained 64-line knee. Because the clock was not pinned,
this result rejects using the current knee as a validated physical mechanism
but is not sufficient to fit replacement cycle penalties.

The simulator metric database reports whole-run totals, including transfers;
they are not kernel-scoped counters. Nevertheless, the primary sweep holds
them exactly invariant, so its modeled timing steps are explicit penalties and
not consequences of additional modeled transactions:

| Strides | L1V writes H/M/MSHR | L2 writes H/M/MSHR | DRAM reads/writes |
|--------:|---------------------:|--------------------:|------------------:|
| all ten | 0/512/0              | 0/518/0             | 61,474/61,990     |

The stream-length/footprint control swept repeats `1, 2, 4, 8, 16, 32, 64`
at strides 1, 2, and 64. Each repeat adds one dynamic wave-store instruction,
1 KiB of useful writes, and 122,944 allocated bytes. Thus both dynamic work
and footprint grow; this is not a pure loop-latency measurement. All 21
simulator points passed verification.

| Repeats | Allocation (MiB) | Useful writes (KiB) | Sim stride 1 (us) | Sim stride 2 (us) | Sim stride 64 (us) |
|--------:|-----------------:|--------------------:|------------------:|------------------:|-------------------:|
| 1  | 0.117 | 1  | 3.910 | 5.910   | 6.281   |
| 2  | 0.234 | 2  | 3.958 | 8.169   | 8.823   |
| 4  | 0.469 | 4  | 4.054 | 12.690  | 13.905  |
| 8  | 0.938 | 8  | 4.247 | 21.730  | 24.070  |
| 16 | 1.876 | 16 | 4.632 | 39.810  | 44.400  |
| 32 | 3.752 | 32 | 5.402 | 75.970  | 85.060  |
| 64 | 7.504 | 64 | 6.943 | 148.290 | 166.380 |

The hardware companion used the same 21 cases in ten deterministic randomized
rounds, again with a fresh process for every point. All 210 runs returned zero;
policy/clock snapshots remained `auto`/200 MHz and temperatures were 44-49 C.
Each hardware cell below is `median / MAD / population CV` in microseconds.

| Repeats | HW stride 1 | HW stride 2 | HW stride 64 |
|--------:|------------:|------------:|-------------:|
| 1  | 91.873 / 7.715 / 17.35% | 91.046 / 9.242 / 17.60% | 77.471 / 3.507 / 10.06% |
| 2  | 60.464 / 9.017 / 15.39% | 66.174 / 16.932 / 21.76% | 64.596 / 6.312 / 12.64% |
| 4  | 53.075 / 1.603 / 14.15% | 53.852 / 0.752 / 4.15%  | 54.979 / 0.326 / 7.74%  |
| 8  | 62.443 / 0.852 / 2.03%  | 63.044 / 0.426 / 1.11%  | 64.847 / 0.527 / 2.82%  |
| 16 | 80.853 / 0.953 / 1.72%  | 81.479 / 0.927 / 2.95%  | 84.759 / 0.577 / 5.86%  |
| 32 | 116.670 / 0.827 / 2.14% | 118.363 / 1.618 / 1.54% | 124.836 / 1.529 / 2.52% |
| 64 | 189.827 / 0.952 / 2.08% | 190.033 / 1.398 / 2.04% | 201.354 / 1.353 / 4.40% |

The one- and two-repeat hardware results are launch-dominated and
non-monotonic. Over 16-64 repeats, the per-round paired robust slope increment
is `+14.0 ns/repeat` for stride 2 versus 1 (MAD 52.1 ns) and
`+156.8 ns/repeat` for stride 64 versus 1 (MAD 72.1 ns, positive in 9/10
rounds). This supports a sustained cost for sufficiently distant lines, but
the primary sweep locates no discrete transition at exactly 64 lines. The
unpinned slopes are diagnostic and must not set simulator cycle parameters.

The corresponding simulator linear fits are:

| Stride | Repeat range | Intercept (us) | Slope (ns/repeat) | R² | Max residual (us) |
|-------:|-------------:|---------------:|------------------:|---:|------------------:|
| 1  | 1-64 | 3.861678 | 48.143723   | 0.999999970 | 0.000277 |
| 2  | 1-64 | 3.649764 | 2260.005114 | 0.999999999952 | 0.000775 |
| 64 | 1-64 | 3.740057 | 2541.248801 | 0.999999999985 | 0.000445 |

Kernel time remains linear despite the footprint crossing 128 KiB between
one and two repeats. Whole-run L2 write-miss totals do change discontinuously
at that boundary (1927 at one repeat versus 38 at two), confirming that the
size control crosses an initialization/DMA reporting regime even though the
isolated driver `kernel_time` fit is unaffected. A pinned-clock hardware
repetition remains necessary before changing the store timing configuration.

### Private-spill work-group sweep

Commit `356443d5` adds an exact-HSACO pair around the corrected production
matrix-multiplication body. `control4` has private segment size 0, zero spills,
75 VGPRs, and no MUBUF instructions. `scratch4` matches production matmul with
private segment size 20, four VGPR spills, 64 VGPRs, and exactly four private
stores plus four private loads at offsets 0/4/8/12. Both use an 8x8 wave64
work-group, identical inputs and output verification, and the same global,
LDS, barrier, VALU, and output work. Every simulator point passed full CPU
reference verification.

| Work-groups | Control (us) | Scratch4 (us) | Delta (us) | Delta (ns/work-group) |
|------------:|-------------:|--------------:|-----------:|----------------------:|
| 1  | 12.167 | 14.820 | 2.653  | 2653.000 |
| 4  | 12.494 | 15.147 | 2.653  | 663.250  |
| 7  | 12.721 | 15.297 | 2.576  | 368.000  |
| 14 | 18.186 | 22.128 | 3.942  | 281.571  |
| 16 | 22.976 | 28.195 | 5.219  | 326.188  |
| 28 | 27.818 | 34.430 | 6.612  | 236.143  |
| 56 | 54.320 | 68.284 | 13.964 | 249.357  |

The fixed delta through seven work-groups shows that the modeled private path
overlaps across available CUs; beyond one wave per CU, total spill cost grows
with concurrency. At the production N=128 grid size of 16 work-groups, the
paired modeled cost is 5.219 us. This is a production-like compiler/resource
delta, not eight isolated instruction latencies: the variants intentionally
have different VGPR allocation and occupancy consequences.

Counter-enabled whole-run totals at one work-group show the intended traffic
change. Control has L1V read H/M `0/128` and write H/M `0/64`; scratch4 has
`59/149` and `0/144`. The four private stores therefore add 80 modeled L1V
write transactions for one wave under the private-address mapping. Transfer
traffic is included in these totals, so the counters support the paired
traffic difference but are not a kernel-only CPI decomposition.

The hardware diagnostic used 20 warm-ups and 1,000 timed iterations per fresh
process, with ten deterministic randomized rounds. All 140 runs returned zero;
every pre/post snapshot reported `auto`/200 MHz and temperatures were 44-50 C.
Each variant cell is `median / MAD / population CV` in microseconds. Paired
delta is the per-round median of `(scratch4-control4)/workgroups`.

| Work-groups | HW control4 | HW scratch4 | Paired delta (ns/work-group) | Delta MAD (ns) | Positive rounds |
|------------:|------------:|------------:|-----------------------------:|---------------:|----------------:|
| 1  | 132.106 / 0.010 / 0.008% | 141.674 / 0.016 / 0.022% | 9576.00  | 23.50  | 10/10 |
| 4  | 132.426 / 0.009 / 0.009% | 141.385 / 0.019 / 0.013% | 2240.00  | 3.75   | 10/10 |
| 7  | 132.646 / 0.004 / 0.006% | 142.118 / 0.014 / 0.014% | 1353.21  | 1.71   | 10/10 |
| 14 | 132.702 / 0.008 / 0.007% | 141.747 / 0.014 / 0.014% | 645.89   | 1.32   | 10/10 |
| 16 | 210.589 / 0.423 / 0.321% | 177.768 / 3.449 / 2.283% | -2070.75 | 210.66 | 0/10  |
| 28 | 170.575 / 2.720 / 3.070% | 180.269 / 3.918 / 2.837% | 274.21   | 142.91 | 9/10  |
| 56 | 239.564 / 0.699 / 0.821% | 248.548 / 2.972 / 1.894% | 169.24   | 61.93  | 8/10  |

The warmed curve is precise at small grids but strongly non-monotonic.
Control4 is flat through 14 work-groups, jumps by 77.887 us at 16, then drops
by 40.014 us at 28 in every round. Scratch4 has a smaller transition, making
the paired delta negative at 16 in all ten rounds. At 28 and 56 the paired
effect is positive but occasionally reverses sign. This falsifies a monotonic
per-work-group scratch penalty under the tested state and shows that the
compiler/resource difference cannot be separated from a grid/occupancy regime
change with this pair alone. The simulator curve is monotonic and misses the
16-work-group control cliff.

The next control should separate resource allocation from memory traffic: add
a no-spill variant at the scratch kernel's 64-VGPR occupancy if the compiler
can produce one, retain the existing 75-VGPR control, and densely sweep grids
around 14-28. A pinned-clock repetition is still required before changing
scratch latency, coalescing, concurrency, or occupancy limits.

## Immutable full-suite comparison

The suite was built and run once from calibration commit `12d5269a`, with
verification enabled and two simulator jobs. The raw artifacts currently
reside in `/tmp/gfx90c-final-suite.Izkqsc`; the table below is their durable
summary. Signed error is `HW/Sim - 1`; the acceptance gate is strict
`abs(error) < 10%`.

Commit `17117fab` restores the same timing configuration after the rejected
dispatch experiment. A fresh exact-commit run is kept separate
from this immutable historical run rather than silently relabeling artifacts.

A fresh suite built from documentation commit `1c42d025` reproduced every
time in the table exactly. All ten overall `.rc` and simulator `.sim.rc` files
were zero and verification passed. Its raw artifacts are in
`/tmp/gfx90c-final-suite-restored.N8rxNo`; the checked-in
`sim_ground_truth.txt` is the durable timing summary. This confirms simulator
reproducibility, while the legacy hardware-provenance limitation below still
applies.

| Benchmark | Sim (us) | Pinned HW (us) | HW/Sim | Error | Gate |
|-----------|---------:|---------------:|-------:|------:|------|
| vectoradd | 27.261 | 29.364 | 1.0771x | +7.71% | Pass |
| relu | 14.572 | 13.123 | 0.9006x | -9.94% | Pass |
| matrixmult | 42.229 | 39.107 | 0.9261x | -7.39% | Not scored (stale) |
| matrixtranspose | 129.067 | 140.773 | 1.0907x | +9.07% | Pass |
| bitonicsort | 745.124 | 811.360 | 1.0889x | +8.89% | Pass |
| aes | 15.476 | 16.962 | 1.0960x | +9.60% | Pass |
| fir | 11.316 | 12.003 | 1.0607x | +6.07% | Pass |
| kmeans | 40.439 | 39.220 | 0.9699x | -3.01% | Pass |
| pagerank | 126.412 | 130.638 | 1.0334x | +3.34% | Pass |
| nw | 134.211 | 123.052 | 0.9169x | -8.31% | Pass |

All nine scored comparisons against the usable legacy pinned targets pass the
strict gate, with a MARE of 7.33%. This is a calibration score, not fully
matched hardware validation: every target still needs a fresh pinned run with
the exact-HSACO single-shot harness. As a non-validation sensitivity check,
including the stale matrixmult number would produce 7.34% MARE and its
numerical comparison is also below 10%.

The matrixmult hardware target predates the global-row indexing fix and is not
a valid absolute reference for the corrected kernel. It remains in the table
only to make the historical comparison explicit. A corrected pinned
measurement is still required. ReLU has only 0.06 percentage points of gate
margin, AES 0.40, and matrixtranspose 0.93; these are passes, not evidence of
robust prediction across sizes or clock states.

The immutable configuration combines these main settings:

| Mechanism | Accepted setting |
|-----------|------------------|
| L2 and DRAM | L2 bank latency 64 cycles; DRAM depth 9, stage latency 14 |
| LDS and barrier | latency 4; issue interval 1; max in flight 4; barrier 4 |
| Read coalescing | maximum penalty 13 |
| Split-line loads | penalty 22, at most 2 dwords |
| Dependent loads | issue penalty 6000, exact issue age 3, at most 2 dwords, FLAT/GLOBAL only |
| Stores | partial-line penalty 103; wide near-stride 240; far-stride 270 at 64 lines |
| Dispatch | first launch 3750 cycles; subsequent launch 7500; completion 1450 |

The core mechanisms are in `ee43afda`, `a8127c48`, `52e90380`, and
`8d88b485`. The full-suite gate does not replace the size-sweep and paired
latency/throughput requirements below.

## Separate latency from throughput

Following the paired-probe principle used in
[recent GPU microbenchmarking work](https://arxiv.org/html/2507.10789v1), an
application size curve identifies fixed-cost, scaling, and knee mismatches but
does not by itself distinguish dependent latency from independent throughput.
The repository-wide procedure is in
[`how_to_microbenchmark_amd.md`](../../../how_to_microbenchmark_amd.md). Use
both:

- a dependent chain with explicit consumption and `s_waitcnt` to expose result
  latency; and
- several independent operations, lanes, waves, or work-groups to expose
  issue rate and sustainable concurrency.

Disassemble the exact HSACO before interpreting either probe. Confirm the
intended opcode, dependency chain, wait placement, unroll count, register
usage, and absence of unintended spills or compiler-eliminated work. A timing
parameter is supported only when it predicts the relevant slope or knee and
survives the paired control and full-suite canaries.

## Historical K-means feature-component sweep

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

The earlier component-level sweep, before calibration commit `12d5269a`, used
4,096 points and 5 clusters:

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
required for independent absolute L2 validation, rather than as prerequisites
for the committed application-suite fit.

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
latency. Their safe conclusion is directional: an unqualified reduction of
general L2/global-load latency to fit matmul is not supported by this probe and
would also worsen PageRank and FIR unless their distinct access mechanisms are
modeled separately.

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
verification. The simulator remained 28.7% slower than hardware, improved from
50.7% slower. With the document's signed convention, `HW/Sim - 1`, the
remaining error at that milestone was -22.3%.

## Matrix-multiplication correctness history

Commit `3fe47656` parameterized the exact-HSACO harness with
`--matrix-size`. The current corrected size results are recorded in the
verified application-size section above.

The benchmark verifier also contained an unrelated loop-variable error that
checked only part of one row. Commit `91808957` now checks every output element
and adds a regression test, ensuring future size/ISA variants cannot produce a
false pass.

The stronger verifier then exposed a source-level bug at the first Y
work-group boundary: `globalPosA` used local Y, causing every Y work-group to
reuse matrix A rows 0-31. The native and OpenCL sources now use global Y and
the gfx90c HSACO was regenerated. The corrected N=128 kernel passes full
verification. Under the model used for that source-only comparison, timing
changed only 45.141 to 45.300 us in simulation and 518.458 to 525.635 us on
the same auto/200 MHz hardware state (+0.35% and +1.38%). Those historical
numbers isolate the source correction; they are not the current model's size
sweep. The old pinned target remains recorded for provenance but is stale for
absolute validation of the corrected kernel.

## Earlier accepted fixes

Three benchmark-driven fixes improved the K-means/matmul state:

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

- matrix multiplication before the later source correction:
  50.728 -> 45.141 us (historical pinned target 39.107);
- K-means: 69.385 -> 50.473 us (HW 39.220).

## Rejected candidates

### Isolated L2 bank latency 128 -> 64 cycles

This historical experiment preceded the uninterrupted K-means sequence and
matrix-index correction. Selected results:

| Benchmark | 128-cycle sim (us) | 64-cycle sim (us) | HW (us) |
|-----------|-------------------:|------------------:|--------:|
| matrixmult | 45.141 | 40.196 | 39.107 |
| kmeans | 59.096 | 50.992 | 39.220 |
| pagerank | 112.948 | 96.415 | 130.638 |
| fir | 8.338 | 7.896 | 12.003 |

Matmul became close and K-means improved, but full-suite MARE worsened from
17.9% to 18.8%. PageRank and FIR, already too fast, became severe outliers.
The isolated candidate was reverted at that milestone. This result rejects a
uniform latency change, not the accepted mechanism-separated model in
`12d5269a`. The immutable configuration uses 64 cycles only together with
access-class filters for split-line and dependent loads plus separate store
stride costs.

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

### Dispatch-overhead phase rebalance

Commit `1d078dd0` tested a first-launch/completion rebalance from 3750/1450 to
4050/1400 cycles. It moved the four-work-group AES default exactly as fixed
cost arithmetic predicts, from 15.476 to 15.633 us, improving its signed error
from +9.60% to +8.50%. Large-grid results were not stable under the same
change: ReLU moved to 14.470 us and vector-add to 26.898 us even though their
net fixed-cost change was only -12 cycles. Matrix transpose regressed from
129.067 to 126.138 us and failed the strict gate at +11.60%.

Code review confirmed that launch and completion costs are each charged once.
The larger movements come from changing the pre-dispatch event phase and its
downstream memory/port ordering, not literal double charging. A smaller
3950/1425-cycle canary produced AES 15.586, ReLU 14.639, vector-add 26.897,
and transpose 129.145 us; ReLU failed at -10.36%. Commit `17117fab` restores
3750/1450. This rejects fixed-overhead tuning as a robust way to improve the
narrow AES/ReLU margins; future work must measure launch behavior separately
and validate adjacent sizes before changing the event phase.

### Remove the 64-line wide-store tier

Commit `24d78517` removed only the 270-cycle far tier while retaining the
240-cycle non-local wide-store cost. This directly tested the store-stride
result: hardware supports a broad far-stride cost at sufficient stream length,
but not a discrete transition at exactly 64 lines.

The probe behaved arithmetically: strides 63, 64, and 65 all became
76.078 us, compared with 76.078, 85.060, and 85.060 us in the accepted
configuration. Correctness verification passed. Targeted application results
were:

| Benchmark/size | Accepted sim (us) | No-far-tier sim (us) | Change |
|----------------|------------------:|----------------------:|-------:|
| transpose W=192 | 25.771 | 25.771 | 0.000 |
| transpose W=256 | 35.498 | 35.498 | 0.000 |
| transpose W=320 | 54.129 | 50.532 | -3.597 |
| transpose W=512 | 129.067 | 118.220 | -10.847 |
| matrixmult N=128 | 42.229 | 42.229 | 0.000 |
| AES 4096 bytes | 15.476 | 15.476 | 0.000 |

At W=512, the legacy transpose comparison regressed from +9.07% to +19.08%
and failed the strict gate. The other nine default suite results were
unchanged. The far tier is therefore synthetic compensation for a missing
transpose cost, not a validated mechanism, but removing it alone is also not
an acceptable calibration. The accepted tier is restored pending a pinned
probe and a replacement mechanism that predicts both the smooth hardware
stride curve and the transpose width curve.

## Remaining validation

The vector and producer/consumer probes rule out broad, uniform full-wave L1V
and L2 latency reductions. The size sweeps reinforce that K-means feature
layout, point-count scaling, and matrix-multiplication saturation are distinct
axes. The immutable default-size suite passes its available target gate, but
the transpose width curve demonstrates why that is not sufficient. Candidate
mechanisms must also survive dependent-latency and independent-throughput
controls plus geometric sweeps around regime boundaries.

The corrected matrix kernel still needs a replacement pinned N=128 hardware
measurement when privileged clock control is available. Until then, its
diagnostic auto-clock curve can validate scaling shape but not absolute error.
The ReLU/AES narrow gate margins, a pinned-clock repetition of the full-line
store-stride probe, and a pinned corrected matrix-multiplication target are the
highest-priority follow-up measurements.
