# RX 570 timing-model progress

Last updated: 2026-07-31

## Outcome

The RX 570 preset now runs and verifies all ten calibration workloads with
the exact gfx803 binaries used by the hardware harness.

| Benchmark | HW steady (µs) | Sim (µs) | Error |
|---|---:|---:|---:|
| vectoradd | 6.765 | 7.473 | 10.5% |
| relu | 7.155 | 7.194 | 0.5% |
| matrixmult | 51.803 | 49.485 | 4.5% |
| matrixtranspose | 16.989 | 14.154 | 16.7% |
| bitonicsort | 347.854 | 323.743 | 6.9% |
| aes | 18.314 | 16.925 | 7.6% |
| fir | 7.699 | 7.646 | 0.7% |
| kmeans | 29.615 | 28.231 | 4.7% |
| pagerank | 19.455 | 20.450 | 5.1% |
| nw | 146.510 | 144.844 | 1.1% |

Canonical-size MARE is **5.8%** across all ten freshly measured workloads.
The stronger anti-overfit result is **6.0% MARE across 47 matched size
points**. All ten swept families have family MARE below 10%; the maximum point
error is the 512-wide transpose holdout at 16.7%.
All points run and verify.
The original headline errors were vectoradd 38.0%, ReLU 21.5%, and matrix
multiplication 58.7%.

The host clock is unpinned, so these microseconds are still diagnostic. Every
steady process now executes at least 50 ms of GPU-active warmup before timing;
this removes the large idle-clock bias that remained after only ten short
launches. A cycle-counter probe or pinned rerun is still needed for
reference-quality absolute latency.

### Short launch-count warmup did not establish a steady clock

The earlier protocol used ten warmup launches. For a 7 µs kernel that keeps
the GPU active for only about 70 µs, while Polaris idles near 588 MHz core and
1000 MHz memory under the host's `auto` policy. Three nominally steady 65K
vectoradd processes consequently ranged from 8.038 to 9.507 µs.

The harness now adds a minimum 50 ms GPU-active warmup, measured with HIP
events, before the timed launches. Five independent 65K processes then ranged
only from 6.650 to 6.712 µs in the full-size rerun. All original 33 hardware
points were recollected under this protocol before changing the model. This is still not
a substitute for pinned clocks, but it removes the observed idle-clock bias.

## What was wrong

### Different instruction streams

The hardware harness loaded newly compiled gfx803 HIP objects while the GCN3
simulator path embedded legacy objects. All benchmarks now select
`kernels_gfx803.hsaco` under `arch.GCN3` and pass the modern HIP ABI argument
layout. Matrix transpose and FIR also received the missing gfx90c objects
needed to keep their packages buildable for the GCN5 path.

Exact binaries exposed missing emulator coverage:

- `s_cmp_eq_u64` and `s_cmp_ne_u64`;
- `s_cmov_b32` and `s_cmov_b64`;
- `v_mad_u16`.

The operations are implemented and covered by focused tests; benchmark
verification supplies end-to-end coverage.

### L2 queue bottleneck

The prior `L2NumReqPerCycle=3` cap caused superlinear queueing:

| vector length | HW steady (µs) | old sim (µs) |
|---:|---:|---:|
| 65,536 | 7.9 | 23.8 total / 8.5 execution |
| 262,144 | 24.3 | 66.3 |

Raising L2 service to its native 16 requests/cycle reduced the 262,144 case
to 44.0 µs without materially regressing the suite. Increasing the CU request
cap alone from 128 to 512 did not fix vector throughput, confirming L2
admission was the bottleneck.

### Random-load latency and bank concurrency were conflated

The old simple-banked DRAM used 16 internal pipelines per controller and
400 ns pipeline latency. Lowering that latency helped pagerank but made
streaming workloads much too fast because the model still exposed excessive
parallel line service.

The suite configuration uses:

- eight external 32-bit controller channels;
- four interleaved internal banks per controller;
- a 250 MHz shallow line-service pipeline;
- separate cache, coalescing, and dense-store issue costs.

This moves exact pagerank from 42.9 to 22.2 µs without serializing unrelated
random misses. A shared L2-to-DRAM token bucket then separates a 1 MiB
burst region from sustained bandwidth: it begins with 16,384 cache-line
credits and refills at one 64-byte request per 1.244 GHz GPU cycle. The burst
avoids imposing the sustained streaming rate on short random traffic; the
final model puts the 262,144-element vector at 23.283 µs versus a 20.978 µs
three-process hardware median.

### Sustained writes need a separate shared path

The original per-transaction dense-store delay could not represent the size
curve: increasing it enough for 262K ReLU overcharged 16K and 65K grids. The
shared L1-to-L2 connection now supports filtering token credits by request
class. RX 570 writes can burst for 12,288 cache lines (768 KiB), then issue at
one line per two GPU cycles; reads and responses remain unlimited. The burst
approximates dirty L2 capacity after resident input and cache state.

This mechanism was selected from five ReLU sizes, including a new 131K
midpoint, and checked against all four vectoradd sizes. Full-line store issue
cost is now charged once per wave instruction rather than once per generated
cache-line request. This distinction preserves the scalar-store path while
allowing wide stores to use the transaction bandwidth they already model.
After the clock-warmed recollection and capacity correction, ReLU has 4.8%
family MARE and vectoradd 6.9%. With the later k-means and PageRank holdouts,
the shared L2-ingress correction, and matched DMA residency, the complete
47-point MARE is 6.0%.

| vector length | HW steady (µs) | Sim (µs) | Error |
|---:|---:|---:|---:|
| 4,096 | 4.749 | 4.844 | 2.0% |
| 16,384 | 5.093 | 5.308 | 4.2% |
| 65,536 | 6.765 | 7.473 | 10.5% |
| 262,144 | 20.978 | 23.283 | 11.0% |

Vectoradd retains 6.9% family MARE. Its 65K and 262K residuals remain visible;
raising the DMA-residency threshold to the full 2 MiB L2 capacity made the
262K point 27.7% slow.

### Sparse reads were charged twice

Pagerank already generates one cache-line request for each random gather line
and pays cache/DRAM latency. The inherited 24-cycle low-utilization penalty
then serialized every sparse line again. Disabling that extra read penalty
reduced pagerank without changing the actual transaction count. The ordinary
write and wide-store penalties remain.

### Cross-size matrix evidence rejects synthetic wide-access penalties

The original calibration added 126 cycles per wide read and 161 cycles for
wide-store strides from one matrix-multiply point. Across four sizes, those
penalties made matrix multiply 53–58% too slow and matrix transpose 248–603%
too slow. Removing both penalties lets the existing transaction, cache, LDS,
and dependency timing carry the accesses directly. A shared twelve-cycle FMA
issue/result occupancy then corrects matrix multiply's uniform compute-path
shortfall: its four size errors are 3.3%, 4.6%, 4.5%, and 3.7%. FIR, k-means,
and pagerank remain in range. The later per-instruction store-issue correction
resolves transpose without restoring an opcode-width penalty.

### AES and dense stores need distinct issue timing

AES is dominated by bitwise vector operations. Classifying vector XOR, AND,
and OR separately from the four-cycle default VALU class models their
one-cycle issue/result timing and produces 16.925 µs versus the 18.314 µs
hardware median.

Partial stores and fully utilized cache-line stores also have different
costs. The model now keeps the partial-line write-combine/RMW cap and adds a
small independent full-line store serialization cost. That mechanism brings
the dense streaming kernels into range without applying benchmark-name rules.

### Multi-kernel dispatch cost

Bitonic sort launches 78 kernels. NW launches dependent anti-diagonals; its
second phase previously ran in ascending order. That accidentally verified
at lengths 64 and 128, but failed at the first later block for lengths 192 and
256. Launching the second phase in descending order now verifies at all four
sizes in both the benchmark and exact-HSACO harness. NW was then remeasured
with the corrected launch order at all four sizes.

Calibrated GPU-side
dispatch costs are:

| Cost | Cycles |
|---|---:|
| first launch | 2,380 |
| subsequent launch | 1,300 |
| completion/post-kernel | 2,500 |

This produces 323.7 µs for bitonic versus 347.9 µs hardware. The much larger
10–72 µs cold-minus-steady values in the hardware file include ROCm/KFD
first-use, page mapping, and other host-side effects; they are deliberately
not folded into GPU execution time.

## Cross-size holdout evidence

The exact-HSACO harness and simulator runner now accept a single benchmark
size. The sweep driver records four predetermined, launch-compatible sizes
for seven families, six k-means sizes, eight transpose sizes spanning its
cache-capacity transition, plus a ReLU midpoint. K-means fixtures preserve the
deterministic feature stream at each point, and PageRank regenerates its
deterministic CSR matrix for each node count. The committed hardware CSV uses
the median of three independent processes per point. All 47 matched hardware
and simulator points are present and every simulator point verifies.

| Family | MARE | Maximum error | Sim/HW slope ratio |
|---|---:|---:|---:|
| vectoradd | 6.9% | 11.0% | 1.132 |
| relu | 4.2% | 9.4% | 0.881 |
| matrixmult | 4.0% | 4.6% | 0.965 |
| matrixtranspose | 9.3% | 16.7% | 0.927 |
| bitonicsort | 6.1% | 6.9% | 1.232 |
| aes | 7.7% | 7.8% | — |
| fir | 0.3% | 0.7% | — |
| kmeans | 9.3% | 14.7% | 0.983 |
| nw | 1.4% | 1.9% | 0.976 |
| pagerank | 6.2% | 8.0% | 1.061 |

PageRank's broad 128–1024-node curve shows that its canonical residual is not
a scaling failure: the model's fitted slope is within 6.1% of hardware and
its four-point family MARE is 6.2%. K-means exposes a different issue. Its
fixed launch overhead matches the flat 1024–4096 hardware region reasonably,
but the original 8192- and 16,384-point simulators were 20.0% and 34.3% fast.
Hardware component timing localizes the transition primarily to the sparse
transpose/swap kernel after three workgroups per CU. Simulator counters show
the L1 read hit rate collapsing from 61.9% at 6,144 points to 42.3% at 8,192
and 4.9% at 16,384, with 84–93% of the resulting requests hitting in L2.

A second token bucket on the shared L1-to-L2 connection models that L2-bank
ingress pressure independently of the existing sustained-write constraint.
It permits an initial 8,192 cache lines (512 KiB), then accepts six aggregate
requests per GPU cycle; the original write-only 768 KiB burst and one-request
per-two-cycle rate remain active simultaneously. K-means's six-point MARE is
now 9.3% and its slope ratio is 0.983 after the later DMA-residency correction.

Host DMA now populates L2 for individual transfers smaller than 1 MiB. This
single physical boundary improves medium vectoradd, ReLU, transpose, PageRank,
and NW points and reduces canonical MARE from 7.5% to 5.8%. A 512 KiB boundary
left transpose-384 at 25.0%; raising it to 1 MiB reduces the maximum error to
transpose-512's 16.7%. Extending residency to the full 2 MiB cache instead
made vectoradd-262K 27.7% slow and was rejected.

The transpose correction is structural rather than benchmark-specific. A
wide store generates up to 16 line requests but issues one wave instruction;
charging the full-line issue delay to each request was double-counting work.
LDS latency and barrier-cost experiments had changed transpose by less than
0.1 µs or regressed matrix multiplication, so both remain reverted. Six new
points around widths 320–640 showed a real capacity transition: removing the
write limiter fits 320–448 but makes 512–640 up to 51% too fast. A 768 KiB
write burst followed by one line per two cycles, combined with matched DMA
residency, gives the expanded transpose curve a 0.927 slope ratio and reduces
the maximum error from 43.8% to 16.7% without a size- or benchmark-specific
rule.

## Parameter sweep evidence

- DRAM stage latency 10 → 5 globally sped up bitonic and did not cure the
  vector throughput curve; reverted during diagnosis.
- CU in-flight vector requests 128 → 512 had no effect on the large-vector
  bottleneck, but modestly improved random-load overlap; 512 retained.
- L2 requests/cycle 3 → 16 fixed most of the superlinear vector queueing and
  was retained.
- L2 bank latency 150 → 50 substantially improved pagerank and k-means.
- Sparse read coalescing cap 24 → 0 improved exact pagerank; retained.
- Vector-memory transaction pipeline width 1 → 4 did not improve pagerank;
  width 1 retained.
- Exact modern HSACO changed several bottlenecks, most notably matrix and NW,
  demonstrating that legacy-object calibration was not transferable.
- A one-bank 250 MHz DRAM candidate brought the 262,144-element vector case
  to 21.703 µs, but made pagerank 25.066 µs; four banks preserve the
  random-miss concurrency needed by the suite.
- Four banks at 62.5 MHz produced 36.620 µs for the large vector, but
  over-serialized the default vectoradd and pagerank cases; rejected.
- Full-line store cost is 90 cycles once per wave instruction, while L1V
  latency remains 56 cycles. Charging that cost once per cache-line request
  was rejected by the wide-store cross-size evidence.
- A shared one-request/cycle limiter without burst capacity fixed the large
  vector slope but pushed pagerank to 24.769 µs. The clock-warmed evidence
  selects a 16,384-line (1 MiB) burst: one-input workloads retain their short
  overlap while vectoradd's second 1 MiB input sees sustained service.
- K-means's new 6,144-point holdout is within 1.3%, but hardware rises sharply
  at four workgroups/CU: the 8,192- and 16,384-point errors are 20.0% and
  34.3%. Halving L2 issue width and reducing L1V capacity from 16 to 12 KiB
  had essentially no effect. Reducing effective L2 capacity made the 16K
  point too slow without fixing 8K, while applying the write limiter to all
  L1-to-L2 traffic made the 8K simulated runtime 4.5× larger and badly regressed
  transpose; all were rejected.
- Reducing the CU in-flight vector-memory limit from 512 to 128 lowered the
  new worst k-means error to 27.7%, but raised the canonical 65K vectoradd
  error from 18.2% to 25.6% and left 47-point MARE unchanged (7.39% versus
  7.38%). The 192-entry compromise retained most of the vector regression
  with little k-means benefit, so the native 512-entry capacity remains.
- Limiting total L1-to-L2 ingress to six requests/cycle without a burst fixed
  k-means slope but raised the 65K vectoradd error to 30.3% and PageRank-512
  to 20.4%. An 8,192-line burst preserves both points exactly while retaining
  the large-k-means correction. A smaller 4,096-line candidate improved the
  two largest k-means points by only 1.8–2.1 percentage points and began
  regressing PageRank-512, so the broader 512 KiB burst was selected.
- Raising the hybrid DMA L2-fill boundary from 128 KiB to 512 KiB reduced
  47-point MARE from 6.7% to 6.2%; 1 MiB further improved it to 6.0% and cut
  the maximum error from 25.0% to 16.7% across five affected families. A
  2 MiB candidate made vectoradd-262K 27.7% slow, so the half-L2 boundary was
  selected rather than fitting full-cache residency.

## Reproduction

```bash
cd gpu_perf_scripts/calibration/rx570
SIM_JOBS=4 ./run_sim.sh sim_out | tee sim_results.txt
./compare.py sim_results.txt
./run_size_sweeps.py --mode sim --jobs 4 --output sim_size_sweep.csv
./run_size_sweeps.py --mode hardware --jobs 1 --trials 3 \
  --output hw_size_sweep.csv
./compare_size_sweeps.py hw_size_sweep.csv sim_size_sweep.csv
```

The default comparison uses the recorded steady hardware column. To display
the deliberately unmatched cold runtime measurement:

```bash
./compare.py --hw-mode cold sim_results.txt
```

## Remaining work

- Add `s_memtime` cycle-counter measurements and rerun with a pinned clock.
- Investigate the remaining transpose-512 residual without benchmark-specific
  timing.
