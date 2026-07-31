# RX 570 timing-model progress

Last updated: 2026-07-31

## Outcome

The RX 570 preset now runs and verifies all ten calibration workloads with
the exact gfx803 binaries used by the hardware harness.

| Benchmark | HW steady (µs) | Sim (µs) | Error |
|---|---:|---:|---:|
| vectoradd | 9.299 | 7.340 | 21.1% |
| relu | 6.997 | 6.375 | 8.9% |
| matrixmult | 52.396 | 49.350 | 5.8% |
| matrixtranspose | 22.579 | 25.402 | 12.5% |
| bitonicsort | 348.697 | 321.321 | 7.9% |
| aes | 18.595 | 16.892 | 9.2% |
| fir | 7.875 | 7.566 | 3.9% |
| kmeans | 30.045 | 27.345 | 9.0% |
| pagerank | 19.176 | 22.334 | 16.5% |
| nw | 146.999 | 139.076 | 5.4% |

Canonical-size MARE is **10.0%** across all ten freshly measured workloads.
The stronger anti-overfit result is **7.0% MARE across 33 matched size
points**. All eight swept families have family MARE below 10%; the maximum
point error is the 65K vectoradd holdout at 21.1%. All points run and verify.
The original headline errors were vectoradd 38.0%, ReLU 21.5%, and matrix
multiplication 58.7%.

The host clock is unpinned, so these microseconds are still diagnostic. A
cycle-counter probe or pinned rerun is needed for reference-quality absolute
latency.

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

This moves exact pagerank from 42.9 to 22.3 µs without serializing unrelated
random misses. A shared L2-to-DRAM token bucket then separates a 625 KiB
burst region from sustained bandwidth: it begins with 10,000 cache-line
credits and refills at one 64-byte request per 1.244 GHz GPU cycle. The burst
avoids imposing the sustained streaming rate on short random traffic; the
final model puts the 262,144-element vector at 27.147 µs versus a 29.664 µs
three-process hardware median.

### Sustained writes need a separate shared path

The original per-transaction dense-store delay could not represent the size
curve: increasing it enough for 262K ReLU overcharged 16K and 65K grids. The
shared L1-to-L2 connection now supports filtering token credits by request
class. RX 570 writes can burst for 3,072 cache lines (192 KiB), then issue at
three lines per five GPU cycles; reads and responses remain unlimited.

This mechanism was selected from five ReLU sizes, including a new 131K
midpoint, and checked against all four vectoradd sizes. Full-line store issue
cost is now charged once per wave instruction rather than once per generated
cache-line request. This distinction preserves the scalar-store path while
allowing wide stores to use the transaction bandwidth they already model.
The change reduces transpose family MARE from 28.8% to 8.8% and the complete
33-point MARE from 8.6% to 7.0%. A 128 KiB burst still overcharges 65K points.

| vector length | HW steady (µs) | Sim (µs) | Error |
|---:|---:|---:|---:|
| 4,096 | 4.643 | 4.808 | 3.6% |
| 16,384 | 5.059 | 5.034 | 0.5% |
| 65,536 | 9.299 | 7.340 | 21.1% |
| 262,144 | 29.664 | 27.147 | 8.5% |

Vectoradd has 8.4% family MARE, with the 65K point retained as a visible
21.1% outlier. The model was selected from the full curves rather than tuned
to remove that single point.

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
shortfall: its four size errors are 6.2%, 5.3%, 5.8%, and 4.8%. FIR, k-means,
and pagerank remain in range. The later per-instruction store-issue correction
resolves transpose without restoring an opcode-width penalty.

### AES and dense stores need distinct issue timing

AES is dominated by bitwise vector operations. Classifying vector XOR, AND,
and OR separately from the four-cycle default VALU class models their
one-cycle issue/result timing and produces 16.892 µs versus the 18.595 µs
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

This produces 321.3 µs for bitonic versus 348.7 µs hardware. The much larger
10–71 µs cold-minus-steady values in the hardware file include ROCm/KFD
first-use, page mapping, and other host-side effects; they are deliberately
not folded into GPU execution time.

## Cross-size holdout evidence

The exact-HSACO harness and simulator runner now accept a single benchmark
size, and the sweep driver records four predetermined, launch-compatible
sizes for vectoradd, ReLU, matrix multiplication, matrix transpose, bitonic
sort, AES, FIR, and NW, plus a ReLU midpoint. The committed hardware CSV uses
the median of three independent processes per point. All 33 matched hardware
and simulator points are present and every simulator point verifies.

| Family | MARE | Maximum error | Sim/HW slope ratio |
|---|---:|---:|---:|
| vectoradd | 8.4% | 21.1% | 0.903 |
| relu | 8.9% | 13.5% | 0.856 |
| matrixmult | 5.5% | 6.2% | 0.961 |
| matrixtranspose | 8.8% | 12.5% | 1.142 |
| bitonicsort | 6.1% | 7.9% | 1.187 |
| aes | 9.4% | 9.8% | — |
| fir | 3.2% | 4.1% | — |
| nw | 5.2% | 5.5% | 0.944 |

The transpose correction is structural rather than benchmark-specific. A
wide store generates up to 16 line requests but issues one wave instruction;
charging the full-line issue delay to each request was double-counting work.
LDS latency and barrier-cost experiments had changed transpose by less than
0.1 µs or regressed matrix multiplication, so both remain reverted.

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
- Full-line store cost is 48 cycles once per wave instruction, while L1V
  latency remains 56 cycles. Charging that cost once per cache-line request
  was rejected by the wide-store cross-size evidence.
- A shared one-request/cycle limiter without burst capacity fixed the large
  vector slope but pushed pagerank to 24.769 µs. A 10,000-line token bucket
  is the selected compromise between random-miss overlap and streaming slope;
  smaller 6,500- and 9,000-line candidates overcharged pagerank.

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
- Add size sweeps for k-means and pagerank, and investigate the 65K
  vectoradd and canonical pagerank residuals without benchmark-specific timing.
