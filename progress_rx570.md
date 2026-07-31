# RX 570 timing-model progress

Last updated: 2026-07-31

## Outcome

The RX 570 preset now runs and verifies all ten calibration workloads with
the exact gfx803 binaries used by the hardware harness.

| Benchmark | HW steady (µs) | Sim (µs) | Error |
|---|---:|---:|---:|
| vectoradd | 7.101 | 7.271 | 2.4% |
| relu | 6.837 | 6.273 | 8.2% |
| matrixmult | 73.458 | 80.310 | 9.3% |
| matrixtranspose | — | 77.155 | — |
| bitonicsort | 350.214 | 320.428 | 8.5% |
| aes | 18.084 | 18.616 | 2.9% |
| fir | 7.637 | 7.692 | 0.7% |
| kmeans | 29.424 | 26.731 | 9.2% |
| pagerank | 20.356 | 22.303 | 9.6% |
| nw | 195.281 | 180.161 | 7.7% |

Steady-state MARE is **6.5%** across the nine benchmarks with steady hardware
data, and every measured workload is below 10% error. All ten run and verify,
but matrix transpose has only a cold hardware measurement and is excluded
from steady MARE; its simulated 77.155 µs is 2.0% below the 78.722 µs cold
measurement. The original headline errors were vectoradd 38.0%, ReLU 21.5%,
and matrix multiplication 58.7%.

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
random misses. A shared L2-to-DRAM token bucket then separates the 896 KiB
burst region from sustained bandwidth: it begins with 14,336 cache-line
credits and refills at one 64-byte request per 1.244 GHz GPU cycle. The
default vectoradd and pagerank points remain bit-for-bit unchanged, while the
262,144-element vector improves from 12.839 to 23.775 µs versus 24.3 µs
hardware.

| vector length | HW steady (µs) | Sim (µs) | Error |
|---:|---:|---:|---:|
| 4,096 | 4.400 | 5.055 | 14.9% |
| 16,384 | 5.000 | 5.209 | 4.2% |
| 65,536 | 7.101 | 7.271 | 2.4% |
| 262,144 | 24.300 | 23.775 | 2.2% |

The sustained-throughput error is now below 10%; the remaining vector outlier
is the smallest, launch-dominated point.

### Sparse reads were charged twice

Pagerank already generates one cache-line request for each random gather line
and pays cache/DRAM latency. The inherited 24-cycle low-utilization penalty
then serialized every sparse line again. Disabling that extra read penalty
reduced pagerank without changing the actual transaction count. The ordinary
write and wide-store penalties remain.

### Matrix needs a wide-load path

The exact gfx803 matrix kernel uses `flat_load_dwordx4` and became VMem-bound
in the simulator. A configurable wide-read serialization cost, 126 cycles per
generated line for these x4 loads, brings matrix execution to 80.3 µs versus
73.5 µs hardware, within the 10% target. It does not penalize ordinary dword
streaming loads.

### AES and dense stores need distinct issue timing

AES is dominated by bitwise vector operations. Classifying vector XOR, AND,
and OR separately from the four-cycle default VALU class models their
one-cycle issue/result timing and reduces AES from 20.243 to 18.616 µs.

Partial stores and fully utilized cache-line stores also have different
costs. The model now keeps the partial-line write-combine/RMW cap and adds a
small independent full-line store serialization cost. That mechanism brings
the dense streaming kernels into range without applying benchmark-name rules.

### Multi-kernel dispatch cost

Bitonic sort launches 78 kernels and NW launches 16. Calibrated GPU-side
dispatch costs are:

| Cost | Cycles |
|---|---:|
| first launch | 2,690 |
| subsequent launch | 1,300 |
| completion/post-kernel | 2,500 |

This produces 320.4 µs for bitonic versus 350.2 µs hardware. The much larger
10–50 µs cold-minus-steady values in the hardware file include ROCm/KFD
first-use, page mapping, and other host-side effects; they are deliberately
not folded into GPU execution time.

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
- Full-line store cost 8 cycles and L1V latency 56 cycles jointly put
  vectoradd, ReLU, k-means, and pagerank below 10%.
- A shared one-request/cycle limiter without burst capacity fixed the large
  vector slope but pushed pagerank to 24.769 µs. A 14,336-line token bucket
  preserves its random-miss overlap and retains the large-vector correction.

## Reproduction

```bash
cd gpu_perf_scripts/calibration/rx570
SIM_JOBS=4 ./run_sim.sh sim_out | tee sim_results.txt
./compare.py sim_results.txt
```

The default comparison uses the recorded steady hardware column. To display
the deliberately unmatched cold runtime measurement:

```bash
./compare.py --hw-mode cold sim_results.txt
```

## Remaining work

- Investigate small-grid launch timing; the 4,096-element vector point is
  14.9% slow while the three larger size-sweep points are below 10%.
- Add `s_memtime` cycle-counter measurements and rerun with a pinned clock.
- The current user cannot access `/dev/kfd` (`root:render`, mode 0660), so new
  hardware collection needs render-group access or administrator help.
