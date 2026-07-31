# RX 570 timing-model progress

Last updated: 2026-07-31

## Outcome

The RX 570 preset now runs and verifies all ten calibration workloads with
the exact gfx803 binaries used by the hardware harness.

| Benchmark | HW steady (µs) | Sim (µs) | Error |
|---|---:|---:|---:|
| vectoradd | 7.101 | 7.898 | 11.2% |
| relu | 6.837 | 5.902 | 13.7% |
| matrixmult | 73.458 | 74.388 | 1.3% |
| matrixtranspose | — | 72.515 | — |
| bitonicsort | 350.214 | 355.086 | 1.4% |
| aes | 18.084 | 20.243 | 11.9% |
| fir | 7.637 | 5.814 | 23.9% |
| kmeans | 29.424 | 25.982 | 11.7% |
| pagerank | 20.356 | 24.983 | 22.7% |
| nw | 195.281 | 170.430 | 12.7% |

Steady-state MARE is **12.3%** across the nine benchmarks with steady hardware
data. All ten run and verify, but matrix transpose has only a cold hardware
measurement and is excluded from steady MARE. The original headline errors
were vectoradd 38.0%, ReLU 21.5%, and matrix multiplication 58.7%; matrix is
now 1.3%, and no measured steady-state workload exceeds 23.9%.

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

### Latency and bandwidth were conflated

The old simple-banked DRAM used 16 internal pipelines per controller and
400 ns pipeline latency. Lowering that latency helped pagerank but made
streaming workloads much too fast because the model still exposed excessive
parallel line service.

The final configuration separates the two:

- eight external 32-bit controller channels;
- one serialized internal pipeline per controller;
- 250 MHz line-service clock, or 128 GB/s aggregate for 64-byte lines;
- 40-cycle depth, or 160 ns access latency.

This moved exact pagerank from 42.9 to 25.0 µs while retaining a finite
streaming bandwidth ceiling.

### Sparse reads were charged twice

Pagerank already generates one cache-line request for each random gather line
and pays cache/DRAM latency. The inherited 24-cycle low-utilization penalty
then serialized every sparse line again. Disabling that extra read penalty
reduced pagerank without changing the actual transaction count. The ordinary
write and wide-store penalties remain.

### Matrix needs a wide-load path

The exact gfx803 matrix kernel uses `flat_load_dwordx4` and became VMem-bound
in the simulator. A configurable wide-read serialization cost, 126 cycles per
generated line for these x4 loads, brings matrix execution to 74.4 µs versus
73.5 µs hardware. It does not penalize ordinary dword streaming loads.

### Multi-kernel dispatch cost

Bitonic sort launches 78 kernels and NW launches 16. Calibrated GPU-side
dispatch costs are:

| Cost | Cycles |
|---|---:|
| first launch | 1,000 |
| subsequent launch | 1,300 |
| completion/post-kernel | 2,500 |

This produces 355.1 µs for bitonic versus 350.2 µs hardware. The much larger
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

- FIR is 23.9% fast and pagerank is 22.7% slow. Addressing them should use
  general compute/memory overlap mechanisms, not benchmark-name exceptions.
- Add `s_memtime` cycle-counter measurements and rerun with a pinned clock.
- The current user cannot access `/dev/kfd` (`root:render`, mode 0660), so new
  hardware collection needs render-group access or administrator help.
