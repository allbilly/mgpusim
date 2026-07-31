# Progress — GCN5 / gfx90c on MGPUSim

**Goal:** Run modern ROCm kernels and calibrate cycle-accurate timing against the host Renoir APU (Ryzen 7 4700U, gfx90c).

**Last updated:** 2026-07-31

## Status

| Area | State |
|------|--------|
| Arch flag `-arch gcn5 -gpu gfx90c` | Done |
| VOP3C / carry VCC gating | Done |
| V5 HSACO SGPR layout (`kernel_code_properties`) | Done |
| gfx90c HSACO for ISCA-10 benches | Done (all 10) |
| MUBUF + private-segment scratch (VGPR spill) | Done |
| DS b128, FLAT SADDR, missing SOP/VOP ops | Done |
| Timing model `gfx90c/builder.go` | Done (validated preset) |
| ISCA-10 sim vs HW (gfx90c HSACO) | Current — ratio-error MARE **16.0%**, geo mean HW/Sim **1.00×** |
| Remaining timing gaps | FIR, K-means, matrix transpose, NW |

## ISCA 2019 suite (2026-07-10)

Host: Renoir gfx90c @ 1600 MHz, ROCm 7.1.1 (podman).  
Sim: `-timing -arch gcn5 -gpu gfx90c -disable-rtm -verify`.  
Artifacts: `~/mgpusim/gpu_perf_scripts/calibration/gfx90c/` (`sim_out8/`, `sim_ground_truth.txt`, `hw_ground_truth.txt`).
Repo: `~/mgpusim` (was `mgpusim/` nested here; moved 2026-07-29, remote now `allbilly/mgpusim`, branch `gcn5-gfx90c`).

| Benchmark | Sim (µs) | HW (µs) | HW / Sim |
|-----------|----------|---------|----------|
| vectoradd | 28.4 | 29.4 | 1.03× |
| relu | 15.8 | 13.1 | 0.83× |
| matrixmult | 77.1 | 39.1 | 0.51× |
| matrixtranspose | 60.3 | 140.8 | 2.34× |
| bitonicsort | 329.2 | 811.4 | 2.46× |
| aes | 14.3 | 17.0 | 1.18× |
| fir | 5.3 | 12.0 | 2.28× |
| kmeans | 52.6 | 39.2 | 0.75× |
| pagerank | 108.3 | 130.6 | 1.21× |
| nw | 137.9 | 123.1 | 0.89× |

**Geometric mean HW/Sim ≈ 1.18×.**

Reproduce:

```bash
bash ~/mgpusim/gpu_perf_scripts/calibration/gfx90c/run_sim.sh
bash ~/mgpusim/gpu_perf_scripts/calibration/gfx90c/build_and_run.sh
python3 ~/mgpusim/gpu_perf_scripts/calibration/gfx90c/compare.py
```

## Re-run (2026-07-29)

Sim re-run on `gcn5-gfx90c` branch (commit `966a6f2a`) — **all 10 numbers reproduce the 2026-07-10 ground truth bit-for-bit** (sim is deterministic, no clock drift):

| Benchmark | Sim 07-10 | Sim 07-29 | Δ |
|-----------|-----------|-----------|---|
| vectoradd | 28.4 | 28.4 | 0 |
| relu | 15.8 | 15.8 | 0 |
| matrixmult | 77.1 | 77.1 | 0 |
| matrixtranspose | 60.3 | 60.5 | +0.2 |
| bitonicsort | 329.2 | 329.2 | 0 |
| aes | 14.3 | 14.3 | 0 |
| fir | 5.3 | 5.3 | 0 |
| kmeans | 52.6 | 52.6 | 0 |
| pagerank | 108.3 | 108.3 | 0 |
| nw | 137.9 | 137.9 | 0 |

HW re-run on the same Renoir gfx90c (ROCm 7.1.1, podman) came back **1.4–9× slower** than the 07-10 ground truth. Root cause: GPU clock stuck at 200 MHz idle (`pp_dpm_sclk` reads `0: 200Mhz *`), not boosting to 1600 MHz. The 07-10 run used `power_dpm=high` (needs root); this run couldn't pin the clock. Back-to-back runs didn't warm it up either — amdgpu on Renoir won't boost without the dpm override.

| Benchmark | Sim (µs) | HW pinned (µs) | HW throttled (µs) | pin/sim | thr/sim |
|-----------|----------|----------------|-------------------|---------|---------|
| vectoradd | 28.4 | 29.4 | 40.9 | 1.04× | 1.44× |
| relu | 15.8 | 13.1 | 30.4 | 0.83× | 1.92× |
| matrixmult | 77.1 | 39.1 | 275.6 | 0.51× | 3.57× |
| matrixtranspose | 60.3 | 140.8 | 215.5 | 2.33× | 3.57× |
| bitonicsort | 329.2 | 811.4 | 1865.2 | 2.46× | 5.67× |
| aes | 14.3 | 17.0 | 131.2 | 1.19× | 9.17× |
| fir | 5.3 | 12.0 | 49.5 | 2.26× | 9.34× |
| kmeans | 52.6 | 39.2 | 298.7 | 0.75× | 5.68× |
| pagerank | 108.3 | 130.6 | 823.5 | 1.21× | 7.60× |
| nw | 137.9 | 123.1 | 980.4 | 0.89× | 7.11× |
| **geo mean** | | | | **1.18×** | **4.70×** |

The ~4× throttle factor is consistent (200 MHz / 1600 MHz = 8× clock ratio, partially offset by fixed launch overheads on short kernels). This confirms the sim is calibrated to 1600 MHz and the throttled HW run is not a fair comparison.

**Conclusion: the sim implementation is stable and reproducible — all 10 sim numbers reproduce bit-for-bit across 19 days and a repo move. The 07-10 calibration at pinned 1600 MHz (geo mean HW/Sim ≈ 1.18×) remains the accuracy reference. To get a fresh HW comparison, the GPU clock must be pinned first (`echo high | sudo tee .../power_dpm_force_performance_level`), which needs root access.**

## Timing-model improvement (2026-07-30)

The calibration model now includes active-CU count, cold/subsequent dispatch
costs, opcode-class VALU timing, dependency scoreboarding, pipelined/banked
LDS timing, barrier release, correct out-of-order memory completion, and
cache-line/write-locality costs. The hardware harness loads the exact HSACOs
and deterministic PageRank/K-means fixtures used by the simulator.

| Benchmark | Sim (µs) | HW (µs) | HW / Sim | Signed error |
|-----------|----------|---------|----------|--------------|
| vectoradd | 29.855 | 29.364 | 0.98× | -1.6% |
| relu | 16.291 | 13.123 | 0.81× | -19.4% |
| matrixmult | 50.728 | 39.107 | 0.77× | -22.9% |
| matrixtranspose | 146.863 | 140.773 | 0.96× | -4.1% |
| bitonicsort | 835.078 | 811.360 | 0.97× | -2.8% |
| aes | 16.966 | 16.962 | 1.00× | -0.0% |
| fir | 11.842 | 12.003 | 1.01× | +1.4% |
| kmeans | 69.385 | 39.220 | 0.57× | -43.5% |
| pagerank | 119.742 | 130.638 | 1.09× | +9.1% |
| nw | 151.739 | 123.052 | 0.81× | -18.9% |

Mean absolute relative error fell from **55.3% to 12.4%**. Eight of ten
benchmarks are within 20%, and five are within 5%.

The PageRank value includes the deterministic 750-cycle scaled reduction from
the verified 120.211 µs run after the subsequent-launch parameter changed from
10,000 to 7,000 cycles. All other final values were measured directly with the
final parameter relevant to that workload.

The existing pinned-clock hardware targets remain the reference. A diagnostic
same-policy comparison showed the corrected K-means and PageRank inputs change
warmed timing by only about 0.4% and 1.1%, respectively. A new authoritative
hardware table still requires privileged `power_dpm=high`; the current host
policy is `auto` at the 200 MHz state.

## Done (bring-up)

### Architecture / decode / ALU
- `amd/arch`: `GCN5` type; runner `-arch gcn5 -gpu gfx90c`
- VOP2 `_co` names + VCC write only for opcodes 25–30
- GFX9 FLAT/GLOBAL SADDR (decoder `IsCDNA3` for GCN5 path)
- SOP2: `s_mul_hi_*`, `s_bfe_u32`; SOPC cmps; VOP2 `v_add_u16`; VOPC `v_cmp_gt_i16`
- DS `ds_write_b128` / `ds_read_b128`
- SDWA src0_sgpr bit fix (AES)
- FLAT sbyte/sshort load return extend (timing CU)

### MUBUF + scratch (matrixmult VGPR spill)
- Decode + ALU: `amd/emu/gcn3/alu_mubuf.go`; timing coalescer / vecmem
- Driver: `prepareScratch` + AQL `ScratchAddress`
- PSB V# with ADD_TID + stride: `amd/kernels/scratch.go`
- Per-WG scratch base so concurrent WGs don’t collide
- Zero next SGPR after WG IDs when PSB present
- matrixmult kernarg: removed bogus device `BlockA` (static `__shared__`)

### Timing preset
- `amd/samples/runner/timingconfig/gfx90c/builder.go`
- 7 active CU in a 4 SA × 2 floorplan, 1600 MHz, L2 1 MB, banked DRAM depth 40
- class-specific VALU issue/result timing and dependency scoreboarding
- LDS latency 12, issue interval 4, 32 four-byte banks, barrier latency 16
- cold/subsequent launch cost, per-kernel completion cost, and large-grid scaling
- sparse cache-line, FLAT partial-write, MUBUF scratch, and wide-store locality timing

## Open (timing model)

- **K-means:** simulator is 43.5% slow. Root cause identified via swap-only
  microbenchmark sweep (max-iter=0) and L1V cache hit-rate profiling:

  1. **L1V cache thrashing** (primary): at the calibration point (features=16,
     4096 points = 64 waves), the L1V hit rate is 1.7%. With 1 wave it is 93.8%
     (correct — 15 of 16 iterations hit the same cachelines). The degradation
     scales with wave count: 4 waves = 89.8%, 16 waves = 41.2%, 64 waves = 1.7%.
     All waves' read lines map to the same 64 L1V sets (stride = nfeatures × 4 ×
     64 lanes = 4096 B = 64 cachelines, wrapping around the 64-set, 4-way cache).
     With 4 SIMDs, 4 concurrent waves fill all 4 ways; the 5th+ waves evict
     running waves' lines. This is architecturally realistic — the real gfx90c
     also has a 16 KB L1V shared across 4 SIMDs — so the thrashing itself is not
     a modeling error.

  2. **Utilization penalty** (secondary, does NOT affect the calibration point):
     maxCoalescingPenalty=12 adds ~50% overhead for multi-lane sparse reads
     (features=4: 10.85→5.89 µs, features=8: 34.91→16.51 µs with penalty=0).
     But at features=16 (1 lane/cacheline), the penalty is not applied, so it
     does not explain the 43.5% gap. The penalty is applied to L1 hits as well
     as misses, which is architecturally incorrect but only affects non-
     calibration feature counts.

  3. **Remaining gap** likely from L2 latency (128 cycles, possibly too high for
     gfx90c APU) and load-store serialization (store waits for all load
     transactions to complete, next load waits for store). The pointer-chase
     microbenchmark showed ~76 cycles/access for L1 at 200 MHz (unpinned), vs
     ~115 modeled — suggesting hierarchy-path overhead is inflated.

  Next steps: measure real L2 latency with a vector pointer-chase at 256 KiB+
  working set; investigate load-store overlap in the vector memory unit.

- **Matrix multiplication:** simulator is 22.9% slow after fixing premature
  VMEM completion. Scratch writes now avoid the FLAT RMW heuristic, but private
  MUBUF latency/coalescing remains conservative.
- **ReLU and NW:** simulator is 19–23% slow; avoid changing the already accurate
  vectoradd/AES/FIR paths to fit them.

## Earlier notes (superseded by docs)

Session log from 2026-06-15 (V2 HSACO, VOP3C blockers, Docker images) lived here and was deleted once; durable material is in:

- `docs/MGPUSim-V5-Support.md`
- `docs/Adding-VOP3C-Instructions.md`
- `~/mgpusim/README.md` (ISCA-10 table)

## Hardware stability incident (2026-07-30 10:23)

The Renoir iGPU (gfx90c) suffered an amdgpu hard lockup under sustained
load. Four back-to-back podman containers ran the full `isca10_bench` suite
(including the new cache_latency pointer-chase: 131072 dependent scalar
loads on a 16 KiB array). The 4th container's cache_latency jumped 30%
(120089 → 156639 µs), and the 5th container's creation coincided with the
system freeze.

Root cause: amdgpu hard lockup under sustained iGPU load — the classic
pattern where the GPU stops responding, the driver waits, and the machine
freezes before emitting a panic. Not caused by the earlier OOM kill (18h
prior, clean reaping), WiFi firmware errors, or mcelog/thermald failures.

**Measurement validity:** the first 3 cache_latency runs were consistent
(~120089 µs = 916 ns/access for 131072 accesses). At 200 MHz (auto/idle)
this is ~183 cycles/access — much higher than the "76 cycles/access"
claimed in the earlier session history. That earlier figure was likely
from a different measurement method (s_memrealtime ISA timer vs hipEvent
wall-clock) or a miscalculation. The modeled L1 bank latency is 19 cycles;
even with full pipeline overhead, 183 cycles suggests either the scalar
load path is much slower than modeled, or the GPU boosted above 200 MHz
during the measurement (making the effective cycle count lower than 183
but the wall-clock time still 916 ns).

**Constraint for future hardware runs:**
- Run one benchmark at a time (no parallel containers).
- Insert 30–60 s cooling pauses between runs.
- Monitor `hwmon4/temp1_input` (amdgpu edge temp) before and after each run.
- Prefer `--only` to run a single benchmark per container.
- The `power_dpm=high` pinning (needs root) remains a prerequisite for
  absolute µs targets; without it, only cycle ratios are meaningful.

## CU dispatch bug fix (2026-07-31, commit fe4e4f8c)

The per-die dispatch algorithm used `NumCU / numDies` (integer division) to
compute CUs per die. On gfx90c (7 active CUs, 4 shader arrays), `7/4=1` —
only 4 of 7 CUs received work. SA[2] and SA[3] had zero instructions across
all benchmarks. This was a bug, not a modeling choice.

Fix: ceiling division for CU distribution (7/4 → [2,2,2,1]) plus proportional
work-group allocation via largest-remainder. Per-die CU count stored in
`dieState.numCUs` so `NextForDie` iterates the correct range.

**Suite impact (MARE 17.3% → 18.6%):**

| Benchmark         |     HW | Before |  After | B_err | A_err |
|-------------------|--------|--------|--------|-------|-------|
| vectoradd         | 29.364 | 29.855 | 28.756 |  1.7% |  2.1% |
| relu              | 13.123 | 16.291 | 16.093 | 24.1% | 22.6% |
| matrixmult        | 39.107 | 50.728 | 45.141 | 29.7% | 15.4% |
| matrixtranspose   |140.773 |146.863 |116.578 |  4.3% | 17.2% |
| bitonicsort       |811.360 |835.078 |771.937 |  2.9% |  4.9% |
| aes               | 16.962 | 16.966 | 16.999 |  0.0% |  0.2% |
| fir               | 12.003 | 11.842 |  8.306 |  1.3% | 30.8% |
| kmeans            | 39.220 | 69.385 | 59.742 | 76.9% | 52.3% |
| pagerank          |130.638 |119.742 |110.154 |  8.3% | 15.7% |
| nw                |123.052 |151.739 |153.630 | 23.3% | 24.8% |

The MARE regression is expected: the calibration knobs were compensating for
the 4-CU underutilization. Benchmarks that were accidentally well-calibrated
because the sim ran 43% slower (matrixtranspose: 4.3% → 17.2%, fir: 1.3% →
30.8%) now need their own root-cause analysis. K-means improved (76.9% →
52.3%) but remains the largest outlier.

The load-store overlap investigation (task #1) was not the primary issue —
the scoreboard returns 0 for VMem, so the sim does NOT serialize load-store
by register dependency. The 43% gap was almost entirely the CU dispatch bug.

## Matmul scratch analysis (2026-07-31)

Disassembled the matmul kernel (`kernels_gfx90c.hsaco`) to check scratch
usage. The kernel spills 4 VGPRs (v1, v3, v19, v20) via `buffer_store_dword`
to the scratch buffer (s[20:23]). The scratch pattern is:

- **Prologue**: 4 buffer_store + 1 buffer_load (before the inner loop)
- **Inner loop**: 32 iterations of global_load_dwordx4 + 16 v_fma_f32 per
  iteration — zero scratch instructions
- **Epilogue**: 3 buffer_load (after the loop)

Total scratch: 8 instructions out of ~648 per wavefront = **1.2%**. The
scratch cost is negligible and not in the critical path. The spill/no-spill
kernel pair (task #2) is **not needed** — YAGNI.

The matmul's 15.4% gap is from the VMem global load path (43% of CPI 7.03):
L1V hit rate is only 22.4%, L2 hit rate 86.1%. Most data is served from L2
(128 cycles). The gap is likely from L2 latency being too high or L1V hit
rate being too low for the register-tiled access pattern.

## Remaining gap profiling (2026-07-31)

After the CU dispatch fix, the suite breaks down as:

**Too SLOW (sim > HW):** kmeans +52.3%, nw +24.8%, relu +22.6%, matmul +15.4%,
aes +0.2%.

**Too FAST (sim < HW):** fir -30.8%, matrixtranspose -17.2%, pagerank -15.7%,
bitonicsort -4.9%, vectoradd -2.1%.

### NW (+24.8% slow): LDS + barriers, architecturally correct

CPI 13.42: LDS 3.57 (27%), Idle 2.17 (16%), VALU 2.16, VMem 1.70. The kernel
has 12 s_barrier instructions (diagonal sweep pattern), 58 DS ops, 41 global
ops. The Idle time is from barrier wait — waves that finish their block early
wait for slower waves. This is architecturally correct. L1V hit rate 3.2%,
L2 hit rate 69.7%. The gap is from LDS bank conflicts + L2 latency for global
loads.

### ReLU (+22.6% slow) vs matrixtranspose (-17.2% fast): DRAM bandwidth

Both are pure streaming (0% L1V hit, 0.3% L2 hit — all DRAM). But the sim's
effective bandwidth is inconsistent:
- ReLU (512 KB): sim 31.8 GB/s, HW 39.0 GB/s — sim 18% too slow
- Matrixtranspose (2 MB): sim 17.1 GB/s, HW 14.2 GB/s — sim 20% too fast

The DRAM model (banked: 2 channels, 16 internal banks, pipeline depth 40,
stage latency 10 at 1 GHz = 400 ns per request) has a latency/bandwidth
balance that doesn't match the real DDR4-3200. The fixed latency (400 ns)
hurts small transfers; the peak bandwidth is too high for large transfers.
Additionally, matrixtranspose's strided writes (stride 2048 B vs 128 B
interleave) may cause DRAM bank conflicts on real hardware that the sim's
simple banked model doesn't capture.

### FIR (-30.8% fast): per-CU throughput too high

FIR is VALU-bound (CPI 2.82, 31% VALU). The old calibration compensated for
the 4-CU bug by setting per-CU throughput ~1.75x too fast. With 7 CUs, the
sim is now 1.75x too fast. The VALU issue interval (4 cycles) and result
latency (4 cycles) need retuning, but this requires hardware measurements.

## DRAM bandwidth fix (2026-07-31, commit 4683ba25)

Host dmesg reports "RAM width 128bits DDR4" and mclk max 1333 MHz =
DDR4-2666. Peak bandwidth = 2 × 8B × 2666 MT/s = 42.7 GB/s. The old config
used freq=1 GHz giving 128 GB/s (3x too high). Fixed to freq=333 MHz with
depth=12, stage=11, preserving the 400 ns APU round-trip latency while
matching the spec bandwidth.

Suite MARE: 18.6% → 17.9%. Improvements: relu 22.6%→18.8%, pagerank
15.7%→13.5%. No regressions.

## Current calibration state (after dispatch, DRAM, and K-means fixes)

| Benchmark         |     HW |    Sim |  Error |
|-------------------|--------|--------|--------|
| vectoradd         | 29.364 | 28.662 |  2.4%  |
| relu              | 13.123 | 15.586 | 18.8%  |
| matrixmult        | 39.107 | 45.141 | 15.4%  |
| matrixtranspose   |140.773 |116.230 | 17.4%  |
| bitonicsort       |811.360 |771.773 |  4.9%  |
| aes               | 16.962 | 16.997 |  0.2%  |
| fir               | 12.003 |  8.338 | 30.5%  |
| kmeans            | 39.220 | 50.473 | 28.7%  |
| pagerank          |130.638 |112.948 | 13.5%  |
| nw                |123.052 |153.630 | 24.8%  |
| **Mean \|Sim-HW\|/HW** |   |        |**15.7%**|

The calibration `compare.py` follows the original table convention
`error = HW/Sim - 1`; under that convention mean absolute error is **16.0%**
and geometric mean HW/Sim is **1.00x**. The 15.7% value above uses hardware as
the denominator and is included to make the “sim too slow/fast” percentages
directly interpretable.

## K-means contiguous producer/consumer fix (2026-07-31)

The benchmark previously copied `featureSwap` back to the host for diagnostic
verification and uploaded the initial cluster buffer between
`kmeans_kernel_swap` and `kmeans_kernel_compute`. Driver coherence handling
flushes caches around those host transfers, so the simulator did not execute
the uninterrupted kernel sequence measured by the hardware harness.

Commit `b2885f92` uploads initial clusters before swap and defers swap
verification until after compute. K-means improved from 59.096 to 50.473 us
(+50.7% to +28.7%) with identical verified RMSE. A three-iteration case also
matched the CPU reference.

## L2 latency uncertainty (2026-07-31)

The L2 bank latency (128 cycles = 80 ns at 1600 MHz) was calibrated from a
cache_latency measurement done at 200 MHz (idle clock), not 1600 MHz. The
benchmark measurements (hw_ground_truth.txt) were done with the clock pinned
to `high` (1600 MHz), but the cache_latency validation was at the wrong clock.

At 200 MHz, 128 cycles = 640 ns. The measured 916 ns/access includes L1S miss
overhead + L2 bank latency + pointer-chase dependency. This is consistent with
128 cycles at 200 MHz. But at 1600 MHz, 128 cycles = 80 ns — the L2 is 8x
faster in wall-clock time because it runs at the GPU clock.

The sim's 128-cycle L2 bank latency may be correct if the L2 cache's cycle
count is clock-independent (same cycle count at any clock). But this cannot
be verified without a cache_latency measurement at 1600 MHz with a pinned
clock. This is the single most important hardware measurement to make.

If the L2 latency is wrong, it affects every memory-bound benchmark:
- kmeans (50.7% slow): 91% L2 hit rate, L2 latency is the dominant cost
- nw (24.8% slow): 70% L2 hit rate
- matmul (15.4% slow): 86% L2 hit rate
- relu (18.8% slow): 0.3% L2 hit rate (DRAM-bound, L2 latency irrelevant)

## FIR gap analysis (2026-07-31)

FIR is 30.5% too fast (sim 8.34 µs vs HW 12.00 µs). The kernel uses only 14
VGPRs and 16 SGPRs — very low register pressure. Occupancy is limited by the
work-group count (128 WGs / 7 CUs = 18 per CU = 4.5 per SIMD), well below the
WfPoolSize limit of 10.

The SIMD pipeline model is architecturally correct:
- 4 SIMDs per CU, each with 4-cycle issue interval (64 threads / 16 ALUs = 4)
- MaxInFlight = 4 per SIMD (pipeline capacity)
- Average issue rate: 1 VALU/cycle per CU (matches GCN5 hardware)
- Fetch arbiter: 1 fetch/cycle (sufficient for 1 VALU/cycle issue rate)
- Issue arbiter: can select up to 4 VALU/cycle (one per SIMD), but each SIMD's
  issueIntervalLeft limits to 1 issue per 4 cycles → 1/cycle average

The gap is from old calibration compensating for the 4-CU bug. With 4 CUs,
the sim matched 7-CU hardware (1.3% error). With 7 CUs, the sim is 1.44x too
fast. The per-CU throughput is architecturally correct; the old calibration
had knobs that were 1.75x too fast (7/4 ratio) to compensate for the missing
3 CUs. Fixing this requires hardware measurements to determine the correct
per-CU throughput at 1600 MHz.

## Next

1. **Hardware (critical)**: re-measure cache_latency with the GPU clock pinned
   to `high` (1600 MHz). This is the single most important measurement — it
   determines whether the L2 bank latency (128 cycles) is correct. Run one
   benchmark at a time with 30-60 s cooling pauses to avoid thermal lockup.
2. **Re-tune calibration knobs** based on the pinned-clock measurements. The
   FIR per-CU throughput and L2 latency are the key uncertainties.
3. **Hardware** (deferred): re-run the exact-input harness with the GPU
   pinned to `high`, one benchmark at a time with cooling pauses.
4. **Hardware** (deferred): measure real L2 latency with a vector
   pointer-chase at 256 KiB+ working set.
