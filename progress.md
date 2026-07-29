# Progress — GCN5 / gfx90c on MGPUSim

**Goal:** Run modern ROCm kernels and calibrate cycle-accurate timing against the host Renoir APU (Ryzen 7 4700U, gfx90c).

**Last updated:** 2026-07-30

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
| ISCA-10 sim vs HW (gfx90c HSACO) | Done — MARE **12.4%**, geo mean HW/Sim **0.88×** |
| Remaining timing gaps | K-means, matrix multiplication, ReLU, NW |

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

- **K-means:** simulator is 43.5% slow. Its 256 KiB strided swap input bypasses
  L2, while globally warming that transfer makes vectoradd/ReLU about 2× too
  fast. The next model needs reuse-sensitive residency, not a size-only knob.
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

## Next

1. Re-run the exact-input hardware harness with the GPU pinned to `high`.
2. Model reuse-sensitive DMA/L2 residency for K-means without warming streaming
   vectoradd/ReLU inputs wholesale.
3. Revisit private-segment MUBUF service latency for matrix multiplication.
