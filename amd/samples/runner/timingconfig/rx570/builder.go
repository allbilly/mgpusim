// Package rx570 provides a timing model for AMD Polaris 20 / Radeon RX 570
// (gfx803). The chip markets as "GCN 4.0" but this codebase classifies
// gfx803 under arch.GCN3 (see amd/arch/arch.go); the ISA is executed by the
// existing GCN3 ALU, exactly as the polaris10 (RX 480) preset does.
//
// Physical parameters below (CU count, clocks, cache sizes, GDDR5 bandwidth)
// are taken from the AMD RX 570 product page and corroborating spec sheets.
// Timing latencies are calibrated against the ISCA-10 benchmark suite run on
// the physical RX 570 (see gpu_perf_scripts/calibration/rx570/hw_ground_truth.txt).
// The steady-state calibration achieves 6.3% MARE across the 9 benchmarks
// with steady hardware data, with every error below 10%; all 10 benchmarks
// execute and verify. Cold
// hipEvent measurements are reported separately because they include
// workload-dependent host/runtime first-use costs outside the GPU model.
package rx570

import (
	"github.com/sarchlab/akita/v5/mem"
	"github.com/sarchlab/akita/v5/timing"
	"github.com/sarchlab/mgpusim/v5/amd/samples/runner/timingconfig/r9nano"
	"github.com/sarchlab/mgpusim/v5/amd/timing/cu"
)

// Shader-array geometry for Polaris 20 / RX 570.
// RX 570 exposes 32 CUs (2048 stream processors / 64) across 4 shader
// engines, i.e. 8 CUs per shader array. (RX 480 / polaris10 ships 36 CUs.)
const (
	NumCUPerShaderArray = 8
	NumShaderArray      = 4
)

// MakeBuilder returns a GPU builder configured for RX 570 (Polaris 20, gfx803).
//
// The microarchitecture reuses the Fiji (GCN3) timing pipeline that polaris10
// uses, with RX 570 resource counts, GDDR5 bandwidth, and clocks. The
// structural timing mechanisms (register scoreboarding, opcode-class VALU
// timing, pipelined/banked LDS, coalescing penalties, per-die dispatch, and
// separate first/subsequent launch costs) are adopted from the calibrated
// gfx90c model so the preset is structurally complete; their numeric values
// are calibrated against RX 570 hardware.
func MakeBuilder() r9nano.Builder {
	return r9nano.MakeBuilder().
		WithFreq(1244*timing.MHz). // RX 570 boost clock (base 1168)
		WithNumCUPerShaderArray(NumCUPerShaderArray).
		WithNumShaderArray(NumShaderArray).
		WithL2CacheSize(2*mem.MB). // 2 MB L2 (shared with polaris10)
		// Preserve the cache model's native 16-request/cycle bank service
		// rate. Reducing this value created a superlinear queue bottleneck
		// for large streaming inputs.
		WithL2NumReqPerCycle(16).
		WithDramSize(4*mem.GB). // 4 GB GDDR5 (8 GB variant exists)
		WithNumMemoryBank(8).   // 256-bit GDDR5 = 8 x 32-bit channels
		// L1V: 16 KB per CU. The 56-cycle modeled bank latency includes the
		// cache pipeline and calibrated contention seen by the exact kernels.
		WithL1VCacheSize(16*mem.KB).
		WithL1VBankLatency(56).
		// L2 hit latency: 50 cycles, about 40 ns at 1.244 GHz.
		WithL2BankLatency(50).
		// Banked GDDR5. Each of the eight 32-bit controller channels exposes
		// four interleaved banks so unrelated misses can overlap. A shallow
		// 250 MHz pipeline avoids the former 400 ns per-request latency that
		// double-counted random-load stalls. Aggregate sustained bandwidth is
		// enforced independently on the shared L2-to-DRAM path below.
		WithBankedDRAM(true).
		WithDRAMMemFreq(250*timing.MHz).
		WithDRAMNumInternalBanks(4).
		WithDRAMBankPipelineWidth(1).
		WithDRAMBankPipelineDepth(1).
		WithDRAMStageLatency(1).
		// A shared request token bucket allows an initial 896 KiB of cache-line
		// traffic, then limits sustained L2-to-DRAM issue to one 64-byte line
		// per GPU cycle. This retains bank-level random-miss concurrency while
		// matching the measured large-vector slope.
		WithL2ToDRAMRequestRate(1, 1).
		WithL2ToDRAMRequestBurst(14336).
		// Structural timing mechanisms adopted from the calibrated gfx90c
		// model; numeric values are calibrated against RX 570 hardware.
		WithRegisterScoreboard(true).
		WithVALUTiming(cu.VALUTiming{
			DefaultIssueInterval:         4, // 16-wide SIMD, 4 cyc/wavefront
			DefaultResultLatency:         4,
			BitwiseIssueInterval:         1,
			BitwiseResultLatency:         1,
			FMAIssueInterval:             4,
			FMAResultLatency:             4,
			IntegerMultiplyIssueInterval: 4,
			IntegerMultiplyResultLatency: 4,
			TranscendentalIssueInterval:  4,
			TranscendentalResultLatency:  16,
			// Polaris native FP64 rate is 1/16 of FP32.
			FP64IssueInterval: 16,
			FP64ResultLatency: 16,
			MaxInFlight:       4,
		}).
		WithLDSPipelineLatency(12).
		WithLDSThroughput(4, 4).
		WithLDSBanking(32, 4, 1).
		WithBarrierLatency(16).
		// Sparse reads already pay for the generated cache-line requests and
		// memory latency. An extra per-line read coalescing stall was double
		// counting pagerank's random access cost, so its cap is disabled.
		WithMaxCoalescingPenalty(0).
		// Partial lines pay a write-combine/RMW cost; dense lines pay a small
		// independent store-issue serialization cost.
		WithMaxWriteCoalescingPenalty(120).
		WithFullLineWritePenalty(12).
		// A flat_load_dwordx4 transfers 16 bytes per active lane. Charge
		// each generated cache-line transaction 126 serialization cycles to
		// model Polaris's wide-load path. This moves the exact gfx803
		// matrix kernel from 49 to 74 us of execution time, matching the
		// RX 570 steady-state measurement (73.5 us), without penalizing
		// ordinary dword streaming loads.
		WithMaxWideReadPenalty(126).
		WithMaxWideWriteStridePenalty(161).
		WithVecMemTransPipelineWidth(1).
		// Keep the model's native shared-CU request capacity. A 128-entry
		// override needlessly serialized independent pagerank misses.
		WithInFlightVectorMemAccessLimit(512).
		// Dispatch across 4 shader engines in parallel.
		// GPU-side dispatch costs only: 2380 cycles before the first kernel,
		// 1300 before later launches, and 2500 after each kernel completes.
		// Host/KFD first-use overhead is intentionally excluded; it varies
		// from 10 to 50 us in the recorded cold hipEvent measurements and
		// is not a property of GPU kernel execution.
		WithCPAlg("per-die").
		WithCPNumDies(NumShaderArray).
		WithCPWavefrontDispatchCycles(1).
		WithCPConstantKernelLaunchOverhead(2380).
		WithCPSubsequentKernelLaunchOverhead(1300).
		WithCPWGScalingThreshold(100000).
		WithCPConstantKernelOverhead(2500).
		// RX 570 is a discrete card: host DMA targets GDDR5 over PCIe and
		// mostly bypasses the GPU L2. The r9nano builder's non-L2 DMA path
		// is broken on the current HEAD (the ToMemDRAM port is declared but
		// only built in hybrid mode, panicking otherwise), so the preset
		// uses hybrid DMA. The L2-fill threshold is set to 128 KB: small
		// working sets (NW's 129×129 int matrix = 65 KB) are pre-loaded
		// into L2 via DMA, making them L2-hit on first kernel access. Larger
		// transfers (vectoradd's 256 KB arrays) bypass L2 and go direct to
		// DRAM, matching the streaming access pattern. Matrixmult's 64 KB
		// inputs are also pre-loaded; its calibrated wide-load path includes
		// that initial cache residency.
		WithDMAThroughL2(true).
		WithDMAThroughL2MaxBytes(128 * mem.KB)
}
