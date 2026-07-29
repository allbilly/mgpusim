// Package gfx90c provides a timing model for AMD GCN5 APUs (e.g. Renoir
// gfx90c) with fewer CUs and DDR system memory instead of HBM.
package gfx90c

import (
	"github.com/sarchlab/akita/v5/mem"
	"github.com/sarchlab/akita/v5/timing"
	"github.com/sarchlab/mgpusim/v5/amd/emu"
	"github.com/sarchlab/mgpusim/v5/amd/insts"
	"github.com/sarchlab/mgpusim/v5/amd/samples/runner/timingconfig/r9nano"
	"github.com/sarchlab/mgpusim/v5/amd/timing/cu"
)

// Shader-array geometry for Renoir-class gfx90c.
// Host rocminfo reports 7 CUs (one fused off); keep the full 8-CU floorplan
// (4 SA × 2 CU) so CP/SA wiring matches the calibrated GCN5 timing path.
const (
	NumCUPerShaderArray = 2
	NumShaderArray      = 4
	ActiveCUCount       = 7
)

// MakeBuilder returns a GPU builder configured for gfx90c APU iGPUs.
//
// Memory hierarchy is calibrated following the MGPUSim ISCA 2019 methodology:
// per-level bank latencies tuned with cache_latency microbenchmarks, and a
// banked DRAM model (dual DDR4-3200 channels) for bandwidth limits without
// the simulation cost of the cycle-accurate DRAM controller.
func MakeBuilder() r9nano.Builder {
	return r9nano.MakeBuilder().
		WithFreq(1600*timing.MHz). // rocminfo Max Clock on Renoir 4700U
		WithNumCUPerShaderArray(NumCUPerShaderArray).
		WithNumShaderArray(NumShaderArray).
		WithActiveCUCount(ActiveCUCount).
		WithL2CacheSize(1*mem.MB). // 1 MB shared L2
		WithDramSize(2*mem.GB).    // matches host VRAM (2 GiB)
		WithNumMemoryBank(2).      // dual-channel DDR4 (one controller per channel)
		// L1V: 16 KB per CU; ~12 ns bank latency at 1.6 GHz (cache_latency L1 plateau).
		WithL1VCacheSize(16*mem.KB).
		WithL1VBankLatency(19). // 12 ns * 1.6 GHz ≈ 19 cyc
		// L2: ~80 ns bank latency at 1.6 GHz (cache_latency L2 plateau).
		WithL2BankLatency(128).
		// Banked DDR4-3200: depth 40 → ~vectoradd matches HW (~26 µs).
		WithBankedDRAM(true).
		WithDRAMMemFreq(1*timing.GHz).
		WithDRAMNumInternalBanks(16).
		WithDRAMBankPipelineWidth(1).
		WithDRAMBankPipelineDepth(40).
		WithDRAMStageLatency(10).
		WithRegisterScoreboard(true).
		WithVALUTiming(cu.VALUTiming{
			DefaultIssueInterval:         4,
			DefaultResultLatency:         4,
			FMAIssueInterval:             4,
			FMAResultLatency:             4,
			IntegerMultiplyIssueInterval: 4,
			IntegerMultiplyResultLatency: 4,
			TranscendentalIssueInterval:  4,
			TranscendentalResultLatency:  16,
			FP64IssueInterval:            8,
			FP64ResultLatency:            8,
			MaxInFlight:                  4,
		}).
		WithLDSPipelineLatency(12).
		WithLDSThroughput(4, 4).
		WithLDSBanking(32, 4, 1).
		WithBarrierLatency(16).
		WithMaxCoalescingPenalty(12).
		// Partial-line stores pay write-combine/read-modify-write cost.
		WithMaxWriteCoalescingPenalty(103).
		// Wide non-local stores can exceed the write-combine window.
		WithMaxWideWriteStridePenalty(146).
		// Match the eight-request/cycle L1V front end.
		WithVecMemTransPipelineWidth(8).
		// Dispatch: 4 shader arrays; ~1.25 µs post-kernel tax per launch.
		WithCPAlg("per-die").
		WithCPNumDies(NumShaderArray).
		WithCPWavefrontDispatchCycles(1).
		// A cold first dispatch contributes ~2.3 us once queue submission is
		// amortized; back-to-back launches expose the full ~6.2 us queue gap.
		// Keep launch and completion costs separate so single- and
		// multi-kernel workloads scale consistently.
		WithCPConstantKernelLaunchOverhead(3750).
		WithCPSubsequentKernelLaunchOverhead(10000).
		WithCPWGScalingThreshold(128).
		WithCPConstantKernelOverhead(2000).
		// APU: small H2D transfers warm L2. Keep the two 64 KiB matrix
		// inputs resident, while the 256 KiB+ streaming buffers bypass it.
		WithDMAThroughL2(true).
		WithDMAThroughL2MaxBytes(128 * mem.KB).
		// GFX9 FLAT/GLOBAL: SADDR=0 is a valid scalar base (not OFF).
		WithDecoderBuilder(func() emu.Decoder {
			d := insts.NewDisassembler()
			d.IsCDNA3 = true
			return d
		})
}
