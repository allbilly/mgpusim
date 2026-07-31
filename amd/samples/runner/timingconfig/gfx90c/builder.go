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
		// Separate effective hit latency from sustained per-bank request rate.
		WithL2BankLatency(64).
		// Banked DDR4-2666, 128-bit (2x64): host dmesg reports "RAM width
		// 128bits DDR4", mclk max 1333 MHz = DDR4-2666. Peak BW = 42.7 GB/s.
		// With 2 channels, 64B lines, width=1: freq = 42.7/(2*64) = 333 MHz.
		// APU round-trip (SoC fabric + MMC + DRAM) is about 400 ns. A
		// 9-stage, 14-cycle pipeline gives 126 cycles, or about 378 ns, while
		// retaining the lower sustained rate exposed by multi-stream sweeps.
		WithBankedDRAM(true).
		WithDRAMMemFreq(333*timing.MHz).
		WithDRAMNumInternalBanks(16).
		WithDRAMBankPipelineWidth(1).
		WithDRAMBankPipelineDepth(9).
		WithDRAMStageLatency(14).
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
		WithLDSPipelineLatency(4).
		WithLDSThroughput(1, 4).
		WithLDSBanking(32, 4, 1).
		WithBarrierLatency(4).
		WithMaxCoalescingPenalty(13).
		WithSplitLineLoadPenalty(22).
		WithSplitLineLoadMaxDwords(2).
		WithDependentLoadIssueWindow(6000, 3, 3).
		WithDependentLoadMaxDwords(2).
		WithDependentLoadFlatOnly(true).
		// Partial-line stores pay write-combine/read-modify-write cost.
		WithMaxWriteCoalescingPenalty(103).
		// Wide non-local stores can exceed the write-combine window.
		WithMaxWideWriteStridePenalty(240).
		// The transpose size sweep shows an extra cost once transaction gaps
		// reach 4 KiB; keep this as an empirical far-stride tier.
		WithMaxWideWriteStrideFarPenalty(270, 64).
		// The CU-to-cache issue path is shared and admits one coalesced
		// transaction group per cycle.
		WithVecMemTransPipelineWidth(1).
		// Dispatch: 4 shader arrays; ~0.91 µs post-kernel tax per launch.
		WithCPAlg("per-die").
		WithCPNumDies(NumShaderArray).
		WithCPWavefrontDispatchCycles(1).
		// A cold first dispatch contributes ~2.3 us once queue submission is
		// amortized; back-to-back launches expose the full ~6.2 us queue gap.
		// Keep launch and completion costs separate so single- and
		// multi-kernel workloads scale consistently.
		WithCPConstantKernelLaunchOverhead(3750).
		WithCPSubsequentKernelLaunchOverhead(7500).
		WithCPWGScalingThreshold(128).
		WithCPConstantKernelOverhead(1450).
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
