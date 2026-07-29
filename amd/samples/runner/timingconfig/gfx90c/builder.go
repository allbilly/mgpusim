// Package gfx90c provides a timing model for AMD GCN5 APUs (e.g. Renoir
// gfx90c) with fewer CUs and DDR system memory instead of HBM.
package gfx90c

import (
	"github.com/sarchlab/akita/v5/mem"
	"github.com/sarchlab/akita/v5/timing"
	"github.com/sarchlab/mgpusim/v5/amd/emu"
	"github.com/sarchlab/mgpusim/v5/amd/insts"
	"github.com/sarchlab/mgpusim/v5/amd/samples/runner/timingconfig/r9nano"
)

// Shader-array geometry for Renoir-class gfx90c.
// Host rocminfo reports 7 CUs (one fused off); keep the full 8-CU floorplan
// (4 SA × 2 CU) so CP/SA wiring matches the calibrated GCN5 timing path.
const (
	NumCUPerShaderArray = 2
	NumShaderArray      = 4
)

// MakeBuilder returns a GPU builder configured for gfx90c APU iGPUs.
//
// Memory hierarchy is calibrated following the MGPUSim ISCA 2019 methodology:
// per-level bank latencies tuned with cache_latency microbenchmarks, and a
// banked DRAM model (dual DDR4-3200 channels) for bandwidth limits without
// the simulation cost of the cycle-accurate DRAM controller.
func MakeBuilder() r9nano.Builder {
	return r9nano.MakeBuilder().
		WithFreq(1600 * timing.MHz). // rocminfo Max Clock on Renoir 4700U
		WithNumCUPerShaderArray(NumCUPerShaderArray).
		WithNumShaderArray(NumShaderArray).
		WithL2CacheSize(1 * mem.MB). // 1 MB shared L2
		WithDramSize(2 * mem.GB).    // matches host VRAM (2 GiB)
		WithNumMemoryBank(2).        // dual-channel DDR4 (one controller per channel)
		// L1V: 16 KB per CU; ~12 ns bank latency at 1.6 GHz (cache_latency L1 plateau).
		WithL1VCacheSize(16 * mem.KB).
		WithL1VBankLatency(19). // 12 ns * 1.6 GHz ≈ 19 cyc
		// L2: ~80 ns bank latency at 1.6 GHz (cache_latency L2 plateau).
		WithL2BankLatency(128).
		// Banked DDR4-3200: depth 40 → ~vectoradd matches HW (~26 µs).
		WithBankedDRAM(true).
		WithDRAMMemFreq(1 * timing.GHz).
		WithDRAMNumInternalBanks(16).
		WithDRAMBankPipelineWidth(1).
		WithDRAMBankPipelineDepth(40).
		WithDRAMStageLatency(10).
		WithRegisterScoreboard(true).
		WithScoreboardVALULatency(8).
		WithLDSPipelineLatency(12).
		// Dispatch: 4 shader arrays; ~1.25 µs post-kernel tax per launch.
		WithCPAlg("per-die").
		WithCPNumDies(NumShaderArray).
		WithCPWavefrontDispatchCycles(1).
		WithCPConstantKernelOverhead(2000).
		// APU: only small H2D (<64 KB) warms L2; ISCA-10 buffers hit DRAM.
		WithDMAThroughL2(true).
		WithDMAThroughL2MaxBytes(64 * mem.KB).
		// GFX9 FLAT/GLOBAL: SADDR=0 is a valid scalar base (not OFF).
		WithDecoderBuilder(func() emu.Decoder {
			d := insts.NewDisassembler()
			d.IsCDNA3 = true
			return d
		})
}