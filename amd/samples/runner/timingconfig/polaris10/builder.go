// Package polaris10 provides a timing model for AMD GCN4 Polaris GPUs
// (e.g. Radeon RX 480, gfx804).
package polaris10

import (
	"github.com/sarchlab/akita/v5/mem"
	"github.com/sarchlab/akita/v5/timing"
	"github.com/sarchlab/mgpusim/v5/amd/samples/runner/timingconfig/r9nano"
)

// Shader-array geometry for Polaris 10 / RX 480 (36 CUs, 4 shader engines).
const (
	NumCUPerShaderArray = 9
	NumShaderArray      = 4
)

// MakeBuilder returns a GPU builder configured for Polaris 10-class hardware.
// The microarchitecture reuses the Fiji (GCN3) timing pipeline with Polaris
// resource counts, GDDR5 memory, and clocks; the ISA is executed by the GCN3 ALU.
func MakeBuilder() r9nano.Builder {
	return r9nano.MakeBuilder().
		WithFreq(1266 * timing.MHz). // RX 480 boost clock
		WithNumCUPerShaderArray(NumCUPerShaderArray).
		WithNumShaderArray(NumShaderArray).
		WithL2CacheSize(2 * mem.MB). // 2 MB L2
		WithDramSize(8 * mem.GB).    // 8 GB GDDR5
		WithNumMemoryBank(8).        // 256-bit GDDR5 (8 x 32-bit channels)
		// L1V: 16 KB per CU, ~14 ns bank latency at 1.266 GHz.
		WithL1VCacheSize(16 * mem.KB).
		WithL1VBankLatency(18).
		// L2: ~75 ns bank latency at 1.266 GHz.
		WithL2BankLatency(95).
		// Banked GDDR5: 8 x 32-bit channels, ~256 GB/s aggregate peak.
		WithBankedDRAM(true).
		WithDRAMMemFreq(1 * timing.GHz).
		WithDRAMNumInternalBanks(16).
		WithDRAMBankPipelineWidth(1).
		WithDRAMBankPipelineDepth(32).
		WithDRAMStageLatency(6).
		// Dispatch across 4 shader engines in parallel.
		WithCPAlg("per-die").
		WithCPNumDies(NumShaderArray).
		WithCPWavefrontDispatchCycles(1).
		WithCPConstantKernelOverhead(0)
}
