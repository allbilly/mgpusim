// Package vega64 provides a timing model for AMD Vega-class GCN5 GPUs
// (e.g. Radeon Vega 64, gfx900).
package vega64

import (
	"github.com/sarchlab/akita/v5/mem"
	"github.com/sarchlab/akita/v5/timing"
	"github.com/sarchlab/mgpusim/v5/amd/samples/runner/timingconfig/r9nano"
)

// Shader-array geometry for Vega 64 (64 CUs, 4 CUs per shader array).
const (
	NumCUPerShaderArray = 4
	NumShaderArray      = 16
)

// MakeBuilder returns a GPU builder configured for Vega 64-class hardware.
// The microarchitecture reuses the Fiji (GCN3) timing pipeline with Vega
// resource counts and clocks; the ISA is executed by the GCN3 ALU.
func MakeBuilder() r9nano.Builder {
	return r9nano.MakeBuilder().
		WithFreq(1248 * timing.MHz). // Vega 64 base engine clock
		WithNumCUPerShaderArray(NumCUPerShaderArray).
		WithNumShaderArray(NumShaderArray).
		WithL2CacheSize(4 * mem.MB). // 4 MB L2
		WithDramSize(8 * mem.GB).    // 8 GB HBM2
		WithNumMemoryBank(16)
}
