package emu

import (
	"github.com/sarchlab/mgpusim/v5/amd/insts"
)

// UsesPackedWorkItemIDs reports whether work-item IDs should be packed into
// v0 as (z<<20)|(y<<10)|x. CDNA3 HIP kernels use this layout; GCN3/GCN4/GCN5-class
// kernels loaded from V5 code objects still expect separate v0/v1/v2.
func UsesPackedWorkItemIDs(co *insts.KernelCodeObject, alu ALU) bool {
	if co == nil || co.Version != insts.CodeObjectV5 {
		return false
	}
	return alu != nil && alu.ArchName() == "CDNA3"
}
