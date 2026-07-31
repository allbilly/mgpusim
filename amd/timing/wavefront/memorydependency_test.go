package wavefront

import (
	"testing"

	"github.com/sarchlab/mgpusim/v5/amd/insts"
)

func TestRecentMemoryAddressDependency(t *testing.T) {
	wf := NewWavefront(nil)
	wf.EnableMemoryDependencyTracking()
	wf.SetEXEC(^uint64(0))

	load := insts.NewInst()
	load.ExeUnit = insts.ExeUnitVMem
	load.FormatType = insts.FLAT
	load.Opcode = 20
	load.Dst = insts.NewVRegOperand(0, 1, 1)
	wf.TrackIssuedInstruction(load)
	wf.MarkMemoryLoadDestination(load)

	addressALU := insts.NewInst()
	addressALU.ExeUnit = insts.ExeUnitVALU
	addressALU.Src0 = insts.NewVRegOperand(0, 1, 1)
	addressALU.Dst = insts.NewVRegOperand(0, 2, 2)
	wf.TrackIssuedInstruction(addressALU)

	dependentLoad := insts.NewInst()
	dependentLoad.ExeUnit = insts.ExeUnitVMem
	dependentLoad.Addr = insts.NewVRegOperand(0, 2, 2)
	dependentLoad.Dst = insts.NewVRegOperand(0, 4, 1)

	if !wf.AddressDependsOnRecentLoad(dependentLoad, 1) {
		t.Fatal("expected the propagated load-to-address dependency")
	}
	if wf.AddressDependsOnRecentLoad(dependentLoad, 0) {
		t.Fatal("dependency should not have zero age")
	}
	if !wf.AddressDependsOnLoadInWindow(dependentLoad, 1, 1) {
		t.Fatal("dependency should match its exact age window")
	}
	if wf.AddressDependsOnLoadInWindow(dependentLoad, 2, 12) {
		t.Fatal("dependency should be younger than the minimum age")
	}
}

func TestOverwritingAddressRegisterClearsMemoryDependency(t *testing.T) {
	wf := NewWavefront(nil)
	wf.EnableMemoryDependencyTracking()
	wf.SetEXEC(^uint64(0))
	wf.MemoryDependency.MemoryDerivedVGPR[2] = 1
	wf.MemoryDependency.InstructionIssueSequence = 1

	overwrite := insts.NewInst()
	overwrite.ExeUnit = insts.ExeUnitVALU
	overwrite.Src0 = insts.NewIntOperand(0, 7)
	overwrite.Dst = insts.NewVRegOperand(0, 2, 1)
	wf.TrackIssuedInstruction(overwrite)

	load := insts.NewInst()
	load.ExeUnit = insts.ExeUnitVMem
	load.Addr = insts.NewVRegOperand(0, 2, 1)
	wf.TrackIssuedInstruction(load)
	if wf.AddressDependsOnRecentLoad(load, 16) {
		t.Fatal("overwritten address register retained stale memory provenance")
	}
}

func TestVectorStoreDoesNotClearAddressProvenance(t *testing.T) {
	wf := NewWavefront(nil)
	wf.EnableMemoryDependencyTracking()
	wf.SetEXEC(^uint64(0))
	wf.MemoryDependency.MemoryDerivedVGPR[2] = 1
	wf.MemoryDependency.InstructionIssueSequence = 1

	store := insts.NewInst()
	store.ExeUnit = insts.ExeUnitVMem
	store.FormatType = insts.FLAT
	store.Opcode = 28
	store.Addr = insts.NewVRegOperand(0, 2, 1)
	store.Data = insts.NewVRegOperand(0, 8, 1)
	wf.TrackIssuedInstruction(store)

	load := insts.NewInst()
	load.ExeUnit = insts.ExeUnitVMem
	load.FormatType = insts.FLAT
	load.Opcode = 20
	load.Addr = insts.NewVRegOperand(0, 2, 1)
	if !wf.AddressDependsOnRecentLoad(load, 1) {
		t.Fatal("a vector store cleared address provenance")
	}
}

func TestPartialExecConservativelyPreservesProvenance(t *testing.T) {
	wf := NewWavefront(nil)
	wf.EnableMemoryDependencyTracking()
	wf.SetEXEC(1)
	wf.MemoryDependency.MemoryDerivedVGPR[2] = 1
	wf.MemoryDependency.InstructionIssueSequence = 1

	overwrite := insts.NewInst()
	overwrite.ExeUnit = insts.ExeUnitVALU
	overwrite.Src0 = insts.NewIntOperand(0, 7)
	overwrite.Dst = insts.NewVRegOperand(0, 2, 1)
	wf.TrackIssuedInstruction(overwrite)

	load := insts.NewInst()
	load.ExeUnit = insts.ExeUnitVMem
	load.FormatType = insts.FLAT
	load.Opcode = 20
	load.Addr = insts.NewVRegOperand(0, 2, 1)
	if !wf.AddressDependsOnRecentLoad(load, 1) {
		t.Fatal("partial EXEC lost provenance from inactive lanes")
	}
}

func TestDependencyStallRearmsAtTheSamePCAfterIssue(t *testing.T) {
	wf := NewWavefront(nil)
	wf.EnableMemoryDependencyTracking()
	wf.SetEXEC(^uint64(0))
	wf.SetPC(0x100)
	wf.MemoryDependency.MemoryDerivedVGPR[2] = 1
	wf.MemoryDependency.InstructionIssueSequence = 1

	load := insts.NewInst()
	load.ExeUnit = insts.ExeUnitVMem
	load.FormatType = insts.FLAT
	load.Opcode = 20
	load.Addr = insts.NewVRegOperand(0, 2, 1)
	load.Dst = insts.NewVRegOperand(0, 4, 1)

	if !wf.StallRecentLoadAddress(load, 1, 12) {
		t.Fatal("expected the first dynamic load to stall")
	}
	if wf.StallRecentLoadAddress(load, 1, 12) {
		t.Fatal("the first dynamic load stalled too many cycles")
	}
	wf.TrackIssuedInstruction(load)
	if !wf.StallRecentLoadAddress(load, 1, 12) {
		t.Fatal("same-PC loop iteration did not re-arm the dependency stall")
	}
}

func TestDependentVectorStoreDoesNotStall(t *testing.T) {
	wf := NewWavefront(nil)
	wf.EnableMemoryDependencyTracking()
	wf.SetEXEC(^uint64(0))
	wf.MemoryDependency.MemoryDerivedVGPR[2] = 1
	wf.MemoryDependency.InstructionIssueSequence = 1

	store := insts.NewInst()
	store.ExeUnit = insts.ExeUnitVMem
	store.FormatType = insts.FLAT
	store.Opcode = 28
	store.Addr = insts.NewVRegOperand(0, 2, 1)
	if wf.StallRecentLoadAddress(store, 10, 12) {
		t.Fatal("load-to-address penalty must not apply to a store")
	}
}

func TestZeroExecLoadDoesNotStallOrDisarmSamePCReentry(t *testing.T) {
	wf := NewWavefront(nil)
	wf.EnableMemoryDependencyTracking()
	wf.SetPC(0x100)
	wf.MemoryDependency.MemoryDerivedVGPR[2] = 1
	wf.MemoryDependency.InstructionIssueSequence = 1

	load := insts.NewInst()
	load.ExeUnit = insts.ExeUnitVMem
	load.FormatType = insts.FLAT
	load.Opcode = 20
	load.Addr = insts.NewVRegOperand(0, 2, 1)
	load.Dst = insts.NewVRegOperand(0, 4, 1)

	wf.SetEXEC(0)
	if wf.StallRecentLoadAddress(load, 10, 12) {
		t.Fatal("EXEC=0 load should not stall")
	}
	wf.TrackIssuedInstruction(load)

	wf.SetEXEC(^uint64(0))
	if !wf.StallRecentLoadAddress(load, 10, 12) {
		t.Fatal("active same-PC re-entry was left disarmed")
	}
}

func TestDependentLoadWidthLimit(t *testing.T) {
	wf := NewWavefront(nil)
	wf.EnableMemoryDependencyTracking()
	wf.SetEXEC(^uint64(0))
	wf.MemoryDependency.MemoryDerivedVGPR[2] = 1
	wf.MemoryDependency.InstructionIssueSequence = 1

	load := insts.NewInst()
	load.ExeUnit = insts.ExeUnitVMem
	load.FormatType = insts.FLAT
	load.Opcode = 23
	load.Addr = insts.NewVRegOperand(0, 2, 1)
	load.Dst = insts.NewVRegOperand(0, 8, 4)
	if wf.StallNarrowRecentLoadAddressInWindow(load, 10, 0, 12, 2) {
		t.Fatal("four-dword load exceeded the configured width limit")
	}

	load.FormatType = insts.MUBUF
	load.Opcode = 20
	load.Dst = insts.NewVRegOperand(0, 8, 1)
	if wf.StallFilteredRecentLoadAddressInWindow(
		load, 10, 0, 12, 2, true,
	) {
		t.Fatal("MUBUF load passed a FLAT-only dependency filter")
	}

	load.FormatType = insts.FLAT
	load.Opcode = 20
	load.Dst = insts.NewVRegOperand(0, 8, 1)
	if !wf.StallNarrowRecentLoadAddressInWindow(load, 10, 0, 12, 2) {
		t.Fatal("one-dword load should satisfy the width limit")
	}
}

func TestDecodedGlobalLoadWidth(t *testing.T) {
	decoder := insts.NewDisassembler()
	decoder.IsCDNA3 = true
	// global_load_dwordx4 v[16:19], v[20:21], off
	inst, err := decoder.Decode([]byte{
		0x00, 0x80, 0x5c, 0xdc,
		0x14, 0x00, 0x7f, 0x10,
	})
	if err != nil {
		t.Fatal(err)
	}
	if inst.Dst == nil || inst.Dst.RegCount != 4 {
		t.Fatalf("decoded dwordx4 width = %v, want 4", inst.Dst)
	}
}
