package cu

import (
	"testing"

	"github.com/sarchlab/mgpusim/v5/amd/insts"
)

func TestLDSBankConflictCycles(t *testing.T) {
	t.Run("contiguous b32 is conflict free", func(t *testing.T) {
		inst := insts.NewInst()
		inst.Opcode = 54

		got := LDSBankConflictCycles(
			inst,
			^uint64(0),
			func(lane int) uint32 { return uint32(lane * 4) },
			32, 4, 1,
		)
		if got != 0 {
			t.Fatalf("got %d cycles, want 0", got)
		}
	})

	t.Run("strided b32 conflicts within each half wave", func(t *testing.T) {
		inst := insts.NewInst()
		inst.Opcode = 54

		got := LDSBankConflictCycles(
			inst,
			^uint64(0),
			func(lane int) uint32 { return uint32(lane * 128) },
			32, 4, 2,
		)
		if got != 62 {
			t.Fatalf("got %d cycles, want 62", got)
		}
	})

	t.Run("broadcast uses one bank transaction", func(t *testing.T) {
		inst := insts.NewInst()
		inst.Opcode = 54

		got := LDSBankConflictCycles(
			inst,
			^uint64(0),
			func(int) uint32 { return 0 },
			32, 4, 1,
		)
		if got != 0 {
			t.Fatalf("got %d cycles, want 0", got)
		}
	})

	t.Run("b128 occupies four cycles per bank", func(t *testing.T) {
		inst := insts.NewInst()
		inst.Opcode = 255

		got := LDSBankConflictCycles(
			inst,
			^uint64(0),
			func(lane int) uint32 { return uint32(lane * 16) },
			32, 4, 1,
		)
		if got != 3 {
			t.Fatalf("got %d cycles, want 3", got)
		}
	})
}
