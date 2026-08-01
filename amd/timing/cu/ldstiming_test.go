package cu

import (
	"testing"

	"github.com/sarchlab/mgpusim/v5/amd/insts"
)

func TestLDSB128ServiceExtraCycles(t *testing.T) {
	tests := []struct {
		name   string
		inst   *insts.Inst
		extra  int
		cycles int
	}{
		{"write b128", ldsInstWithOpcode(223), 7, 7},
		{"read b128", ldsInstWithOpcode(255), 7, 7},
		{"write b64", ldsInstWithOpcode(77), 7, 0},
		{"read b64", ldsInstWithOpcode(118), 7, 0},
		{"write b32", ldsInstWithOpcode(13), 7, 0},
		{"read b32", ldsInstWithOpcode(54), 7, 0},
		{"disabled", ldsInstWithOpcode(255), 0, 0},
		{"nil instruction", nil, 7, 0},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := LDSB128ServiceExtraCycles(test.inst, test.extra)
			if got != test.cycles {
				t.Fatalf("got %d cycles, want %d", got, test.cycles)
			}
		})
	}
}

func TestLDSB128ContentionExtraCycles(t *testing.T) {
	inst := ldsInstWithOpcode(255)
	if got := LDSB128ContentionExtraCycles(inst, 9, false); got != 0 {
		t.Fatalf("uncontended got %d cycles, want 0", got)
	}
	if got := LDSB128ContentionExtraCycles(inst, 9, true); got != 9 {
		t.Fatalf("contended got %d cycles, want 9", got)
	}
	if got := LDSB128ContentionExtraCycles(
		ldsInstWithOpcode(118), 9, true,
	); got != 0 {
		t.Fatalf("B64 got %d cycles, want 0", got)
	}
}

func ldsInstWithOpcode(opcode insts.Opcode) *insts.Inst {
	inst := insts.NewInst()
	inst.Opcode = opcode
	return inst
}

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
