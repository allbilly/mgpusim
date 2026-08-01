package vmemloadshape

import (
	"math"
	"testing"
)

func TestDefaults(t *testing.T) {
	b := &Benchmark{}
	b.setDefaultsAndValidate()
	if b.WidthDwords != 4 || b.Mode != ModeSerial || b.AliasLanes != 8 ||
		b.ArrayBytes != 8*1024 || b.Repeats != 1024 || b.Workgroups != 16 {
		t.Fatalf("unexpected defaults: %+v", b)
	}
}

func TestSymbols(t *testing.T) {
	tests := []struct {
		width int
		mode  string
		want  string
	}{
		{1, ModeSerial, "vmem_load_dword_serial"},
		{1, ModeIndependent2, "vmem_load_dword_independent2"},
		{1, ModeIndependent4, "vmem_load_dword_independent4"},
		{1, ModeIndependent8, "vmem_load_dword_independent8"},
		{2, ModeSerial, "vmem_load_dwordx2_serial"},
		{2, ModeIndependent2, "vmem_load_dwordx2_independent2"},
		{2, ModeIndependent4, "vmem_load_dwordx2_independent4"},
		{2, ModeIndependent8, "vmem_load_dwordx2_independent8"},
		{4, ModeSerial, "vmem_load_dwordx4_serial"},
		{4, ModeIndependent2, "vmem_load_dwordx4_independent2"},
		{4, ModeIndependent4, "vmem_load_dwordx4_independent4"},
		{4, ModeIndependent8, "vmem_load_dwordx4_independent8"},
	}
	for _, test := range tests {
		b := &Benchmark{WidthDwords: test.width, Mode: test.mode}
		if got := b.symbol(); got != test.want {
			t.Fatalf("symbol %q, want %q", got, test.want)
		}
	}
}

func TestReferencePreservesAliasing(t *testing.T) {
	b := &Benchmark{WidthDwords: 4, Mode: ModeSerial, AliasLanes: 8,
		ArrayBytes: 8 * 1024, Repeats: 16, Workgroups: 1}
	b.setDefaultsAndValidate()
	b.initHostData()
	if got, want := b.expectedAt(0), b.expectedAt(56); got != want {
		t.Fatalf("interleaved aliases disagree: lane 0=%g lane 56=%g", got, want)
	}
	if got, want := b.expectedAt(0), b.expectedAt(1); got == want {
		t.Fatalf("adjacent address groups unexpectedly agree: %g", got)
	}
}

func TestExplicitZeroRepeatsIsZeroTrip(t *testing.T) {
	for _, mode := range []string{
		ModeSerial, ModeIndependent2, ModeIndependent4, ModeIndependent8,
	} {
		b := &Benchmark{WidthDwords: 4, Mode: mode, AliasLanes: 8,
			ArrayBytes: 8 * 1024, Repeats: 0, RepeatsSpecified: true,
			Workgroups: 1}
		b.setDefaultsAndValidate()
		b.initHostData()
		if b.Repeats != 0 {
			t.Fatalf("%s explicit zero became %d repeats", mode, b.Repeats)
		}
		for tid := range b.output {
			if got := b.expectedAt(tid); got != 0 {
				t.Fatalf("%s zero-trip lane %d expected %g, want zero", mode, tid, got)
			}
		}
	}
}

func TestReferenceIsIdenticalAcrossWindowsAtEqualRepeats(t *testing.T) {
	modes := []string{
		ModeSerial, ModeIndependent2, ModeIndependent4, ModeIndependent8,
	}
	var reference []float32
	for _, mode := range modes {
		b := &Benchmark{WidthDwords: 4, Mode: mode, AliasLanes: 8,
			ArrayBytes: 8 * 1024, Repeats: 16, Workgroups: 2}
		b.setDefaultsAndValidate()
		b.initHostData()
		got := make([]float32, len(b.output))
		for tid := range got {
			got[tid] = b.expectedAt(tid)
		}
		if reference == nil {
			reference = got
			continue
		}
		for tid := range got {
			if got[tid] != reference[tid] {
				t.Fatalf(
					"%s lane %d output %g, want %g",
					mode, tid, got[tid], reference[tid],
				)
			}
		}
	}
}

func TestOutputMatchesRequiresExactFiniteZeroForZeroTrip(t *testing.T) {
	if !outputMatches(0, 0, true) {
		t.Fatal("exact zero did not match zero-trip output")
	}
	for _, got := range []float32{1, -1, float32(math.NaN()), float32(math.Inf(1))} {
		if outputMatches(got, 0, true) {
			t.Fatalf("zero-trip output %g unexpectedly matched", got)
		}
	}
}

func TestRejectsInvalidConfigurations(t *testing.T) {
	invalid := []*Benchmark{
		{WidthDwords: 3},
		{WidthDwords: 4, Mode: "other"},
		{WidthDwords: 4, Mode: ModeSerial, AliasLanes: 3},
		{WidthDwords: 4, Mode: ModeSerial, AliasLanes: 8, ArrayBytes: 12 * 1024},
		{WidthDwords: 4, Mode: ModeIndependent2, AliasLanes: 8, ArrayBytes: 8 * 1024, Repeats: 3},
		{WidthDwords: 4, Mode: ModeIndependent4, AliasLanes: 8, ArrayBytes: 8 * 1024, Repeats: 6},
		{WidthDwords: 4, Mode: ModeIndependent8, AliasLanes: 8, ArrayBytes: 8 * 1024, Repeats: 12},
		{WidthDwords: 4, Mode: ModeSerial, AliasLanes: 8, ArrayBytes: 8 * 1024, Repeats: -1},
	}
	for _, b := range invalid {
		assertPanics(t, b.setDefaultsAndValidate)
	}
}

func assertPanics(t *testing.T, action func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	action()
}
