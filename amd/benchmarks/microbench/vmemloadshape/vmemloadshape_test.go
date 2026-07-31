package vmemloadshape

import "testing"

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
		{2, ModeIndependent4, "vmem_load_dwordx2_independent4"},
		{4, ModeSerial, "vmem_load_dwordx4_serial"},
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

func TestRejectsInvalidConfigurations(t *testing.T) {
	invalid := []*Benchmark{
		{WidthDwords: 3},
		{WidthDwords: 4, Mode: "other"},
		{WidthDwords: 4, Mode: ModeSerial, AliasLanes: 3},
		{WidthDwords: 4, Mode: ModeSerial, AliasLanes: 8, ArrayBytes: 12 * 1024},
		{WidthDwords: 4, Mode: ModeIndependent4, AliasLanes: 8, ArrayBytes: 8 * 1024, Repeats: 3},
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
