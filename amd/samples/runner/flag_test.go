package runner

import (
	"flag"
	"fmt"
	"testing"
)

func setVMemReturnFlags(
	t *testing.T,
	timing bool,
	gpuType string,
	loadReturn, wideLoadReturn, cuWideReturn int,
) {
	t.Helper()

	oldTiming := *timingFlag
	oldArch := *archFlag
	oldGPUType := *gpuTypeFlag
	oldLoadReturn := *vmemLoadReturnLaneDwordsPerCycleFlag
	oldWideLoadReturn := *vmemWideLoadReturnLaneDwordsPerCycleFlag
	oldCUWideReturn := *vmemCUWideReturnUnitsPerCycleFlag
	t.Cleanup(func() {
		*timingFlag = oldTiming
		*archFlag = oldArch
		*gpuTypeFlag = oldGPUType
		*vmemLoadReturnLaneDwordsPerCycleFlag = oldLoadReturn
		*vmemWideLoadReturnLaneDwordsPerCycleFlag = oldWideLoadReturn
		*vmemCUWideReturnUnitsPerCycleFlag = oldCUWideReturn
	})

	*timingFlag = timing
	*archFlag = "gcn5"
	*gpuTypeFlag = gpuType
	*vmemLoadReturnLaneDwordsPerCycleFlag = loadReturn
	*vmemWideLoadReturnLaneDwordsPerCycleFlag = wideLoadReturn
	*vmemCUWideReturnUnitsPerCycleFlag = cuWideReturn
}

func requireParseSimulationFlagsPanic(t *testing.T, want string) {
	t.Helper()

	defer func() {
		got := recover()
		if got == nil {
			t.Fatalf("expected panic %q", want)
		}
		if fmt.Sprint(got) != want {
			t.Fatalf("expected panic %q, got %q", want, got)
		}
	}()

	new(Runner).parseSimulationFlags()
}

func TestVMemCUWideReturnFlagIsRegistered(t *testing.T) {
	if flag.Lookup("vmem-cu-wide-return-units-per-cycle") == nil {
		t.Fatal("expected CU-wide vector-memory return flag to be registered")
	}
}

func TestParseVMemCUWideReturnBudget(t *testing.T) {
	setVMemReturnFlags(t, true, "gfx90c", 0, 0, 13)

	runner := new(Runner)
	runner.parseSimulationFlags()

	if runner.VMemCUWideReturnUnitsPerCycle != 13 {
		t.Fatalf(
			"expected CU-wide vector-memory return budget 13, got %d",
			runner.VMemCUWideReturnUnitsPerCycle,
		)
	}
}

func TestRejectInvalidVMemCUWideReturnFlags(t *testing.T) {
	tests := []struct {
		name       string
		timing     bool
		gpuType    string
		loadReturn int
		wideReturn int
		cuWide     int
		want       string
	}{
		{
			name:    "negative",
			timing:  true,
			gpuType: "gfx90c",
			cuWide:  -1,
			want:    "CU-wide vector-memory load return budget cannot be negative",
		},
		{
			name:    "without timing",
			gpuType: "gfx90c",
			cuWide:  13,
			want:    "CU-wide vector-memory load return budget requires -timing -gpu gfx90c",
		},
		{
			name:    "other GPU",
			timing:  true,
			gpuType: "vega64",
			cuWide:  13,
			want:    "CU-wide vector-memory load return budget requires -timing -gpu gfx90c",
		},
		{
			name:       "per-instruction model",
			timing:     true,
			gpuType:    "gfx90c",
			loadReturn: 9,
			cuWide:     13,
			want:       "vector-memory load return bandwidth models are mutually exclusive",
		},
		{
			name:       "per-wave model",
			timing:     true,
			gpuType:    "gfx90c",
			wideReturn: 11,
			cuWide:     13,
			want:       "vector-memory load return bandwidth models are mutually exclusive",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			setVMemReturnFlags(
				t,
				test.timing,
				test.gpuType,
				test.loadReturn,
				test.wideReturn,
				test.cuWide,
			)
			requireParseSimulationFlagsPanic(t, test.want)
		})
	}
}
