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
	loadReturn, wideLoadReturn, cuWideReturn, cuWideWaves, cuWideBurstWaves int,
) {
	t.Helper()

	oldTiming := *timingFlag
	oldArch := *archFlag
	oldGPUType := *gpuTypeFlag
	oldLoadReturn := *vmemLoadReturnLaneDwordsPerCycleFlag
	oldWideLoadReturn := *vmemWideLoadReturnLaneDwordsPerCycleFlag
	oldCUWideReturn := *vmemCUWideReturnUnitsPerCycleFlag
	oldCUWideWaves := *vmemCUWideReturnConcurrentWavesFlag
	oldCUWideBurstWaves := *vmemCUWideReturnBurstConcurrentWavesFlag
	t.Cleanup(func() {
		*timingFlag = oldTiming
		*archFlag = oldArch
		*gpuTypeFlag = oldGPUType
		*vmemLoadReturnLaneDwordsPerCycleFlag = oldLoadReturn
		*vmemWideLoadReturnLaneDwordsPerCycleFlag = oldWideLoadReturn
		*vmemCUWideReturnUnitsPerCycleFlag = oldCUWideReturn
		*vmemCUWideReturnConcurrentWavesFlag = oldCUWideWaves
		*vmemCUWideReturnBurstConcurrentWavesFlag = oldCUWideBurstWaves
	})

	*timingFlag = timing
	*archFlag = "gcn5"
	*gpuTypeFlag = gpuType
	*vmemLoadReturnLaneDwordsPerCycleFlag = loadReturn
	*vmemWideLoadReturnLaneDwordsPerCycleFlag = wideLoadReturn
	*vmemCUWideReturnUnitsPerCycleFlag = cuWideReturn
	*vmemCUWideReturnConcurrentWavesFlag = cuWideWaves
	*vmemCUWideReturnBurstConcurrentWavesFlag = cuWideBurstWaves
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
	if flag.Lookup("vmem-cu-wide-return-concurrent-waves") == nil {
		t.Fatal("expected CU-wide vector-memory concurrent-wave flag to be registered")
	}
	if flag.Lookup("vmem-cu-wide-return-burst-concurrent-waves") == nil {
		t.Fatal("expected CU-wide vector-memory burst flag to be registered")
	}
}

func TestParseVMemCUWideReturnBudget(t *testing.T) {
	setVMemReturnFlags(t, true, "gfx90c", 0, 0, 13, 3, 0)

	runner := new(Runner)
	runner.parseSimulationFlags()

	if runner.VMemCUWideReturnUnitsPerCycle != 13 {
		t.Fatalf(
			"expected CU-wide vector-memory return budget 13, got %d",
			runner.VMemCUWideReturnUnitsPerCycle,
		)
	}
	if runner.VMemCUWideReturnConcurrentWaves != 3 {
		t.Fatalf(
			"expected CU-wide concurrent waves 3, got %d",
			runner.VMemCUWideReturnConcurrentWaves,
		)
	}
}

func TestDefaultVMemCUWideReturnConcurrentWavesToOne(t *testing.T) {
	setVMemReturnFlags(t, true, "gfx90c", 0, 0, 13, 0, 0)

	runner := new(Runner)
	runner.parseSimulationFlags()

	if runner.VMemCUWideReturnConcurrentWaves != 1 {
		t.Fatalf(
			"expected CU-wide concurrent waves to default to 1, got %d",
			runner.VMemCUWideReturnConcurrentWaves,
		)
	}
}

func TestParseVMemCUWideBurstConcurrentWaves(t *testing.T) {
	setVMemReturnFlags(t, true, "gfx90c", 0, 0, 13, 0, 2)

	runner := new(Runner)
	runner.parseSimulationFlags()

	if runner.VMemCUWideReturnConcurrentWaves != 0 {
		t.Fatalf(
			"burst mode must not normalize static waves, got %d",
			runner.VMemCUWideReturnConcurrentWaves,
		)
	}
	if runner.VMemCUWideReturnBurstConcurrentWaves != 2 {
		t.Fatalf(
			"expected burst concurrent waves 2, got %d",
			runner.VMemCUWideReturnBurstConcurrentWaves,
		)
	}
}

func TestRejectInvalidVMemCUWideReturnFlags(t *testing.T) {
	tests := []struct {
		name        string
		timing      bool
		gpuType     string
		loadReturn  int
		wideReturn  int
		cuWide      int
		cuWideWaves int
		burstWaves  int
		want        string
	}{
		{
			name:    "negative",
			timing:  true,
			gpuType: "gfx90c",
			cuWide:  -1,
			want:    "CU-wide vector-memory load return budget cannot be negative",
		},
		{
			name:        "negative concurrent waves",
			timing:      true,
			gpuType:     "gfx90c",
			cuWide:      13,
			cuWideWaves: -1,
			want:        "CU-wide vector-memory concurrent waves cannot be negative",
		},
		{
			name:        "concurrent waves without model",
			timing:      true,
			gpuType:     "gfx90c",
			cuWideWaves: 2,
			want:        "CU-wide vector-memory concurrent waves requires the CU-wide return model",
		},
		{
			name:       "negative burst waves",
			timing:     true,
			gpuType:    "gfx90c",
			cuWide:     13,
			burstWaves: -1,
			want:       "CU-wide vector-memory burst concurrent waves cannot be negative",
		},
		{
			name:       "burst waves without model",
			timing:     true,
			gpuType:    "gfx90c",
			burstWaves: 2,
			want:       "CU-wide vector-memory burst concurrent waves requires the CU-wide return model",
		},
		{
			name:        "burst and static waves",
			timing:      true,
			gpuType:     "gfx90c",
			cuWide:      13,
			cuWideWaves: 2,
			burstWaves:  2,
			want:        "static and burst CU-wide concurrent-wave controls are mutually exclusive",
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
				test.cuWideWaves,
				test.burstWaves,
			)
			requireParseSimulationFlagsPanic(t, test.want)
		})
	}
}
