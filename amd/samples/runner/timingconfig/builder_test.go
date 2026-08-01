package timingconfig

import (
	"testing"

	"github.com/sarchlab/akita/v5/simulation"
	"github.com/sarchlab/mgpusim/v5/amd/timing/cu"
)

// buildPlatform assembles a platform of the given GPU type and count. The
// assembly exercises the port naming, the port assignment, and the mapper
// snapshotting of all the components, all of which panic on error.
func buildPlatform(t *testing.T, gpuType string, numGPUs int) {
	t.Helper()

	s := simulation.MakeBuilder().
		WithoutMonitoring().
		WithOutputFileName(t.TempDir() + "/sim").
		Build()
	defer s.Terminate()

	gpuDriver := MakeBuilder().
		WithSimulation(s).
		WithNumGPUs(numGPUs).
		WithGPUType(gpuType).
		Build()

	if gpuDriver == nil {
		t.Fatal("driver must not be nil")
	}

	if len(gpuDriver.GPUs) != numGPUs {
		t.Fatalf("expected %d GPUs, got %d", numGPUs, len(gpuDriver.GPUs))
	}

	if s.GetComponentByName("Driver") == nil {
		t.Fatal("driver must be registered with the simulation")
	}

	cpName := "GPU[1].CommandProcessor"
	if s.GetComponentByName(cpName) == nil {
		t.Fatalf("component %s must be registered", cpName)
	}
}

func TestBuildR9NanoPlatform(t *testing.T) {
	buildPlatform(t, "r9nano", 1)
}

func TestBuildR9NanoMultiGPUPlatform(t *testing.T) {
	buildPlatform(t, "r9nano", 2)
}

func TestBuildMI300XPlatform(t *testing.T) {
	buildPlatform(t, "mi300x", 1)
}

func TestBuildVega64Platform(t *testing.T) {
	buildPlatform(t, "vega64", 1)
}

func TestBuildGfx90cPlatform(t *testing.T) {
	buildPlatform(t, "gfx90c", 1)
}

func TestBuildGfx90cPlatformWithVMemLoadReturnBandwidth(t *testing.T) {
	s := simulation.MakeBuilder().
		WithoutMonitoring().
		WithOutputFileName(t.TempDir() + "/sim").
		Build()
	defer s.Terminate()

	MakeBuilder().
		WithSimulation(s).
		WithGPUType("gfx90c").
		WithVMemLoadReturnLaneDwordsPerCycle(9).
		Build()

	component := s.GetComponentByName("GPU[1].SA[0].CU[0]")
	computeUnit, ok := component.(*cu.Comp)
	if !ok {
		t.Fatalf("expected gfx90c compute unit, got %T", component)
	}
	if computeUnit.Spec().VMemLoadReturnLaneDwordsPerCycle != 9 {
		t.Fatalf(
			"expected vector-memory load return bandwidth 9, got %d",
			computeUnit.Spec().VMemLoadReturnLaneDwordsPerCycle,
		)
	}
}

func TestBuildGfx90cPlatformWithVMemWideLoadReturnBandwidth(t *testing.T) {
	s := simulation.MakeBuilder().
		WithoutMonitoring().
		WithOutputFileName(t.TempDir() + "/sim").
		Build()
	defer s.Terminate()

	MakeBuilder().
		WithSimulation(s).
		WithGPUType("gfx90c").
		WithVMemWideLoadReturnLaneDwordsPerCycle(11).
		Build()

	component := s.GetComponentByName("GPU[1].SA[0].CU[0]")
	computeUnit, ok := component.(*cu.Comp)
	if !ok {
		t.Fatalf("expected gfx90c compute unit, got %T", component)
	}
	if computeUnit.Spec().VMemWideLoadReturnLaneDwordsPerCycle != 11 {
		t.Fatalf(
			"expected wave-wide vector-memory load return bandwidth 11, got %d",
			computeUnit.Spec().VMemWideLoadReturnLaneDwordsPerCycle,
		)
	}
}

func TestBuildGfx90cPlatformWithVMemCUWideReturnBudget(t *testing.T) {
	s := simulation.MakeBuilder().
		WithoutMonitoring().
		WithOutputFileName(t.TempDir() + "/sim").
		Build()
	defer s.Terminate()

	MakeBuilder().
		WithSimulation(s).
		WithGPUType("gfx90c").
		WithVMemCUWideReturnUnitsPerCycle(13).
		WithVMemCUWideReturnConcurrentWaves(3).
		Build()

	component := s.GetComponentByName("GPU[1].SA[0].CU[0]")
	computeUnit, ok := component.(*cu.Comp)
	if !ok {
		t.Fatalf("expected gfx90c compute unit, got %T", component)
	}
	if computeUnit.Spec().VMemCUWideReturnUnitsPerCycle != 13 {
		t.Fatalf(
			"expected CU-wide vector-memory return budget 13, got %d",
			computeUnit.Spec().VMemCUWideReturnUnitsPerCycle,
		)
	}
	if computeUnit.Spec().VMemCUWideReturnConcurrentWaves != 3 {
		t.Fatalf(
			"expected CU-wide concurrent waves 3, got %d",
			computeUnit.Spec().VMemCUWideReturnConcurrentWaves,
		)
	}
}

func TestRejectSimultaneousVMemLoadReturnBandwidthModels(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected simultaneous return models to panic")
		}
	}()

	MakeBuilder().
		WithGPUType("gfx90c").
		WithVMemLoadReturnLaneDwordsPerCycle(9).
		WithVMemWideLoadReturnLaneDwordsPerCycle(11).
		Build()
}

func TestRejectVMemLoadReturnBandwidthForOtherGPUs(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected unsupported GPU configuration to panic")
		}
	}()

	MakeBuilder().
		WithGPUType("vega64").
		WithVMemLoadReturnLaneDwordsPerCycle(9).
		Build()
}

func TestRejectVMemCUWideReturnBudgetWithOtherReturnModels(t *testing.T) {
	for name, configureOldModel := range map[string]func(Builder) Builder{
		"per-instruction": func(b Builder) Builder {
			return b.WithVMemLoadReturnLaneDwordsPerCycle(9)
		},
		"per-wave": func(b Builder) Builder {
			return b.WithVMemWideLoadReturnLaneDwordsPerCycle(11)
		},
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected simultaneous return models to panic")
				}
			}()

			configureOldModel(MakeBuilder()).
				WithGPUType("gfx90c").
				WithVMemCUWideReturnUnitsPerCycle(13).
				Build()
		})
	}
}

func TestRejectNegativeVMemCUWideReturnBudget(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected negative CU-wide return budget to panic")
		}
	}()

	MakeBuilder().
		WithGPUType("gfx90c").
		WithVMemCUWideReturnUnitsPerCycle(-1).
		Build()
}

func TestRejectVMemCUWideReturnBudgetForOtherGPUs(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected unsupported GPU configuration to panic")
		}
	}()

	MakeBuilder().
		WithGPUType("vega64").
		WithVMemCUWideReturnUnitsPerCycle(13).
		Build()
}

func TestDefaultVMemCUWideConcurrentWavesToOne(t *testing.T) {
	s := simulation.MakeBuilder().
		WithoutMonitoring().
		WithOutputFileName(t.TempDir() + "/sim").
		Build()
	defer s.Terminate()

	MakeBuilder().
		WithSimulation(s).
		WithGPUType("gfx90c").
		WithVMemCUWideReturnUnitsPerCycle(13).
		Build()

	component := s.GetComponentByName("GPU[1].SA[0].CU[0]")
	computeUnit, ok := component.(*cu.Comp)
	if !ok {
		t.Fatalf("expected gfx90c compute unit, got %T", component)
	}
	if computeUnit.Spec().VMemCUWideReturnConcurrentWaves != 1 {
		t.Fatalf(
			"expected CU-wide concurrent waves to default to 1, got %d",
			computeUnit.Spec().VMemCUWideReturnConcurrentWaves,
		)
	}
}

func TestBuildGfx90cPlatformWithVMemCUWideBurstConcurrentWaves(t *testing.T) {
	s := simulation.MakeBuilder().
		WithoutMonitoring().
		WithOutputFileName(t.TempDir() + "/sim").
		Build()
	defer s.Terminate()

	MakeBuilder().
		WithSimulation(s).
		WithGPUType("gfx90c").
		WithVMemCUWideReturnUnitsPerCycle(13).
		WithVMemCUWideReturnBurstConcurrentWaves(1).
		WithVMemCUWideReturnBurstAssistInterval(3).
		Build()

	component := s.GetComponentByName("GPU[1].SA[0].CU[0]")
	computeUnit, ok := component.(*cu.Comp)
	if !ok {
		t.Fatalf("expected gfx90c compute unit, got %T", component)
	}
	if computeUnit.Spec().VMemCUWideReturnConcurrentWaves != 0 {
		t.Fatalf(
			"burst mode must not normalize static waves, got %d",
			computeUnit.Spec().VMemCUWideReturnConcurrentWaves,
		)
	}
	if computeUnit.Spec().VMemCUWideReturnBurstConcurrentWaves != 1 {
		t.Fatalf(
			"expected CU-wide burst concurrent waves 1, got %d",
			computeUnit.Spec().VMemCUWideReturnBurstConcurrentWaves,
		)
	}
	if computeUnit.Spec().VMemCUWideReturnBurstAssistInterval != 3 {
		t.Fatalf(
			"expected CU-wide burst assist interval 3, got %d",
			computeUnit.Spec().VMemCUWideReturnBurstAssistInterval,
		)
	}
}

func TestRejectInvalidVMemCUWideConcurrentWaves(t *testing.T) {
	tests := []struct {
		name  string
		units int
		waves int
	}{
		{name: "negative", units: 13, waves: -1},
		{name: "without CU-wide model", units: 0, waves: 2},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected invalid concurrent-wave configuration to panic")
				}
			}()

			MakeBuilder().
				WithGPUType("gfx90c").
				WithVMemCUWideReturnUnitsPerCycle(test.units).
				WithVMemCUWideReturnConcurrentWaves(test.waves).
				Build()
		})
	}
}

func TestRejectInvalidVMemCUWideBurstConcurrentWaves(t *testing.T) {
	tests := []struct {
		name   string
		units  int
		static int
		burst  int
	}{
		{name: "negative", units: 13, burst: -1},
		{name: "without CU-wide model", burst: 2},
		{name: "with static control", units: 13, static: 2, burst: 2},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected invalid burst configuration to panic")
				}
			}()

			MakeBuilder().
				WithGPUType("gfx90c").
				WithVMemCUWideReturnUnitsPerCycle(test.units).
				WithVMemCUWideReturnConcurrentWaves(test.static).
				WithVMemCUWideReturnBurstConcurrentWaves(test.burst).
				Build()
		})
	}
}

func TestRejectInvalidVMemCUWideBurstAssistInterval(t *testing.T) {
	tests := []struct {
		name     string
		units    int
		static   int
		burst    int
		interval int
	}{
		{name: "negative", units: 13, burst: 1, interval: -1},
		{name: "without CU-wide model", burst: 1, interval: 2},
		{name: "without Q1", units: 13, burst: 2, interval: 2},
		{name: "with static control", units: 13, static: 1, burst: 1, interval: 2},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected invalid burst assist configuration to panic")
				}
			}()

			MakeBuilder().
				WithGPUType("gfx90c").
				WithVMemCUWideReturnUnitsPerCycle(test.units).
				WithVMemCUWideReturnConcurrentWaves(test.static).
				WithVMemCUWideReturnBurstConcurrentWaves(test.burst).
				WithVMemCUWideReturnBurstAssistInterval(test.interval).
				Build()
		})
	}
}

func TestBuildPolaris10Platform(t *testing.T) {
	buildPlatform(t, "polaris10", 1)
}
