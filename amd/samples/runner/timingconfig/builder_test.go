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

func TestBuildPolaris10Platform(t *testing.T) {
	buildPlatform(t, "polaris10", 1)
}
