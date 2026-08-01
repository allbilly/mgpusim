package shaderarray

import "testing"

func TestVMemReturnFanoutBandwidthPropagation(t *testing.T) {
	builder := MakeBuilder().WithVMemReturnFanoutLaneDwordsPerCycle(7)

	spec := builder.cuSpec()
	if spec.VMemReturnFanoutLaneDwordsPerCycle != 7 {
		t.Fatalf(
			"expected vector-memory return fanout bandwidth 7, got %d",
			spec.VMemReturnFanoutLaneDwordsPerCycle,
		)
	}
}

func TestVMemLoadReturnBandwidthPropagation(t *testing.T) {
	builder := MakeBuilder().WithVMemLoadReturnLaneDwordsPerCycle(9)

	spec := builder.cuSpec()
	if spec.VMemLoadReturnLaneDwordsPerCycle != 9 {
		t.Fatalf(
			"expected vector-memory load return bandwidth 9, got %d",
			spec.VMemLoadReturnLaneDwordsPerCycle,
		)
	}
}

func TestVMemWideLoadReturnBandwidthPropagation(t *testing.T) {
	builder := MakeBuilder().WithVMemWideLoadReturnLaneDwordsPerCycle(11)

	spec := builder.cuSpec()
	if spec.VMemWideLoadReturnLaneDwordsPerCycle != 11 {
		t.Fatalf(
			"expected wide vector-memory load return bandwidth 11, got %d",
			spec.VMemWideLoadReturnLaneDwordsPerCycle,
		)
	}
}

func TestVMemCUWideReturnBudgetPropagation(t *testing.T) {
	builder := MakeBuilder().WithVMemCUWideReturnUnitsPerCycle(13)

	spec := builder.cuSpec()
	if spec.VMemCUWideReturnUnitsPerCycle != 13 {
		t.Fatalf(
			"expected CU-wide vector-memory return budget 13, got %d",
			spec.VMemCUWideReturnUnitsPerCycle,
		)
	}
}
