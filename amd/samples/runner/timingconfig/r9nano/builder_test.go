package r9nano

import "testing"

func TestVMemLoadReturnBandwidthConfiguration(t *testing.T) {
	builder := MakeBuilder().WithVMemLoadReturnLaneDwordsPerCycle(9)

	if builder.vmemLoadReturnLaneDwordsPerCycle != 9 {
		t.Fatalf(
			"expected vector-memory load return bandwidth 9, got %d",
			builder.vmemLoadReturnLaneDwordsPerCycle,
		)
	}
}

func TestVMemWideLoadReturnBandwidthConfiguration(t *testing.T) {
	builder := MakeBuilder().WithVMemWideLoadReturnLaneDwordsPerCycle(11)

	if builder.vmemWideLoadReturnLaneDwordsPerCycle != 11 {
		t.Fatalf(
			"expected wide vector-memory load return bandwidth 11, got %d",
			builder.vmemWideLoadReturnLaneDwordsPerCycle,
		)
	}
}

func TestVMemCUWideReturnBudgetConfiguration(t *testing.T) {
	builder := MakeBuilder().WithVMemCUWideReturnUnitsPerCycle(13)

	if builder.vmemCUWideReturnUnitsPerCycle != 13 {
		t.Fatalf(
			"expected CU-wide vector-memory return budget 13, got %d",
			builder.vmemCUWideReturnUnitsPerCycle,
		)
	}
}
