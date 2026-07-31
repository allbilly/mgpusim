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
