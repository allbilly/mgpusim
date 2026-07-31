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
