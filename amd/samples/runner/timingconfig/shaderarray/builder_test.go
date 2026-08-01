package shaderarray

import "testing"

func TestLDSB128ServiceExtraCyclesPropagation(t *testing.T) {
	builder := MakeBuilder().WithLDSB128ServiceExtraCycles(17)
	spec := builder.cuSpec()

	if spec.LDSB128ServiceExtraCycles != 17 {
		t.Fatalf(
			"expected LDS B128 service extra cycles 17, got %d",
			spec.LDSB128ServiceExtraCycles,
		)
	}
}

func TestPrivateSegmentCoalescingPenaltyPropagation(t *testing.T) {
	builder := MakeBuilder().
		WithMaxCoalescingPenalty(13).
		WithMaxPrivateSegmentCoalescingPenalty(29)
	spec := builder.cuSpec()

	if spec.MaxPrivateSegmentCoalescingPenalty != 29 {
		t.Fatalf(
			"expected private-segment coalescing penalty 29, got %d",
			spec.MaxPrivateSegmentCoalescingPenalty,
		)
	}
}

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
	builder := MakeBuilder().
		WithVMemCUWideReturnUnitsPerCycle(13).
		WithVMemCUWideReturnConcurrentWaves(3)

	spec := builder.cuSpec()
	if spec.VMemCUWideReturnUnitsPerCycle != 13 {
		t.Fatalf(
			"expected CU-wide vector-memory return budget 13, got %d",
			spec.VMemCUWideReturnUnitsPerCycle,
		)
	}
	if spec.VMemCUWideReturnConcurrentWaves != 3 {
		t.Fatalf(
			"expected CU-wide concurrent waves 3, got %d",
			spec.VMemCUWideReturnConcurrentWaves,
		)
	}
}

func TestVMemCUWideBurstConcurrentWavesPropagation(t *testing.T) {
	builder := MakeBuilder().
		WithVMemCUWideReturnUnitsPerCycle(13).
		WithVMemCUWideReturnBurstConcurrentWaves(1).
		WithVMemCUWideReturnBurstAssistInterval(3)

	spec := builder.cuSpec()
	if spec.VMemCUWideReturnBurstConcurrentWaves != 1 {
		t.Fatalf(
			"expected CU-wide burst concurrent waves 1, got %d",
			spec.VMemCUWideReturnBurstConcurrentWaves,
		)
	}
	if spec.VMemCUWideReturnBurstAssistInterval != 3 {
		t.Fatalf(
			"expected CU-wide burst assist interval 3, got %d",
			spec.VMemCUWideReturnBurstAssistInterval,
		)
	}
}

func TestVMemCUWideBurstAssistDepthScalePropagation(t *testing.T) {
	builder := MakeBuilder().
		WithVMemCUWideReturnUnitsPerCycle(13).
		WithVMemCUWideReturnBurstConcurrentWaves(1).
		WithVMemCUWideReturnBurstAssistDepthScale(16)

	spec := builder.cuSpec()
	if spec.VMemCUWideReturnBurstAssistDepthScale != 16 {
		t.Fatalf(
			"expected CU-wide burst assist depth scale 16, got %d",
			spec.VMemCUWideReturnBurstAssistDepthScale,
		)
	}
}
