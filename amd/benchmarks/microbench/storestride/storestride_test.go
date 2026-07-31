package storestride

import "testing"

func TestExpectedOutputUsesFullLinesAndLeavesGapsZero(t *testing.T) {
	b := &Benchmark{
		StrideLines:           2,
		AllocationStrideLines: 4,
		Repeats:               2,
		Workgroups:            1,
	}
	b.setDefaultsAndValidate()
	output := b.expectedOutput()

	for epoch := 0; epoch < b.Repeats; epoch++ {
		baseLine := uint64(epoch) * b.epochSpan
		for line := uint64(0); line < b.epochSpan; line++ {
			wantNonzero := line%2 == 0 && line <= 30
			for word := uint64(0); word < wordsPerLine; word++ {
				gotNonzero := output[int((baseLine+line)*wordsPerLine+word)] != 0
				if gotNonzero != wantNonzero {
					t.Fatalf("epoch %d line %d word %d nonzero=%t, want %t",
						epoch, line, word, gotNonzero, wantNonzero)
				}
			}
		}
	}
}

func TestAllocationFootprintDoesNotDependOnActualStride(t *testing.T) {
	makeBenchmark := func(stride int) *Benchmark {
		b := &Benchmark{
			StrideLines:           stride,
			AllocationStrideLines: 128,
			Repeats:               32,
			Workgroups:            2,
		}
		b.setDefaultsAndValidate()
		return b
	}

	dense := makeBenchmark(1)
	far := makeBenchmark(128)
	if dense.outputWords != far.outputWords || dense.epochSpan != far.epochSpan {
		t.Fatalf("allocation changed with actual stride: dense=%d/%d far=%d/%d",
			dense.outputWords, dense.epochSpan, far.outputWords, far.epochSpan)
	}
}
