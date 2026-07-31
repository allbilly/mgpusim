package scratchspill

import "testing"

func TestInputAndReferenceGeometry(t *testing.T) {
	b := &Benchmark{Variant: VariantControl4, Workgroups: 7}
	b.setDefaultsAndValidate()
	b.initMemForTest()
	if got, want := len(b.matrixA), 32*32; got != want {
		t.Fatalf("matrix A length %d, want %d", got, want)
	}
	if got, want := len(b.matrixB), 32*7*32; got != want {
		t.Fatalf("matrix B length %d, want %d", got, want)
	}
	if got, want := len(b.output), 32*7*32; got != want {
		t.Fatalf("output length %d, want %d", got, want)
	}
	if got := b.expectedAt(3, 17); got == 0 {
		t.Fatal("reference unexpectedly collapsed to zero")
	}
}

func TestDefaultsSelectProductionLikeVariant(t *testing.T) {
	b := &Benchmark{}
	b.setDefaultsAndValidate()
	if b.Variant != VariantScratch4 || b.Workgroups != 16 {
		t.Fatalf("defaults are variant=%q workgroups=%d", b.Variant, b.Workgroups)
	}
}

func TestRejectsInvalidConfiguration(t *testing.T) {
	assertPanics(t, func() {
		b := &Benchmark{Variant: "unknown", Workgroups: 1}
		b.setDefaultsAndValidate()
	})
	assertPanics(t, func() {
		b := &Benchmark{Variant: VariantScratch4, Workgroups: -1}
		b.setDefaultsAndValidate()
	})
}

func assertPanics(t *testing.T, action func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	action()
}

// initMemForTest builds host inputs without requiring a driver or allocation.
func (b *Benchmark) initMemForTest() {
	outputCols := b.Workgroups * 32
	b.matrixA = make([]float32, outputRows*innerWidth)
	b.matrixB = make([]float32, innerWidth*outputCols)
	b.output = make([]float32, outputRows*outputCols)
	for row := 0; row < outputRows; row++ {
		for k := 0; k < innerWidth; k++ {
			b.matrixA[row*innerWidth+k] = float32((row+k)%7 - 3)
		}
	}
	for k := 0; k < innerWidth; k++ {
		for col := 0; col < outputCols; col++ {
			b.matrixB[k*outputCols+col] = float32((k*3+col)%5 - 2)
		}
	}
}
