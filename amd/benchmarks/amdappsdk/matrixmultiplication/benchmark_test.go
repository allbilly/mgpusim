package matrixmultiplication

import "testing"

func TestFirstMatrixMismatchChecksEveryElement(t *testing.T) {
	expected := &Matrix{
		Width:  3,
		Height: 2,
		Data:   []float32{0, 1, 2, 3, 4, 5},
	}
	actual := &Matrix{
		Width:  3,
		Height: 2,
		Data:   []float32{0, 1, 2, 3, 4, 99},
	}

	x, y, equal := firstMatrixMismatch(expected, actual, 1e-3)
	if equal {
		t.Fatal("mismatch in the last element was not detected")
	}
	if x != 2 || y != 1 {
		t.Fatalf("got mismatch at [%d, %d], want [2, 1]", x, y)
	}
}

func TestFirstMatrixMismatchAcceptsTolerance(t *testing.T) {
	expected := &Matrix{Width: 1, Height: 1, Data: []float32{1}}
	actual := &Matrix{Width: 1, Height: 1, Data: []float32{1.0005}}

	_, _, equal := firstMatrixMismatch(expected, actual, 1e-3)
	if !equal {
		t.Fatal("values within tolerance should compare equal")
	}
}
