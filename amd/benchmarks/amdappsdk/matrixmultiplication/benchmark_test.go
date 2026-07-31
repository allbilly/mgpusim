package matrixmultiplication

import (
	"math"
	"slices"
	"testing"
)

func TestGenerateInputMatricesIsDeterministicAndFinite(t *testing.T) {
	matrixA1, matrixB1 := GenerateInputMatrices(32, 32, 32)
	matrixA2, matrixB2 := GenerateInputMatrices(32, 32, 32)

	if !slices.Equal(matrixA1.Data, matrixA2.Data) ||
		!slices.Equal(matrixB1.Data, matrixB2.Data) {
		t.Fatal("matrix inputs differ across calls")
	}
	if len(matrixA1.Data) != 32*32 || len(matrixB1.Data) != 32*32 {
		t.Fatalf("unexpected input lengths: A=%d, B=%d",
			len(matrixA1.Data), len(matrixB1.Data))
	}
	for name, values := range map[string][]float32{
		"A": matrixA1.Data,
		"B": matrixB1.Data,
	} {
		for i, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("matrix %s element %d is not finite: %v", name, i, value)
			}
		}
	}
}

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
