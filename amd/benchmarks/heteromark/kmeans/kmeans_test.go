package kmeans

import "testing"

func TestTransposeFeatureLayout(t *testing.T) {
	input := []float32{
		0, 1, 2,
		3, 4, 5,
	}
	got := transposeFeatureLayout(input, 2, 3)
	want := []float32{
		0, 3,
		1, 4,
		2, 5,
	}

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("element %d: got %v, want %v", i, got[i], want[i])
		}
	}
}
