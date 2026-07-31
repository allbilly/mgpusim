package kmeans

import (
	"math"
	"reflect"
	"testing"
)

func TestGenerateFeaturesIsDeterministic(t *testing.T) {
	first := GenerateFeatures(4, 3)
	second := GenerateFeatures(4, 3)
	if !reflect.DeepEqual(first, second) {
		t.Fatal("GenerateFeatures returned different values for the same shape")
	}

	wantBits := []uint32{
		0x3f71f860, 0x3e7ad821, 0x3f27ecc0, 0x3d5e97a5,
	}
	for i, want := range wantBits {
		if got := math.Float32bits(first[i]); got != want {
			t.Errorf("feature %d bits: got %#08x, want %#08x", i, got, want)
		}
	}
}

func TestGenerateFeaturesSizeSweepUsesSamePrefix(t *testing.T) {
	small := GenerateFeatures(1024, 16)
	large := GenerateFeatures(8192, 16)
	if !reflect.DeepEqual(small, large[:len(small)]) {
		t.Fatal("size-sweep inputs do not share the simulator's seeded prefix")
	}
}

func TestOneIterationMembership(t *testing.T) {
	features := []float32{
		0, 0,
		10, 10,
		1, 2,
		8, 9,
	}
	want := []int32{0, 1, 0, 1}
	got := OneIterationMembership(features, 4, 2, 2)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("membership: got %v, want %v", got, want)
	}
}

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
