package main

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"

	"github.com/sarchlab/mgpusim/v5/amd/benchmarks/heteromark/kmeans"
)

func float32Membership(
	features []float32, numPoints, numFeatures, numClusters int,
) []int32 {
	membership := make([]int32, numPoints)
	for point := 0; point < numPoints; point++ {
		minimum := float32(math.MaxFloat32)
		closest := 0
		for cluster := 0; cluster < numClusters; cluster++ {
			distance := float32(0)
			for feature := 0; feature < numFeatures; feature++ {
				difference := features[point*numFeatures+feature] -
					features[cluster*numFeatures+feature]
				distance += difference * difference
			}
			if distance < minimum {
				minimum = distance
				closest = cluster
			}
		}
		membership[point] = int32(closest)
	}
	return membership
}

func readFixtureForTest[T any](t *testing.T, path string, count int) []T {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()

	values := make([]T, count)
	if err := binary.Read(file, binary.LittleEndian, values); err != nil {
		t.Fatal(err)
	}
	return values
}

func TestKMeansSizeSweepFixturesMatchBenchmarkGenerator(t *testing.T) {
	outputDir := t.TempDir()
	if err := generateFixtures(outputDir); err != nil {
		t.Fatal(err)
	}

	for _, numPoints := range kmeansPointSweep {
		wantFeatures := kmeans.GenerateFeatures(numPoints, 16)
		wantMembership := kmeans.OneIterationMembership(
			wantFeatures, numPoints, 16, 5)
		prefix := filepath.Join(
			outputDir, "kmeans_"+strconv.Itoa(numPoints)+"_")
		gotFeatures := readFixtureForTest[float32](
			t, prefix+"features.f32", numPoints*16)
		gotMembership := readFixtureForTest[int32](
			t, prefix+"membership.i32", numPoints)

		if !reflect.DeepEqual(gotFeatures, wantFeatures) {
			t.Fatalf("%d-point feature fixture differs from benchmark input", numPoints)
		}
		if !reflect.DeepEqual(gotMembership, wantMembership) {
			t.Fatalf("%d-point membership fixture differs from benchmark result", numPoints)
		}
		// The simulator's CPU verifier accumulates in float64 while the GPU
		// kernel accumulates in float32 (and may fuse multiply-adds). Identical
		// classifications at both precision extremes guard this sweep against
		// near-tie memberships whose result could depend on rounding.
		if float32Result := float32Membership(
			wantFeatures, numPoints, 16, 5,
		); !reflect.DeepEqual(float32Result, wantMembership) {
			t.Fatalf(
				"%d-point memberships differ between float32 and float64 references",
				numPoints,
			)
		}
	}

	legacy := readFixtureForTest[float32](
		t, filepath.Join(outputDir, "kmeans_features.f32"), 4096*16)
	defaultSize := readFixtureForTest[float32](
		t, filepath.Join(outputDir, "kmeans_4096_features.f32"), 4096*16)
	if !reflect.DeepEqual(legacy, defaultSize) {
		t.Fatal("legacy default K-means fixture is not the 4096-point fixture")
	}
}
