// Command generate_fixtures writes the deterministic inputs shared by the
// simulator benchmarks and the gfx90c hardware timing harness.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/sarchlab/mgpusim/v5/amd/benchmarks/amdappsdk/matrixmultiplication"
	"github.com/sarchlab/mgpusim/v5/amd/benchmarks/heteromark/kmeans"
	"github.com/sarchlab/mgpusim/v5/amd/benchmarks/matrix/csr"
)

var kmeansPointSweep = []int{1024, 2048, 4096, 8192}

type fixture struct {
	name string
	data any
}

func writeBinary[T any](outputDir, name string, values []T) error {
	path := filepath.Join(outputDir, name)
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	return binary.Write(file, binary.LittleEndian, values)
}

func generateFixtures(outputDir string) error {
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return err
	}

	const (
		numFeatures = 16
		numClusters = 5
	)

	outputs := make([]fixture, 0)
	for _, numPoints := range kmeansPointSweep {
		features := kmeans.GenerateFeatures(numPoints, numFeatures)
		membership := kmeans.OneIterationMembership(
			features, numPoints, numFeatures, numClusters)
		prefix := fmt.Sprintf("kmeans_%d_", numPoints)
		outputs = append(outputs,
			fixture{prefix + "features.f32", features},
			fixture{prefix + "membership.i32", membership},
		)

		// Preserve the original default-size fixture for existing scripts.
		if numPoints == 4096 {
			outputs = append(outputs,
				fixture{"kmeans_features.f32", features},
			)
		}
	}

	for _, size := range []uint32{32, 64, 128} {
		matrixA, matrixB := matrixmultiplication.GenerateInputMatrices(
			size, size, size)
		prefix := fmt.Sprintf("matrixmult_%d_", size)
		outputs = append(outputs,
			fixture{prefix + "a.f32", matrixA.Data},
			fixture{prefix + "b.f32", matrixB.Data},
		)
	}

	for _, numNodes := range []uint32{128, 256, 512} {
		numEdges := numNodes * numNodes / 2
		matrix := csr.MakeMatrixGenerator(numNodes, numEdges).GenerateMatrix()
		prefix := fmt.Sprintf("pagerank_%d_", numNodes)
		outputs = append(outputs,
			fixture{prefix + "row_offsets.u32", matrix.RowOffsets},
			fixture{prefix + "columns.u32", matrix.ColumnNumbers},
			fixture{prefix + "values.f32", matrix.Values},
		)

		// Preserve the original default-size fixture names for existing users.
		if numNodes == 512 {
			outputs = append(outputs,
				fixture{"pagerank_row_offsets.u32", matrix.RowOffsets},
				fixture{"pagerank_columns.u32", matrix.ColumnNumbers},
				fixture{"pagerank_values.f32", matrix.Values},
			)
		}
	}

	for _, output := range outputs {
		var err error
		switch data := output.data.(type) {
		case []float32:
			err = writeBinary(outputDir, output.name, data)
		case []uint32:
			err = writeBinary(outputDir, output.name, data)
		case []int32:
			err = writeBinary(outputDir, output.name, data)
		}
		if err != nil {
			return fmt.Errorf("write %s: %w", output.name, err)
		}
	}
	return nil
}

func main() {
	outputDir := flag.String("out", "build", "fixture output directory")
	flag.Parse()

	if err := generateFixtures(*outputDir); err != nil {
		panic(err)
	}
}
