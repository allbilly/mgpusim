// Command generate_fixtures writes the deterministic inputs shared by the
// simulator benchmarks and the gfx90c hardware timing harness.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/sarchlab/mgpusim/v5/amd/benchmarks/matrix/csr"
)

func writeBinary[T any](outputDir, name string, values []T) error {
	path := filepath.Join(outputDir, name)
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	return binary.Write(file, binary.LittleEndian, values)
}

func main() {
	outputDir := flag.String("out", "build", "fixture output directory")
	flag.Parse()

	if err := os.MkdirAll(*outputDir, 0o755); err != nil {
		panic(err)
	}

	const (
		numPoints   = 4096
		numFeatures = 16
	)

	rng := rand.New(rand.NewSource(0))
	features := make([]float32, numPoints*numFeatures)
	for i := range features {
		features[i] = rng.Float32()
	}

	type fixture struct {
		name string
		data any
	}
	outputs := []fixture{
		{"kmeans_features.f32", features},
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
			err = writeBinary(*outputDir, output.name, data)
		case []uint32:
			err = writeBinary(*outputDir, output.name, data)
		}
		if err != nil {
			panic(fmt.Errorf("write %s: %w", output.name, err))
		}
	}
}
