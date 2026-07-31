// Command generate_fixtures writes the deterministic inputs shared by the
// simulator benchmarks and the rx570 hardware timing harness. Identical to the
// gfx90c fixture generator (same seeds, same sizes) so sim and HW consume the
// same bytes. It emits every k-means and PageRank holdout size so --size
// selects an input generated with the same dimensions as the Go benchmark.
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

	outputs := []struct {
		name string
		data any
	}{}
	appendOutput := func(name string, data any) {
		outputs = append(outputs, struct {
			name string
			data any
		}{name, data})
	}

	const numFeatures = 16
	for _, numPoints := range []int{1024, 2048, 4096, 6144, 8192, 16384} {
		rng := rand.New(rand.NewSource(0))
		features := make([]float32, numPoints*numFeatures)
		for i := range features {
			features[i] = rng.Float32()
		}
		appendOutput(fmt.Sprintf("kmeans_features_%d.f32", numPoints), features)
	}

	for _, numNodes := range []uint32{128, 256, 512, 1024} {
		numEdges := numNodes * numNodes / 2
		matrix := csr.MakeMatrixGenerator(numNodes, numEdges).GenerateMatrix()
		prefix := fmt.Sprintf("pagerank_%d", numNodes)
		appendOutput(prefix+"_row_offsets.u32", matrix.RowOffsets)
		appendOutput(prefix+"_columns.u32", matrix.ColumnNumbers)
		appendOutput(prefix+"_values.f32", matrix.Values)
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
