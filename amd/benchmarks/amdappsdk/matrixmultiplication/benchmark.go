// Package matrixmultiplication implements the matrix multiplication benchmark
// from AMDAPPSDK.
package matrixmultiplication

import (
	"log"
	"math"
	"math/rand"

	"github.com/sarchlab/mgpusim/v5/amd/arch"
	"github.com/sarchlab/mgpusim/v5/amd/driver"
)

// Benchmark defines a benchmark
type Benchmark struct {
	driver  *driver.Driver
	context *driver.Context
	gpus    []int

	Arch                      arch.Type
	X, Y, Z                   uint32
	MatrixA, MatrixB, MatrixC *Matrix
	useUnifiedMemory          bool
}

// NewBenchmark makes a new benchmark
func NewBenchmark(driver *driver.Driver) *Benchmark {
	b := new(Benchmark)
	b.driver = driver
	b.context = driver.Init()
	return b
}

// SelectGPU selects GPU
func (b *Benchmark) SelectGPU(gpus []int) {
	b.gpus = gpus
}

// Run runs
func (b *Benchmark) Run() {
	b.driver.SelectGPU(b.context, b.gpus[0])
	b.initMem()
	b.exec()
}

// SetUnifiedMemory uses Unified Memory
func (b *Benchmark) SetUnifiedMemory() {
	b.useUnifiedMemory = true
}

// GenerateInputMatrices creates the deterministic inputs used by the
// simulator and by external exact-HSACO calibration harnesses.
func GenerateInputMatrices(x, y, z uint32) (matrixA, matrixB *Matrix) {
	// Use a local random source so the input is reproducible. rand.Seed has
	// been a no-op since Go 1.24, so seeding the global generator no longer
	// produces a deterministic sequence.
	rng := rand.New(rand.NewSource(0))

	matrixA = NewMatrix(x, y)
	for i := uint32(0); i < x; i++ {
		for j := uint32(0); j < y; j++ {
			matrixA.Data[j*x+i] = rng.Float32()
		}
	}

	matrixB = NewMatrix(z, x)
	for i := uint32(0); i < z; i++ {
		for j := uint32(0); j < x; j++ {
			matrixB.Data[j*z+i] = rng.Float32()
		}
	}

	return matrixA, matrixB
}

func (b *Benchmark) initMem() {
	b.MatrixA, b.MatrixB = GenerateInputMatrices(b.X, b.Y, b.Z)
}

func (b *Benchmark) exec() {
	m := NewGPUMatrixMultiplier(b.driver, b.context)
	m.SelectGPU(b.gpus)
	m.Arch = b.Arch
	m.useUnifiedMemory = b.useUnifiedMemory
	b.MatrixC = m.Multiply(b.MatrixA, b.MatrixB)
}

// Verify verifies
func (b *Benchmark) Verify() {
	m := CPUMatrixMultiplier{}
	mCPU := m.Multiply(b.MatrixA, b.MatrixB)
	if mCPU.Width != b.MatrixC.Width || mCPU.Height != b.MatrixC.Height {
		log.Panicf("matrix dimensions differ: expected %dx%d, got %dx%d",
			mCPU.Width, mCPU.Height, b.MatrixC.Width, b.MatrixC.Height)
	}

	x, y, ok := firstMatrixMismatch(mCPU, b.MatrixC, 1e-3)
	if !ok {
		index := x + y*mCPU.Width
		log.Panicf("mismatch at [%d, %d]: expected %f, but get %f",
			x, y, mCPU.Data[index], b.MatrixC.Data[index])
	}

	log.Print("Passed!")
}

func firstMatrixMismatch(
	expected, actual *Matrix,
	tolerance float64,
) (x, y uint32, equal bool) {
	for y = 0; y < expected.Height; y++ {
		for x = 0; x < expected.Width; x++ {
			index := x + y*expected.Width
			if math.Abs(float64(
				expected.Data[index]-actual.Data[index],
			)) > tolerance {
				return x, y, false
			}
		}
	}

	return 0, 0, true
}
