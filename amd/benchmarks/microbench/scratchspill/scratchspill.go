// Package scratchspill implements a paired private-segment spill probe.
package scratchspill

import (
	_ "embed"
	"fmt"
	"log"
	"math"

	"github.com/sarchlab/mgpusim/v5/amd/arch"
	"github.com/sarchlab/mgpusim/v5/amd/driver"
	"github.com/sarchlab/mgpusim/v5/amd/insts"
)

const (
	// VariantControl4 selects the register-resident control kernel.
	VariantControl4 = "control4"
	// VariantScratch4 selects the production-like four-spill kernel.
	VariantScratch4 = "scratch4"

	innerWidth = 32
	outputRows = 32
)

// KernelArgs matches the gfx90c HIP kernarg layout of the paired kernels.
type KernelArgs struct {
	MatrixA             driver.Ptr
	MatrixB             driver.Ptr
	MatrixC             driver.Ptr
	WidthA              uint32
	Padding1            uint32
	HiddenBlockCountX   uint32
	HiddenBlockCountY   uint32
	HiddenBlockCountZ   uint32
	HiddenGroupSizeX    uint16
	HiddenGroupSizeY    uint16
	HiddenGroupSizeZ    uint16
	HiddenRemainderX    uint16
	HiddenRemainderY    uint16
	HiddenRemainderZ    uint16
	Padding2            [16]byte
	HiddenGlobalOffsetX int64
	HiddenGlobalOffsetY int64
	HiddenGlobalOffsetZ int64
	HiddenGridDims      uint16
}

// Benchmark defines the paired private-spill probe.
type Benchmark struct {
	driver  *driver.Driver
	context *driver.Context
	queue   *driver.CommandQueue
	kernel  *insts.KernelCodeObject
	gpus    []int

	Arch       arch.Type
	Variant    string
	Workgroups int

	matrixA []float32
	matrixB []float32
	output  []float32
	dA      driver.Ptr
	dB      driver.Ptr
	dC      driver.Ptr

	useUnifiedMemory bool
}

//go:embed kernels_gfx90c.hsaco
var gcn5HSACOBytes []byte

// NewBenchmark creates a private-spill benchmark.
func NewBenchmark(gpuDriver *driver.Driver) *Benchmark {
	b := &Benchmark{driver: gpuDriver}
	b.context = gpuDriver.Init()
	b.queue = gpuDriver.CreateCommandQueue(b.context)
	return b
}

// SelectGPU selects the single GPU used by this probe.
func (b *Benchmark) SelectGPU(gpus []int) {
	b.gpus = gpus
}

// SetUnifiedMemory requests unified memory.
func (b *Benchmark) SetUnifiedMemory() {
	b.useUnifiedMemory = true
}

func (b *Benchmark) setDefaultsAndValidate() {
	if b.Variant == "" {
		b.Variant = VariantScratch4
	}
	if b.Variant != VariantControl4 && b.Variant != VariantScratch4 {
		log.Panicf("unknown private-spill variant %q", b.Variant)
	}
	if b.Workgroups == 0 {
		b.Workgroups = 16
	}
	if b.Workgroups < 0 {
		log.Panic("private-spill work-group count must be positive")
	}
	if uint64(b.Workgroups) > uint64(^uint32(0))/8 {
		log.Panic("private-spill work-group count exceeds the kernel ABI")
	}
	maxInt := uint64(^uint(0) >> 1)
	if uint64(b.Workgroups) > maxInt/(32*outputRows) {
		log.Panic("private-spill output size overflows")
	}
}

func (b *Benchmark) loadProgram() {
	symbol := "private_scratch4_kernel"
	if b.Variant == VariantControl4 {
		symbol = "private_control4_kernel"
	}
	b.kernel = insts.LoadKernelCodeObjectFromBytes(gcn5HSACOBytes, symbol)
	if b.kernel == nil {
		log.Panicf("failed to load %s", symbol)
	}
}

// Run executes one member of the paired probe.
func (b *Benchmark) Run() {
	if b.Arch != arch.GCN5 {
		log.Panic("the private-spill probe currently requires -arch gcn5")
	}
	if len(b.gpus) != 1 {
		log.Panic("the private-spill probe requires exactly one GPU")
	}
	b.setDefaultsAndValidate()
	b.loadProgram()
	b.driver.SelectGPU(b.context, b.gpus[0])
	b.initMem()
	b.exec()
}

func (b *Benchmark) allocAndCopy(values []float32) driver.Ptr {
	bytes := uint64(len(values) * 4)
	var ptr driver.Ptr
	if b.useUnifiedMemory {
		ptr = b.driver.AllocateUnifiedMemory(b.context, bytes)
	} else {
		ptr = b.driver.AllocateMemory(b.context, bytes)
	}
	b.driver.MemCopyH2D(b.context, ptr, values)
	return ptr
}

func (b *Benchmark) initMem() {
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
	b.dA = b.allocAndCopy(b.matrixA)
	b.dB = b.allocAndCopy(b.matrixB)
	b.dC = b.allocAndCopy(b.output)
}

func (b *Benchmark) exec() {
	globalX := uint32(b.Workgroups * 8)
	args := KernelArgs{
		MatrixA:           b.dA,
		MatrixB:           b.dB,
		MatrixC:           b.dC,
		WidthA:            innerWidth,
		HiddenBlockCountX: uint32(b.Workgroups),
		HiddenBlockCountY: 1,
		HiddenBlockCountZ: 1,
		HiddenGroupSizeX:  8,
		HiddenGroupSizeY:  8,
		HiddenGroupSizeZ:  1,
		HiddenGridDims:    2,
	}
	b.driver.EnqueueLaunchKernel(
		b.queue,
		b.kernel,
		[3]uint32{globalX, 8, 1},
		[3]uint16{8, 8, 1},
		&args,
	)
	b.driver.DrainCommandQueue(b.queue)
}

func (b *Benchmark) expectedAt(row, col int) float32 {
	outputCols := b.Workgroups * 32
	var sum float32
	for k := 0; k < innerWidth; k++ {
		sum += b.matrixA[row*innerWidth+k] *
			b.matrixB[k*outputCols+col]
	}
	return sum
}

// Verify checks every output element against a CPU reference.
func (b *Benchmark) Verify() {
	b.driver.MemCopyD2H(b.context, b.output, b.dC)
	outputCols := b.Workgroups * 32
	for row := 0; row < outputRows; row++ {
		for col := 0; col < outputCols; col++ {
			index := row*outputCols + col
			want := b.expectedAt(row, col)
			if math.Abs(float64(b.output[index]-want)) > 1e-4 {
				log.Panic(fmt.Sprintf(
					"private-spill mismatch at [%d,%d]: got %g, want %g",
					row, col, b.output[index], want))
			}
		}
	}
	log.Printf("Passed!\n")
}
