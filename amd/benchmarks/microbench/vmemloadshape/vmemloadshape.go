// Package vmemloadshape implements a gfx90c vector-memory load-shape probe.
package vmemloadshape

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
	ModeSerial       = "serial"
	ModeIndependent4 = "independent4"
)

// KernelArgs matches all six gfx90c load-shape kernel symbols.
type KernelArgs struct {
	Input           driver.Ptr
	Output          driver.Ptr
	Repeats         uint32
	VectorElements  uint32
	AddressGroups   uint32
	ThreadsPerBlock uint32
}

// Benchmark defines one point in the VMEM load-shape matrix.
type Benchmark struct {
	driver  *driver.Driver
	context *driver.Context
	queue   *driver.CommandQueue
	kernel  *insts.KernelCodeObject
	gpus    []int

	Arch        arch.Type
	WidthDwords int
	Mode        string
	AliasLanes  int
	ArrayBytes  int
	Repeats     int
	Workgroups  int
	input       []float32
	output      []float32
	dInput      driver.Ptr
	dOutput     driver.Ptr
	useUnified  bool
}

//go:embed kernels_gfx90c.hsaco
var gcn5HSACOBytes []byte

func NewBenchmark(gpuDriver *driver.Driver) *Benchmark {
	b := &Benchmark{driver: gpuDriver}
	b.context = gpuDriver.Init()
	b.queue = gpuDriver.CreateCommandQueue(b.context)
	return b
}

func (b *Benchmark) SelectGPU(gpus []int) { b.gpus = gpus }
func (b *Benchmark) SetUnifiedMemory()    { b.useUnified = true }

func isPowerOfTwo(value int) bool {
	return value > 0 && value&(value-1) == 0
}

func (b *Benchmark) setDefaultsAndValidate() {
	if b.WidthDwords == 0 {
		b.WidthDwords = 4
	}
	if b.Mode == "" {
		b.Mode = ModeSerial
	}
	if b.AliasLanes == 0 {
		b.AliasLanes = 8
	}
	if b.ArrayBytes == 0 {
		b.ArrayBytes = 8 * 1024
	}
	if b.Repeats == 0 {
		b.Repeats = 1024
	}
	if b.Workgroups == 0 {
		b.Workgroups = 16
	}
	if b.WidthDwords != 1 && b.WidthDwords != 2 && b.WidthDwords != 4 {
		log.Panic("VMEM width must be 1, 2, or 4 dwords")
	}
	if b.Mode != ModeSerial && b.Mode != ModeIndependent4 {
		log.Panicf("unknown VMEM dependency mode %q", b.Mode)
	}
	if b.AliasLanes != 1 && b.AliasLanes != 2 &&
		b.AliasLanes != 4 && b.AliasLanes != 8 {
		log.Panic("VMEM alias lanes must be 1, 2, 4, or 8")
	}
	if b.ArrayBytes <= 0 || b.ArrayBytes%(b.WidthDwords*4) != 0 {
		log.Panic("VMEM array bytes must be a positive multiple of load width")
	}
	if !isPowerOfTwo(b.ArrayBytes / (b.WidthDwords * 4)) {
		log.Panic("VMEM array must contain a power-of-two number of vectors")
	}
	if b.Repeats <= 0 || (b.Mode == ModeIndependent4 && b.Repeats%4 != 0) {
		log.Panic("VMEM repeats must be positive and independent4 requires a multiple of 4")
	}
	if b.Workgroups <= 0 || uint64(b.Workgroups) > uint64(^uint32(0))/64 {
		log.Panic("invalid VMEM work-group count")
	}
}

func (b *Benchmark) symbol() string {
	width := "dword"
	if b.WidthDwords > 1 {
		width = fmt.Sprintf("dwordx%d", b.WidthDwords)
	}
	return fmt.Sprintf("vmem_load_%s_%s", width, b.Mode)
}

func (b *Benchmark) loadProgram() {
	b.kernel = insts.LoadKernelCodeObjectFromBytes(gcn5HSACOBytes, b.symbol())
	if b.kernel == nil {
		log.Panicf("failed to load %s", b.symbol())
	}
}

func (b *Benchmark) Run() {
	if b.Arch != arch.GCN5 {
		log.Panic("the VMEM load-shape probe currently requires -arch gcn5")
	}
	if len(b.gpus) != 1 {
		log.Panic("the VMEM load-shape probe requires exactly one GPU")
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
	if b.useUnified {
		ptr = b.driver.AllocateUnifiedMemory(b.context, bytes)
	} else {
		ptr = b.driver.AllocateMemory(b.context, bytes)
	}
	b.driver.MemCopyH2D(b.context, ptr, values)
	return ptr
}

func (b *Benchmark) initHostData() {
	b.input = make([]float32, b.ArrayBytes/4)
	for i := range b.input {
		b.input[i] = float32(i%13 + 1)
	}
	b.output = make([]float32, b.Workgroups*64)
}

func (b *Benchmark) initMem() {
	b.initHostData()
	b.dInput = b.allocAndCopy(b.input)
	b.dOutput = b.allocAndCopy(b.output)
}

func (b *Benchmark) exec() {
	args := KernelArgs{
		Input:           b.dInput,
		Output:          b.dOutput,
		Repeats:         uint32(b.Repeats),
		VectorElements:  uint32(b.ArrayBytes / (b.WidthDwords * 4)),
		AddressGroups:   uint32(64 / b.AliasLanes),
		ThreadsPerBlock: 64,
	}
	b.driver.EnqueueLaunchKernel(
		b.queue, b.kernel,
		[3]uint32{uint32(b.Workgroups * 64), 1, 1},
		[3]uint16{64, 1, 1}, &args,
	)
	b.driver.DrainCommandQueue(b.queue)
}

func (b *Benchmark) expectedAt(tid int) float32 {
	wg := tid / 64
	lane := tid % 64
	groupsPerWave := 64 / b.AliasLanes
	addressGroup := lane & (groupsPerWave - 1)
	vectorElements := b.ArrayBytes / (b.WidthDwords * 4)
	mask := vectorElements - 1
	var sum float32
	for r := 0; r < b.Repeats; r++ {
		index := ((wg+r)*groupsPerWave + addressGroup) & mask
		word := index * b.WidthDwords
		for component := 0; component < b.WidthDwords; component++ {
			sum += b.input[word+component]
		}
	}
	return sum
}

func (b *Benchmark) Verify() {
	b.driver.MemCopyD2H(b.context, b.output, b.dOutput)
	for tid, got := range b.output {
		want := b.expectedAt(tid)
		if math.Abs(float64(got-want)) > 1e-4 {
			log.Panicf("VMEM load-shape mismatch at thread %d: got %g, want %g", tid, got, want)
		}
	}
	log.Printf("Passed!\n")
}
