// Package storestride implements a full-cache-line vector-store stride probe.
package storestride

import (
	_ "embed"
	"fmt"
	"log"

	"github.com/sarchlab/mgpusim/v5/amd/arch"
	"github.com/sarchlab/mgpusim/v5/amd/driver"
	"github.com/sarchlab/mgpusim/v5/amd/insts"
)

const (
	cacheLineBytes = uint64(64)
	linesPerWave   = uint64(16)
	wordsPerLine   = uint64(16)
)

var valueMasks = [4]uint32{0, 0x13579bdf, 0x2468ace0, 0xa5a5a5a5}

// KernelArgs is the 20-byte ABI of full_line_store_stride_kernel.
type KernelArgs struct {
	Output                driver.Ptr // offset 0
	StrideLines           uint32     // offset 8
	AllocationStrideLines uint32     // offset 12
	Repeats               uint32     // offset 16
}

// Benchmark defines the full-line store-stride probe.
type Benchmark struct {
	driver  *driver.Driver
	context *driver.Context
	queue   *driver.CommandQueue
	hsaco   *insts.KernelCodeObject
	gpus    []int

	Arch                  arch.Type
	StrideLines           int
	AllocationStrideLines int
	Repeats               int
	Workgroups            int

	output      driver.Ptr
	outputWords int
	epochSpan   uint64

	useUnifiedMemory bool
}

//go:embed kernels_gfx90c.hsaco
var gcn5HSACOBytes []byte

// NewBenchmark creates a store-stride benchmark.
func NewBenchmark(driver *driver.Driver) *Benchmark {
	b := &Benchmark{driver: driver}
	b.context = driver.Init()
	b.queue = driver.CreateCommandQueue(b.context)
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
	if b.StrideLines <= 0 {
		b.StrideLines = 1
	}
	if b.AllocationStrideLines <= 0 {
		b.AllocationStrideLines = 128
	}
	if b.Repeats <= 0 {
		b.Repeats = 32
	}
	if b.Workgroups <= 0 {
		b.Workgroups = 1
	}
	if b.StrideLines > b.AllocationStrideLines {
		log.Panic("store stride must not exceed allocation stride")
	}

	maxUint32 := uint64(^uint32(0))
	if uint64(b.StrideLines) > maxUint32 ||
		uint64(b.AllocationStrideLines) > maxUint32 ||
		uint64(b.Repeats) > maxUint32 ||
		uint64(b.Workgroups) > maxUint32/64 {
		log.Panic("store-stride geometry exceeds the kernel ABI")
	}

	b.epochSpan = (linesPerWave-1)*uint64(b.AllocationStrideLines) + 1
	maxUint64 := ^uint64(0)
	if uint64(b.Repeats) > maxUint64/uint64(b.Workgroups) {
		log.Panic("store-stride epoch count overflows")
	}
	epochs := uint64(b.Repeats) * uint64(b.Workgroups)
	if epochs > maxUint64/b.epochSpan ||
		epochs*b.epochSpan > maxUint64/cacheLineBytes {
		log.Panic("store-stride allocation size overflows")
	}
	if epochs*b.epochSpan > maxUint32/4 {
		log.Panic("store-stride vector index exceeds 32 bits")
	}
	bytes := epochs * b.epochSpan * cacheLineBytes
	maxInt := uint64(^uint(0) >> 1)
	if bytes == 0 || bytes%4 != 0 || bytes/4 > maxInt {
		log.Panic("store-stride output allocation is invalid or too large")
	}
	b.outputWords = int(bytes / 4)
}

func (b *Benchmark) loadProgram() {
	b.hsaco = insts.LoadKernelCodeObjectFromBytes(
		gcn5HSACOBytes, "full_line_store_stride_kernel")
	if b.hsaco == nil {
		log.Panic("failed to load full-line store-stride kernel")
	}
}

// Run executes the probe.
func (b *Benchmark) Run() {
	if b.Arch != arch.GCN5 {
		log.Panic("the store-stride probe currently requires -arch gcn5")
	}
	if len(b.gpus) != 1 {
		log.Panic("the store-stride probe requires exactly one GPU")
	}

	b.setDefaultsAndValidate()
	b.loadProgram()
	b.driver.SelectGPU(b.context, b.gpus[0])
	b.initMem()
	b.exec()
}

func (b *Benchmark) initMem() {
	bytes := uint64(b.outputWords) * 4
	if b.useUnifiedMemory {
		b.output = b.driver.AllocateUnifiedMemory(b.context, bytes)
	} else {
		b.output = b.driver.AllocateMemory(b.context, bytes)
	}
	// Output-only pages must be mapped before the kernel writes them. The same
	// zero initialization also makes holes verifiable and keeps DMA footprint
	// invariant across the actual-stride sweep.
	b.driver.MemCopyH2D(b.context, b.output, make([]uint32, b.outputWords))
}

func (b *Benchmark) exec() {
	args := KernelArgs{
		Output:                b.output,
		StrideLines:           uint32(b.StrideLines),
		AllocationStrideLines: uint32(b.AllocationStrideLines),
		Repeats:               uint32(b.Repeats),
	}
	b.driver.EnqueueLaunchKernel(
		b.queue,
		b.hsaco,
		[3]uint32{uint32(b.Workgroups * 64), 1, 1},
		[3]uint16{64, 1, 1},
		&args,
	)
	b.driver.DrainCommandQueue(b.queue)
}

func (b *Benchmark) expectedOutput() []uint32 {
	expected := make([]uint32, b.outputWords)
	for workgroup := 0; workgroup < b.Workgroups; workgroup++ {
		for repeat := 0; repeat < b.Repeats; repeat++ {
			epoch := uint64(workgroup*b.Repeats + repeat)
			for lane := 0; lane < 64; lane++ {
				lineInEpoch := uint64(lane >> 2)
				slotInLine := uint64(lane & 3)
				line := epoch*b.epochSpan +
					lineInEpoch*uint64(b.StrideLines)
				word := line*wordsPerLine + slotInLine*4
				tag := uint32(epoch*64 + uint64(lane) + 1)
				for component, mask := range valueMasks {
					expected[int(word)+component] = tag ^ mask
				}
			}
		}
	}
	return expected
}

// Verify checks every stored word and every zero-filled gap.
func (b *Benchmark) Verify() {
	actual := make([]uint32, b.outputWords)
	b.driver.MemCopyD2H(b.context, actual, b.output)
	expected := b.expectedOutput()
	for i := range expected {
		if actual[i] != expected[i] {
			log.Panic(fmt.Sprintf(
				"store-stride mismatch at word %d: got %#x, want %#x",
				i, actual[i], expected[i]))
		}
	}
	log.Printf("Passed!\n")
}
