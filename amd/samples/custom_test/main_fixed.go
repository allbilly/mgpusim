// Custom shader test on MGPUSim - Fixed kernarg
package main

import (
	"flag"
	"log"
	"os"

	"github.com/sarchlab/mgpusim/v4/amd/driver"
	"github.com/sarchlab/mgpusim/v4/amd/insts"
	"github.com/sarchlab/mgpusim/v4/amd/samples/runner"
)

// KernelArgs must match the kernel's expected layout
// HIP kernel expects 288 bytes with specific alignment
type KernelArgs struct {
	A      uint64  // offset 0
	B      uint64  // offset 8
	C      uint64  // offset 16
	_      [240]byte // padding to make 256 bytes
	Width  int32   // offset 256
	Height int32   // offset 260
	_      [24]byte // padding to reach 288
}

// Benchmark wraps the driver for custom kernel testing
type Benchmark struct {
	driver  *driver.Driver
	context *driver.Context
	kernel  *insts.KernelCodeObject
	gpus    []int

	Width  uint32
	Height uint32

	hA []float32
	hB []float32
	hC []float32

	dA driver.Ptr
	dB driver.Ptr
	dC driver.Ptr
}

var widthFlag = flag.Uint("width", 1024, "Width of vectors")
var heightFlag = flag.Uint("height", 1024, "Height of vectors")
var kernelFile = flag.String("kernel", "", "Kernel file to load")

func main() {
	flag.Parse()

	r := new(runner.Runner).Init()
	d := r.Driver()

	ctx := d.Init()
	d.SelectGPU(ctx, 0)

	if *kernelFile == "" {
		log.Fatal("Please specify --kernel <file>")
	}

	data, err := os.ReadFile(*kernelFile)
	if err != nil {
		log.Fatal("Failed to read kernel:", err)
	}

	kernel := insts.LoadKernelCodeObjectFromBytes(data, "vectoradd")
	if kernel == nil {
		log.Fatal("Failed to load kernel")
	}

	log.Printf("Kernel loaded: vectoradd")
	log.Printf("  Kernarg size: %d bytes", kernel.KernargSegmentByteSize)
	log.Printf("  PGM_RSRC1: 0x%08x", kernel.ComputePgmRsrc1)
	log.Printf("  PGM_RSRC2: 0x%08x", kernel.ComputePgmRsrc2)

	b := &Benchmark{
		driver:  d,
		context: ctx,
		kernel:  kernel,
		gpus:    []int{1},
		Width:   uint32(*widthFlag),
		Height:  uint32(*heightFlag),
	}

	r.AddBenchmark(b)
	r.Run()
}

func (b *Benchmark) SelectGPU(gpus []int) { b.gpus = gpus }
func (b *Benchmark) SetUnifiedMemory()    {}

func (b *Benchmark) initMem() {
	numData := b.Width * b.Height

	b.hA = make([]float32, numData)
	b.hB = make([]float32, numData)
	b.hC = make([]float32, numData)

	for i := uint32(0); i < numData; i++ {
		b.hB[i] = float32(i)
		b.hC[i] = float32(i) * 100.0
	}

	b.dA = b.driver.AllocateMemory(b.context, uint64(numData*4))
	b.driver.Distribute(b.context, b.dA, uint64(numData*4), b.gpus)
	b.dB = b.driver.AllocateMemory(b.context, uint64(numData*4))
	b.driver.Distribute(b.context, b.dB, uint64(numData*4), b.gpus)
	b.dC = b.driver.AllocateMemory(b.context, uint64(numData*4))
	b.driver.Distribute(b.context, b.dC, uint64(numData*4), b.gpus)

	b.driver.MemCopyH2D(b.context, b.dB, b.hB)
	b.driver.MemCopyH2D(b.context, b.dC, b.hC)
}

func (b *Benchmark) Run() {
	b.driver.SelectGPU(b.context, b.gpus[0])
	b.initMem()
	b.exec()
}

func (b *Benchmark) exec() {
	queue := b.driver.CreateCommandQueue(b.context)
	numData := b.Width * b.Height

	kernArg := KernelArgs{
		A:      uint64(b.dA),
		B:      uint64(b.dB),
		C:      uint64(b.dC),
		Width:  int32(b.Width),
		Height: int32(b.Height),
	}

	b.driver.EnqueueLaunchKernel(
		queue,
		b.kernel,
		[3]uint32{numData, 1, 1},
		[3]uint16{64, 1, 1},
		&kernArg,
	)

	b.driver.DrainCommandQueue(queue)
	b.driver.MemCopyD2H(b.context, b.hA, b.dA)
}

func (b *Benchmark) Verify() {
	numData := b.Width * b.Height
	for i := uint32(0); i < numData; i++ {
		expected := b.hB[i] + b.hC[i]
		if b.hA[i] != expected {
			log.Printf("Mismatch at %d: got %f, expected %f", i, b.hA[i], expected)
		}
	}
	log.Printf("Verification complete")
}
