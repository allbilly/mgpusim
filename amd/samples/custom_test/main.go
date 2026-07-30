// Custom shader test on MGPUSim - Fixed for V5 kernel
package main

import (
	"flag"
	"log"
	"os"

	"github.com/sarchlab/mgpusim/v5/amd/driver"
	"github.com/sarchlab/mgpusim/v5/amd/insts"
	"github.com/sarchlab/mgpusim/v5/amd/samples/runner"
)

// KernelArgs matches the gfx90c HSACO kernarg layout (88 bytes).
// See .note NT_AMDGPU_METADATA in vectoradd_gfx90c_v3.hsaco.
type KernelArgs struct {
	A    driver.Ptr // offset 0
	B    driver.Ptr // offset 8
	C    driver.Ptr // offset 16
	Size int32      // offset 24
	_    int32      // offset 28 (padding before int64 hidden offsets)
	HiddenGlobalOffsetX int64 // offset 32
	HiddenGlobalOffsetY int64 // offset 40
	HiddenGlobalOffsetZ int64 // offset 48
	_    [24]byte // offset 56-79 (hidden_none x3)
	_    [8]byte  // offset 80-87 (hidden_multigrid_sync_arg)
}

type Benchmark struct {
	driver  *driver.Driver
	context *driver.Context
	kernel  *insts.KernelCodeObject
	gpus    []int
	kernelName string

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
var kernelNameFlag = flag.String("name", "vectoradd", "Kernel function name")

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

	kernel := insts.LoadKernelCodeObjectFromBytes(data, *kernelNameFlag)
	if kernel == nil {
		log.Fatal("Failed to load kernel: ", *kernelNameFlag)
	}

	log.Printf("Kernel loaded: %s", *kernelNameFlag)
	log.Printf("  Kernarg size: %d bytes", kernel.KernargSegmentByteSize)
	log.Printf("  PGM_RSRC1: 0x%08x", kernel.ComputePgmRsrc1)
	log.Printf("  PGM_RSRC2: 0x%08x", kernel.ComputePgmRsrc2)
	log.Printf("  Code size: %d bytes", len(kernel.Data))

	b := &Benchmark{
		driver:     d,
		context:    ctx,
		kernel:     kernel,
		kernelName: *kernelNameFlag,
		gpus:       []int{1},
		Width:      uint32(*widthFlag),
		Height:     uint32(*heightFlag),
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
	b.Verify()
}

func (b *Benchmark) exec() {
	queue := b.driver.CreateCommandQueue(b.context)
	numData := b.Width * b.Height

	workgroupSize := uint32(64)

	args := KernelArgs{
		A:                   b.dA,
		B:                   b.dB,
		C:                   b.dC,
		Size:                int32(numData),
		HiddenGlobalOffsetX: 0,
		HiddenGlobalOffsetY: 0,
		HiddenGlobalOffsetZ: 0,
	}

	b.driver.EnqueueLaunchKernel(
		queue,
		b.kernel,
		[3]uint32{numData, 1, 1},
		[3]uint16{uint16(workgroupSize), 1, 1},
		args,
	)

	b.driver.DrainCommandQueue(queue)
	b.driver.MemCopyD2H(b.context, b.hA, b.dA)
}

func (b *Benchmark) Verify() {
	numData := b.Width * b.Height
	errors := 0
	for i := uint32(0); i < numData; i++ {
		expected := b.hB[i] + b.hC[i]
		if b.hA[i] != expected {
			errors++
			if errors <= 5 {
				log.Printf("Mismatch at %d: got %f, expected %f", i, b.hA[i], expected)
			}
		}
	}
	if errors == 0 {
		log.Printf("✓ Verification PASSED")
	} else {
		log.Printf("✗ Verification FAILED: %d mismatches", errors)
	}
}
