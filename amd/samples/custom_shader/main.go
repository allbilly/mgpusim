// Custom shader test on MGPUSim
package main

import (
	"log"
	"os"

	"github.com/sarchlab/mgpusim/v4/amd/driver"
	"github.com/sarchlab/mgpusim/v4/amd/insts"
)

func main() {
	// Read custom kernel file
	kernelFile := os.Args[1]
	data, err := os.ReadFile(kernelFile)
	if err != nil {
		log.Fatal("Failed to read kernel:", err)
	}

	// Initialize driver
	d := driver.NewDriver()
	defer d.Close()
	d.Init()

	// Get GPU
	gpuCount := d.GetNumGPUs()
	log.Printf("Found %d GPUs\n", gpuCount)

	ctx := d.Init()
	d.SelectGPU(ctx, 0)

	// Load kernel
	kernel := insts.LoadKernelCodeObjectFromBytes(data, "vectoradd")
	if kernel == nil {
		log.Fatal("Failed to load kernel")
	}

	log.Printf("Kernel loaded: %+v\n", kernel)
	log.Printf("PGM_RSRC1: 0x%08x\n", kernel.ComputePgmRsrc1)
	log.Printf("PGM_RSRC2: 0x%08x\n", kernel.ComputePgmRsrc2)
	log.Printf("VGPRs: %d\n", kernel.VgprCount)
	log.Printf("SGPRs: %d\n", kernel.SgprCount)
	log.Printf("Code entry: 0x%x\n", kernel.KernelCodeEntryByteOffset)
	log.Printf("Code size: %d bytes\n", len(kernel.Data))

	// Allocate buffers
	size := uint64(1024)
	hA := make([]float32, size)
	hB := make([]float32, size)
	hC := make([]float32, size)

	for i := range hB {
		hB[i] = float32(i)
		hC[i] = float32(i) * 100.0
	}

	dA := d.AllocateMemory(ctx, size*4)
	dB := d.AllocateMemory(ctx, size*4)
	dC := d.AllocateMemory(ctx, size*4)

	d.MemCopyH2D(ctx, dB, hB)
	d.MemCopyH2D(ctx, dC, hC)

	// Create queue and launch
	queue := d.CreateCommandQueue(ctx)

	kernelArgs := struct {
		A      uint64
		B      uint64
		C      uint64
		Width  int32
		Height int32
	}{
		A:      dA,
		B:      dB,
		C:      dC,
		Width:  int32(size),
		Height: 1,
	}

	d.EnqueueLaunchKernel(
		queue,
		kernel,
		[3]uint32{uint32(size), 1, 1},
		[3]uint16{64, 1, 1},
		&kernelArgs,
	)

	d.DrainCommandQueue(queue)

	// Copy result back
	d.MemCopyD2H(ctx, hA, dA)

	// Verify
	mismatch := false
	for i := range hA {
		expected := hB[i] + hC[i]
		if hA[i] != expected {
			log.Printf("mismatch at %d: expected %f, got %f\n", i, expected, hA[i])
			mismatch = true
			if i > 10 {
				break
			}
		}
	}

	if mismatch {
		log.Fatal("Verification FAILED")
	}

	log.Printf("First 5 results: %v\n", hA[:5])
	log.Printf("PASSED - Custom kernel works on MGPUSim!\n")
}
