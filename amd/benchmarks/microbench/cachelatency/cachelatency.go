// Package cachelatency implements the cache_latency microbenchmark, ported
// from sarchlab/gpu_benchmarks (tier1/cache_latency) for the MGPUSim MI300X
// (CDNA3 / gfx942) model.
//
// A single thread walks a linked-list-style index chain where each array
// element holds the index of the next element to visit. Because each load
// depends on the previous result (a true data dependency), the loads cannot
// overlap, directly exposing per-access memory latency rather than bandwidth.
// The thread writes its final index to result[0].
//
// The kernel binary is compiled for gfx942 (CDNA3), gfx804 (GCN4), and gfx90c
// (GCN5); see native/ and the embedded kernels_gfx*.hsaco files.
package cachelatency

import (
	"log"

	// embed hsaco files
	_ "embed"

	"github.com/sarchlab/mgpusim/v5/amd/arch"
	"github.com/sarchlab/mgpusim/v5/amd/driver"
	"github.com/sarchlab/mgpusim/v5/amd/insts"
)

// KernelArgs defines the kernel arguments for the gfx942 (CDNA3) kernel.
//
// The layout is verified against the compiled kernel's AMDGPU metadata
// (kernarg_segment_size = 24): one 8-byte global_buffer pointer, two 4-byte
// by_value scalars, then another 8-byte global_buffer pointer, packed with
// no padding (mgpusim serializes args with binary.Write, which inserts no
// alignment padding). The kernel reads only blockIdx.x / threadIdx.x in a
// guard, so no hidden ABI arguments are emitted.
type KernelArgs struct {
	Arr         driver.Ptr // offset 0
	StartIdx    uint32     // offset 8
	NumAccesses uint32     // offset 12
	Result      driver.Ptr // offset 16
}

// VectorKernelArgs is the 32-byte ABI layout of vector_pointer_chase_kernel.
type VectorKernelArgs struct {
	Arr          driver.Ptr // offset 0
	StartIndices driver.Ptr // offset 8
	NumAccesses  uint32     // offset 16
	ActiveLanes  uint32     // offset 20
	Result       driver.Ptr // offset 24
}

// Benchmark defines the cache_latency benchmark.
type Benchmark struct {
	driver  *driver.Driver
	context *driver.Context
	queue   *driver.CommandQueue
	hsaco   *insts.KernelCodeObject
	gpus    []int

	Arch arch.Type

	// ArrayBytes is the size of the pointer-chasing array in bytes.
	ArrayBytes int
	// CachelineBytes is the chase granularity: one chase node per cacheline, so
	// consecutive accesses land on distinct lines (cache footprint == ArrayBytes).
	// Matches the real benchmark (gfx942 line = 64 B).
	CachelineBytes int
	// MeasureLaps sets the timed access count when NumAccesses is auto (<=0):
	// clamp(MeasureLaps * lines, minTimedAccesses, maxTimedAccesses), matching the
	// real benchmark so total kernel time is comparable.
	MeasureLaps int
	// NumAccesses is the number of dependent loads the single thread performs.
	// If <=0 it is derived from MeasureLaps (see initMem); an explicit value wins.
	NumAccesses int
	// Seed controls the deterministic chain permutation.
	Seed uint32
	// ActiveLanes selects the vector-memory pointer chase when set to 1..64.
	// Zero preserves the original single-thread scalar-cache probe.
	ActiveLanes int

	n            int // total array elements (ArrayBytes/4)
	m            int // number of cacheline chase nodes (n/stride)
	stride       int // element stride between nodes (CachelineBytes/4)
	startIdx     uint32
	startIndices []uint32
	chain        []uint32

	gArr          driver.Ptr
	gStartIndices driver.Ptr
	gResult       driver.Ptr

	useUnifiedMemory bool
}

//go:embed kernels_gfx942.hsaco
var cdna3HSACOBytes []byte

//go:embed kernels_gfx804.hsaco
var gcn4HSACOBytes []byte

//go:embed kernels_gfx90c.hsaco
var gcn5HSACOBytes []byte

// NewBenchmark returns a new cache_latency benchmark.
func NewBenchmark(driver *driver.Driver) *Benchmark {
	b := new(Benchmark)

	b.driver = driver
	b.context = b.driver.Init()
	b.queue = driver.CreateCommandQueue(b.context)

	return b
}

func (b *Benchmark) loadProgram() {
	var hsacoBytes []byte
	switch b.Arch {
	case arch.CDNA3:
		hsacoBytes = cdna3HSACOBytes
	case arch.GCN4:
		hsacoBytes = gcn4HSACOBytes
	case arch.GCN5:
		hsacoBytes = gcn5HSACOBytes
	default:
		log.Panic("the cache_latency benchmark requires -arch cdna3, gcn4, or gcn5")
	}

	symbol := "pointer_chase_kernel"
	if b.ActiveLanes > 0 {
		if b.Arch != arch.GCN5 {
			log.Panic("the vector cache_latency probe currently requires -arch gcn5")
		}
		symbol = "vector_pointer_chase_kernel"
	}
	b.hsaco = insts.LoadKernelCodeObjectFromBytes(hsacoBytes, symbol)
	if b.hsaco == nil {
		log.Panic("Failed to load kernel binary")
	}
}

// SelectGPU selects the GPUs to run on. cache_latency uses a single GPU.
func (b *Benchmark) SelectGPU(gpus []int) {
	b.gpus = gpus
}

// SetUnifiedMemory requests the use of unified memory.
func (b *Benchmark) SetUnifiedMemory() {
	b.useUnifiedMemory = true
}

// Run runs the benchmark.
func (b *Benchmark) Run() {
	if b.Arch != arch.CDNA3 && b.Arch != arch.GCN4 && b.Arch != arch.GCN5 {
		log.Panic("the cache_latency benchmark requires -arch cdna3, gcn4, or gcn5")
	}

	b.loadProgram()

	b.driver.SelectGPU(b.context, b.gpus[0])
	b.initMem()
	b.exec()
}

// Timed-access bounds, matching the real benchmark: a floor so tiny resident
// sizes still get enough laps (and the first cold lap is amortized), and a
// ceiling so the largest sizes stay bounded.
const (
	minTimedAccesses = 131072
	maxTimedAccesses = 16000000
)

// buildChain builds a random Hamiltonian cycle over CACHELINE-strided nodes,
// matching the real benchmark: only the first word of each cacheline is a chase
// node (element indices {0, stride, 2*stride, ...}), so consecutive accesses
// land on distinct cachelines (no intra-line spatial reuse) while the touched
// lines still span the whole array (cache footprint == ArrayBytes). Non-node
// slots stay zero; the chase only ever visits node slots.
//
// The permutation uses a deterministic Fisher-Yates shuffle driven by a
// splitmix-style PRNG so the cycle is exactly reproducible in Verify.
func (b *Benchmark) buildChain() {
	m, s := b.m, uint32(b.stride)

	perm := make([]uint32, m)
	for i := 0; i < m; i++ {
		perm[i] = uint32(i)
	}

	state := uint64(b.Seed)*2654435761 + 0x9E3779B97F4A7C15
	next := func() uint64 {
		state += 0x9E3779B97F4A7C15
		z := state
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
		z = (z ^ (z >> 27)) * 0x94D049BB133111EB
		return z ^ (z >> 31)
	}

	// Fisher-Yates over the m node ordinals.
	for i := m - 1; i > 0; i-- {
		j := int(next() % uint64(i+1))
		perm[i], perm[j] = perm[j], perm[i]
	}

	b.chain = make([]uint32, b.n) // non-node slots stay 0
	for i := 0; i < m; i++ {
		node := perm[i] * s
		nextNode := perm[(i+1)%m] * s
		b.chain[node] = nextNode
	}
	b.startIdx = perm[0] * s
}

// buildVectorChains partitions the cache-line nodes evenly among active lanes.
// Lane l owns node ((ordinal*lanes)+l)*stride, so no two lanes share a line.
func (b *Benchmark) buildVectorChains() {
	lanes := b.ActiveLanes
	nodesPerLane := b.m / lanes
	b.chain = make([]uint32, b.n)
	b.startIndices = make([]uint32, lanes)

	for lane := 0; lane < lanes; lane++ {
		perm := make([]uint32, nodesPerLane)
		for i := range perm {
			perm[i] = uint32(i)
		}

		state := uint64(b.Seed+uint32(lane))*2654435761 +
			0x9E3779B97F4A7C15
		next := func() uint64 {
			state += 0x9E3779B97F4A7C15
			z := state
			z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
			z = (z ^ (z >> 27)) * 0x94D049BB133111EB
			return z ^ (z >> 31)
		}
		for i := nodesPerLane - 1; i > 0; i-- {
			j := int(next() % uint64(i+1))
			perm[i], perm[j] = perm[j], perm[i]
		}

		nodeIndex := func(ordinal uint32) uint32 {
			return (ordinal*uint32(lanes) + uint32(lane)) *
				uint32(b.stride)
		}
		for i := 0; i < nodesPerLane; i++ {
			node := nodeIndex(perm[i])
			nextNode := nodeIndex(perm[(i+1)%nodesPerLane])
			b.chain[node] = nextNode
		}
		b.startIndices[lane] = nodeIndex(perm[0])
	}
}

func (b *Benchmark) initMem() {
	if b.ArrayBytes <= 0 {
		b.ArrayBytes = 16 * 1024 // 16 KB
	}
	if b.CachelineBytes <= 0 {
		b.CachelineBytes = 64 // gfx942/MI300X cacheline
	}
	if b.MeasureLaps <= 0 {
		b.MeasureLaps = 4
	}
	if b.Seed == 0 {
		b.Seed = 42
	}

	b.n = b.ArrayBytes / 4
	b.stride = b.CachelineBytes / 4
	if b.stride < 1 {
		b.stride = 1
	}
	// Need at least two cachelines to form a cycle.
	if b.n < b.stride*2 {
		b.n = b.stride * 2
		b.ArrayBytes = b.n * 4
	}
	b.m = b.n / b.stride // cacheline nodes == touched lines
	if b.ActiveLanes < 0 || b.ActiveLanes > 64 {
		log.Panic("active lanes must be between 0 and 64")
	}
	if b.ActiveLanes > 0 && b.m/b.ActiveLanes < 2 {
		log.Panic("vector cache_latency requires at least two cache lines per active lane")
	}

	// Timed accesses: derive from MeasureLaps like the real benchmark unless an
	// explicit NumAccesses was given. The floor gives tiny resident sizes many
	// laps, so the (necessarily counted) first cold lap is amortized -- the sim
	// records cumulative GPU busy time and cannot run a separate untimed warmup
	// the way the real benchmark does.
	if b.NumAccesses <= 0 {
		na := b.MeasureLaps * b.m
		if na < minTimedAccesses {
			na = minTimedAccesses
		}
		if na > maxTimedAccesses {
			na = maxTimedAccesses
		}
		b.NumAccesses = na
	}

	if b.ActiveLanes > 0 {
		b.buildVectorChains()
	} else {
		b.buildChain()
	}

	resultLen := 1
	if b.ActiveLanes > 0 {
		resultLen = b.ActiveLanes
	}
	if b.useUnifiedMemory {
		b.gArr = b.driver.AllocateUnifiedMemory(
			b.context, uint64(b.n*4))
		if b.ActiveLanes > 0 {
			b.gStartIndices = b.driver.AllocateUnifiedMemory(
				b.context, uint64(b.ActiveLanes*4))
		}
		b.gResult = b.driver.AllocateUnifiedMemory(
			b.context, uint64(resultLen*4))
	} else {
		b.gArr = b.driver.AllocateMemory(b.context, uint64(b.n*4))
		if b.ActiveLanes > 0 {
			b.gStartIndices = b.driver.AllocateMemory(
				b.context, uint64(b.ActiveLanes*4))
		}
		b.gResult = b.driver.AllocateMemory(
			b.context, uint64(resultLen*4))
	}

	b.driver.MemCopyH2D(b.context, b.gArr, b.chain)
	if b.ActiveLanes > 0 {
		b.driver.MemCopyH2D(b.context, b.gStartIndices, b.startIndices)
	}
	// Initialize the result buffer so its page is resident on the GPU before
	// the kernel writes to it (output-only buffers are otherwise never mapped).
	b.driver.MemCopyH2D(b.context, b.gResult, make([]uint32, resultLen))
}

func (b *Benchmark) exec() {
	if b.ActiveLanes > 0 {
		args := VectorKernelArgs{
			Arr:          b.gArr,
			StartIndices: b.gStartIndices,
			NumAccesses:  uint32(b.NumAccesses),
			ActiveLanes:  uint32(b.ActiveLanes),
			Result:       b.gResult,
		}
		b.driver.EnqueueLaunchKernel(
			b.queue,
			b.hsaco,
			[3]uint32{64, 1, 1},
			[3]uint16{64, 1, 1},
			&args,
		)
		b.driver.DrainCommandQueue(b.queue)
		return
	}

	args := KernelArgs{
		Arr:         b.gArr,
		StartIdx:    b.startIdx,
		NumAccesses: uint32(b.NumAccesses),
		Result:      b.gResult,
	}

	// Launch a single work-item (grid 1, block 1), matching the HIP source.
	// The kernel guards on blockIdx.x == 0 && threadIdx.x == 0, so only one
	// thread performs the dependent pointer chase and writes result[0].
	b.driver.EnqueueLaunchKernel(
		b.queue,
		b.hsaco,
		[3]uint32{1, 1, 1},
		[3]uint16{1, 1, 1},
		&args,
	)

	b.driver.DrainCommandQueue(b.queue)
}

// Verify checks the GPU result against a CPU reference walk of the same chain.
// The thread starts at startIdx and applies num_accesses chain hops.
func (b *Benchmark) Verify() {
	resultLen := 1
	starts := []uint32{b.startIdx}
	if b.ActiveLanes > 0 {
		resultLen = b.ActiveLanes
		starts = b.startIndices
	}
	gpuResult := make([]uint32, resultLen)
	b.driver.MemCopyD2H(b.context, gpuResult, b.gResult)

	for lane, start := range starts {
		idx := start
		for i := 0; i < b.NumAccesses; i++ {
			idx = b.chain[idx]
		}
		if gpuResult[lane] != idx {
			log.Fatalf("Lane %d mismatch: expected final index %d, but got %d.\n",
				lane, idx, gpuResult[lane])
		}
	}

	log.Printf("Passed!\n")
}
