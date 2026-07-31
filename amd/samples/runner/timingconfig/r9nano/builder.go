// Package r9nano contains the configuration of GPUs similar to AMD Radeon R9
// Nano.
package r9nano

import (
	"fmt"

	"github.com/sarchlab/akita/v5/mem"
	"github.com/sarchlab/akita/v5/mem/cache/writeback"
	"github.com/sarchlab/akita/v5/mem/dram"
	"github.com/sarchlab/akita/v5/mem/idealmemcontroller"
	"github.com/sarchlab/akita/v5/mem/simplebankedmemory"
	"github.com/sarchlab/akita/v5/mem/vm/mmu"
	"github.com/sarchlab/akita/v5/mem/vm/tlb"
	"github.com/sarchlab/akita/v5/messaging"
	"github.com/sarchlab/akita/v5/modeling"
	"github.com/sarchlab/akita/v5/noc/directconnection"
	"github.com/sarchlab/akita/v5/simulation"
	"github.com/sarchlab/akita/v5/timing"
	"github.com/sarchlab/mgpusim/v5/amd/emu"
	"github.com/sarchlab/mgpusim/v5/amd/samples/runner/timingconfig/gpubuilder"
	"github.com/sarchlab/mgpusim/v5/amd/samples/runner/timingconfig/shaderarray"
	"github.com/sarchlab/mgpusim/v5/amd/timing/cp"
	"github.com/sarchlab/mgpusim/v5/amd/timing/cu"
	"github.com/sarchlab/mgpusim/v5/amd/timing/rdma"
)

// Port buffer sizes. The CP and DMA-to-CP ports mirror the v4 4096-deep
// buffers (v4 used 40M for DMA ToCP; 4096 is plenty). The other sizes mirror
// the v4 component builders' internal port sizes.
const (
	cpPortBufSize           = 4096
	dmaToCPBufSize          = 4096
	dmaToMemBufSize         = 64
	rdmaPortBufSize         = 128
	memCtrlPortBufSize      = 16
	detailedDRAMPortBufSize = 1024 // v4 dram: WithTopPortBufferSize(1024)
	l2TLBPortBufSize        = 1024
	ctrlPortBufSize         = 1
)

// dramBackendKind selects the memory controller implementation.
type dramBackendKind int

const (
	dramIdeal dramBackendKind = iota
	dramDetailed
	dramBanked
)

// unsetCPInt means a CP timing field was not overridden by the platform.
const unsetCPInt = -1

// Builder builds a hardware platform for timing simulation.
type Builder struct {
	simulation *simulation.Simulation

	gpuID                                 uint64
	name                                  string
	freq                                  timing.Freq
	numCUPerShaderArray                   int
	numShaderArray                        int
	l2CacheSize                           uint64
	numMemoryBank                         int
	log2CacheLineSize                     uint64
	log2PageSize                          uint64
	log2MemoryBankInterleavingSize        uint64
	memAddrOffset                         uint64
	dramSize                              uint64
	memoryLatency                         int
	memoryWidth                           int
	l2BankLatency                         int
	l2NumReqPerCycle                      int
	l1vCacheSize                          uint64
	l1vBankLatency                        int
	dramBackend                           dramBackendKind
	dramMemFreq                           timing.Freq
	dramNumInternalBanks                  int
	dramBankPipelineWidth                 int
	dramBankPipelineDepth                 int
	dramStageLatency                      int
	cpAlg                                 string
	cpNumDies                             int
	cpWavefrontDispatchCycles             int
	cpConstantKernelOverhead              int
	cpConstantKernelLaunchOverhead        int
	cpSubsequentKernelLaunchOverhead      int
	cpWGScalingThreshold                  int
	activeCUCount                         int
	registerScoreboard                    bool
	scoreboardVALULatency                 int
	valuTiming                            cu.VALUTiming
	ldsPipelineLatency                    int
	ldsIssueInterval                      int
	ldsMaxInFlight                        int
	ldsBankCount                          int
	ldsBankWidth                          int
	ldsBankConflictPenalty                int
	barrierLatency                        int
	maxCoalescingPenalty                  int
	splitLineLoadPenalty                  int
	splitLineLoadMaxDwords                int
	dependentLoadIssuePenalty             int
	dependentLoadMinAge                   int
	dependentLoadMaxAge                   int
	dependentLoadMaxDwords                int
	dependentLoadFlatOnly                 bool
	maxWriteCoalescingPenalty             int
	maxWideWriteStridePenalty             int
	maxWideWriteStrideFarPenalty          int
	maxWideWriteStrideFarMinDistanceLines int
	vmemReturnFanoutLaneDwordsPerCycle    int
	vmemLoadReturnLaneDwordsPerCycle      int
	vecMemTransPipelineWidth              int
	numSinglePrecisionUnits               int
	dmaThroughL2                          bool
	dmaThroughL2MaxBytes                  uint64
	aluBuilder                            func() emu.ALU
	decoderBuilder                        func() emu.Decoder
	globalStorage                         *mem.Storage
	mmu                                   *mmu.Comp
	rdmaAddressMapper                     mem.AddressToPortMapper
	driverPort                            messaging.RemotePort

	gpu                 *gpubuilder.GPU
	cp                  *cp.Comp
	rdmaEngine          *rdma.Comp
	dmaEngine           *cp.DMAComp
	sas                 []*shaderarray.ShaderArray
	l2Caches            []*writeback.Comp
	l2TLBs              []*tlb.Comp
	drams               []messaging.Component
	internalConn        *directconnection.Comp
	l2ToDramConnection  *directconnection.Comp
	l1AddressMapper     *mem.InterleavedAddressPortMapper
	l1TLBAddressMapper  *mem.SinglePortMapper
	dmaLocalDataSource  *mem.InterleavedAddressPortMapper
	dmaDirectDRAMSource *mem.InterleavedAddressPortMapper
}

// MakeBuilder creates a new builder.
func MakeBuilder() Builder {
	return Builder{
		freq:                             1 * timing.GHz,
		numCUPerShaderArray:              4,
		numShaderArray:                   16,
		l2CacheSize:                      2 * mem.MB,
		numMemoryBank:                    16,
		log2CacheLineSize:                6,
		log2PageSize:                     12,
		log2MemoryBankInterleavingSize:   7,
		memAddrOffset:                    0,
		dramSize:                         4 * mem.GB,
		cpConstantKernelOverhead:         unsetCPInt,
		cpConstantKernelLaunchOverhead:   unsetCPInt,
		cpSubsequentKernelLaunchOverhead: unsetCPInt,
		cpWGScalingThreshold:             unsetCPInt,
	}
}

// WithSimulation sets the simulation to use.
func (b Builder) WithSimulation(sim *simulation.Simulation) Builder {
	b.simulation = sim
	return b
}

// WithGPUID sets the GPU ID to use.
func (b Builder) WithGPUID(id uint64) gpubuilder.GPUBuilder {
	b.gpuID = id
	return b
}

// WithFreq sets the frequency that the GPU works at.
func (b Builder) WithFreq(freq timing.Freq) Builder {
	b.freq = freq
	return b
}

// WithLog2MemoryBankInterleavingSize sets the log2 memory bank interleaving
// size.
func (b Builder) WithLog2MemoryBankInterleavingSize(size uint64) Builder {
	b.log2MemoryBankInterleavingSize = size
	return b
}

// WithLog2CacheLineSize sets the log2 cache line size.
func (b Builder) WithLog2CacheLineSize(size uint64) Builder {
	b.log2CacheLineSize = size
	return b
}

// WithLog2PageSize sets the log2 page size.
func (b Builder) WithLog2PageSize(size uint64) Builder {
	b.log2PageSize = size
	return b
}

// WithMemAddrOffset sets the memory address offset.
func (b Builder) WithMemAddrOffset(offset uint64) gpubuilder.GPUBuilder {
	b.memAddrOffset = offset
	return b
}

// WithNumCUPerShaderArray sets the number of CUs per shader array.
func (b Builder) WithNumCUPerShaderArray(numCUPerShaderArray int) Builder {
	b.numCUPerShaderArray = numCUPerShaderArray
	return b
}

// WithNumShaderArray sets the number of shader arrays.
func (b Builder) WithNumShaderArray(numShaderArray int) Builder {
	b.numShaderArray = numShaderArray
	return b
}

// WithL2CacheSize sets the size of the L2 cache.
func (b Builder) WithL2CacheSize(size uint64) Builder {
	b.l2CacheSize = size
	return b
}

// WithNumMemoryBank sets the number of memory banks.
func (b Builder) WithNumMemoryBank(numMemoryBank int) Builder {
	b.numMemoryBank = numMemoryBank
	return b
}

// WithDramSize sets the size of the DRAM.
func (b Builder) WithDramSize(size uint64) Builder {
	b.dramSize = size
	return b
}

// WithMemoryLatency sets fixed DRAM latency in GPU cycles for the ideal memory
// controller. Zero keeps the default (100 cycles).
func (b Builder) WithMemoryLatency(latency int) Builder {
	b.memoryLatency = latency
	return b
}

// WithMemoryWidth sets how many memory requests the ideal controller accepts
// per cycle. Zero keeps the default (1).
func (b Builder) WithMemoryWidth(width int) Builder {
	b.memoryWidth = width
	return b
}

// WithL2BankLatency sets the L2 cache bank latency in GPU cycles. Zero keeps
// the writeback cache default (10 cycles).
func (b Builder) WithL2BankLatency(latency int) Builder {
	b.l2BankLatency = latency
	return b
}

// WithL2NumReqPerCycle sets how many requests each L2 bank can accept per
// cycle. Zero keeps the writeback-cache default used by this platform.
func (b Builder) WithL2NumReqPerCycle(numReq int) Builder {
	b.l2NumReqPerCycle = numReq
	return b
}

// WithL1VCacheSize sets the L1 vector cache size per CU in bytes. Zero keeps
// the shader-array default.
func (b Builder) WithL1VCacheSize(size uint64) Builder {
	b.l1vCacheSize = size
	return b
}

// WithL1VBankLatency sets the L1 vector cache bank latency in GPU cycles.
// Zero keeps the shader-array default (20 cycles).
func (b Builder) WithL1VBankLatency(latency int) Builder {
	b.l1vBankLatency = latency
	return b
}

// WithDetailedDRAM enables the cycle-accurate DDR4 DRAM controller instead of
// the ideal fixed-latency model. Use for APUs and other DDR-backed GPUs where
// memory bandwidth must be modeled.
func (b Builder) WithDetailedDRAM(enable bool) Builder {
	if enable {
		b.dramBackend = dramDetailed
	}
	return b
}

// WithBankedDRAM enables the simple banked-memory DRAM model (bandwidth-limited,
// much faster to simulate than the cycle-accurate DRAM controller). Parameters
// can be tuned with the WithDRAM* helpers.
func (b Builder) WithBankedDRAM(enable bool) Builder {
	if enable {
		b.dramBackend = dramBanked
	}
	return b
}

// WithDRAMMemFreq sets the clock of the banked DRAM controllers.
func (b Builder) WithDRAMMemFreq(freq timing.Freq) Builder {
	b.dramMemFreq = freq
	return b
}

// WithDRAMNumInternalBanks sets the number of banks inside each banked DRAM
// controller.
func (b Builder) WithDRAMNumInternalBanks(n int) Builder {
	b.dramNumInternalBanks = n
	return b
}

// WithDRAMBankPipelineWidth sets how many requests enter each bank pipeline per
// cycle in the banked DRAM model.
func (b Builder) WithDRAMBankPipelineWidth(width int) Builder {
	b.dramBankPipelineWidth = width
	return b
}

// WithDRAMBankPipelineDepth sets the depth of each bank pipeline in the banked
// DRAM model.
func (b Builder) WithDRAMBankPipelineDepth(depth int) Builder {
	b.dramBankPipelineDepth = depth
	return b
}

// WithDRAMStageLatency sets the per-stage latency in bank pipelines.
func (b Builder) WithDRAMStageLatency(latency int) Builder {
	b.dramStageLatency = latency
	return b
}

// WithCPAlg sets the work-group dispatch algorithm ("round-robin", "greedy",
// "partition", or "per-die").
func (b Builder) WithCPAlg(alg string) Builder {
	b.cpAlg = alg
	return b
}

// WithCPNumDies sets the number of parallel dispatch domains for the per-die
// algorithm (typically the number of shader arrays).
func (b Builder) WithCPNumDies(n int) Builder {
	b.cpNumDies = n
	return b
}

// WithCPWavefrontDispatchCycles sets per-wavefront dispatch cost for per-die
// dispatch.
func (b Builder) WithCPWavefrontDispatchCycles(cycles int) Builder {
	b.cpWavefrontDispatchCycles = cycles
	return b
}

// WithCPConstantKernelOverhead sets fixed post-kernel overhead in GPU cycles.
// The CP default is 3600 cycles; platforms that calibrate against hardware
// kernel time typically set this to 0.
func (b Builder) WithCPConstantKernelOverhead(overhead int) Builder {
	b.cpConstantKernelOverhead = overhead
	return b
}

// WithCPConstantKernelLaunchOverhead sets the delay before the first kernel
// begins dispatching, in GPU cycles.
func (b Builder) WithCPConstantKernelLaunchOverhead(overhead int) Builder {
	b.cpConstantKernelLaunchOverhead = overhead
	return b
}

// WithCPSubsequentKernelLaunchOverhead sets the launch delay for kernels after
// the first one, in GPU cycles.
func (b Builder) WithCPSubsequentKernelLaunchOverhead(overhead int) Builder {
	b.cpSubsequentKernelLaunchOverhead = overhead
	return b
}

// WithCPWGScalingThreshold sets the WG count above which launch overhead is
// amortized.
func (b Builder) WithCPWGScalingThreshold(threshold int) Builder {
	b.cpWGScalingThreshold = threshold
	return b
}

// WithActiveCUCount limits work-group dispatch to the requested number of CUs
// while retaining the full physical shader-array/cache topology. A value of
// zero enables every built CU.
func (b Builder) WithActiveCUCount(count int) Builder {
	b.activeCUCount = count
	return b
}

// WithRegisterScoreboard enables register RAW hazard stalls in each CU.
func (b Builder) WithRegisterScoreboard(enabled bool) Builder {
	b.registerScoreboard = enabled
	return b
}

// WithScoreboardVALULatency sets VALU scoreboard busy cycles (writeback
// latency) when register scoreboard is enabled.
func (b Builder) WithScoreboardVALULatency(latency int) Builder {
	b.scoreboardVALULatency = latency
	return b
}

// WithVALUTiming configures class-specific VALU issue and result timing.
func (b Builder) WithVALUTiming(timingSpec cu.VALUTiming) Builder {
	b.valuTiming = timingSpec
	return b
}

// WithLDSPipelineLatency sets LDS instruction execution latency in cycles.
func (b Builder) WithLDSPipelineLatency(latency int) Builder {
	b.ldsPipelineLatency = latency
	return b
}

// WithLDSThroughput separates LDS issue throughput from result latency.
func (b Builder) WithLDSThroughput(issueInterval, maxInFlight int) Builder {
	b.ldsIssueInterval = issueInterval
	b.ldsMaxInFlight = maxInFlight
	return b
}

// WithLDSBanking enables LDS bank-conflict timing.
func (b Builder) WithLDSBanking(bankCount, bankWidth, penalty int) Builder {
	b.ldsBankCount = bankCount
	b.ldsBankWidth = bankWidth
	b.ldsBankConflictPenalty = penalty
	return b
}

// WithBarrierLatency sets work-group barrier release latency in cycles.
func (b Builder) WithBarrierLatency(latency int) Builder {
	b.barrierLatency = latency
	return b
}

// WithMaxCoalescingPenalty sets the maximum per-transaction lane-utilization
// penalty in cycles.
func (b Builder) WithMaxCoalescingPenalty(penalty int) Builder {
	b.maxCoalescingPenalty = penalty
	return b
}

// WithVMemReturnFanoutLaneDwordsPerCycle sets the per-instruction bandwidth
// for broadcasting returned dwords to duplicate vector-lane destinations.
// Zero disables the model.
func (b Builder) WithVMemReturnFanoutLaneDwordsPerCycle(n int) Builder {
	b.vmemReturnFanoutLaneDwordsPerCycle = n
	return b
}

// WithVMemLoadReturnLaneDwordsPerCycle sets the per-instruction bandwidth
// for retiring all lane-dwords returned by a vector load. Zero disables the
// model.
func (b Builder) WithVMemLoadReturnLaneDwordsPerCycle(n int) Builder {
	b.vmemLoadReturnLaneDwordsPerCycle = n
	return b
}

// WithSplitLineLoadPenalty sets a calibrated alignment overhead for an
// unaligned contiguous load spanning multiple cache lines.
func (b Builder) WithSplitLineLoadPenalty(penalty int) Builder {
	b.splitLineLoadPenalty = penalty
	return b
}

// WithSplitLineLoadMaxDwords restricts the alignment overhead to loads no
// wider than n dwords. Zero disables the width restriction; negative is invalid.
func (b Builder) WithSplitLineLoadMaxDwords(n int) Builder {
	b.splitLineLoadMaxDwords = n
	return b
}

// WithDependentLoadIssuePenalty sets the delay for a vector load whose
// address was derived from a load completed within maxAge instructions.
func (b Builder) WithDependentLoadIssuePenalty(
	penalty, maxAge int,
) Builder {
	return b.WithDependentLoadIssueWindow(penalty, 0, maxAge)
}

// WithDependentLoadIssueWindow sets the delay for a vector load whose address
// dependency age is in the inclusive [minAge, maxAge] window.
func (b Builder) WithDependentLoadIssueWindow(
	penalty, minAge, maxAge int,
) Builder {
	b.dependentLoadIssuePenalty = penalty
	b.dependentLoadMinAge = minAge
	b.dependentLoadMaxAge = maxAge
	return b
}

// WithDependentLoadMaxDwords restricts the dependency delay to vector loads
// no wider than n dwords. Zero disables the width restriction; negative is invalid.
func (b Builder) WithDependentLoadMaxDwords(n int) Builder {
	b.dependentLoadMaxDwords = n
	return b
}

// WithDependentLoadFlatOnly restricts the dependency delay to FLAT/GLOBAL
// loads, excluding MUBUF scratch/spill traffic.
func (b Builder) WithDependentLoadFlatOnly(enabled bool) Builder {
	b.dependentLoadFlatOnly = enabled
	return b
}

// WithMaxWriteCoalescingPenalty sets the maximum partial-line write penalty.
func (b Builder) WithMaxWriteCoalescingPenalty(penalty int) Builder {
	b.maxWriteCoalescingPenalty = penalty
	return b
}

// WithMaxWideWriteStridePenalty sets the non-local wide-store penalty.
func (b Builder) WithMaxWideWriteStridePenalty(penalty int) Builder {
	b.maxWideWriteStridePenalty = penalty
	return b
}

// WithMaxWideWriteStrideFarPenalty sets the distant-line wide-store penalty.
func (b Builder) WithMaxWideWriteStrideFarPenalty(
	penalty, minDistanceLines int,
) Builder {
	b.maxWideWriteStrideFarPenalty = penalty
	b.maxWideWriteStrideFarMinDistanceLines = minDistanceLines
	return b
}

// WithVecMemTransPipelineWidth sets the number of vector-memory cache-line
// transactions that a CU can inject per cycle.
func (b Builder) WithVecMemTransPipelineWidth(width int) Builder {
	b.vecMemTransPipelineWidth = width
	return b
}

// WithNumSinglePrecisionUnits sets FP32 execution width per SIMD (64/units
// cycles per VALU instruction).
func (b Builder) WithNumSinglePrecisionUnits(n int) Builder {
	b.numSinglePrecisionUnits = n
	return b
}

// WithDMAThroughL2 routes host DMA (H2D/D2H) through the L2 cache instead of
// directly to DRAM. Enable for APU/unified-memory platforms where host writes
// populate the GPU cache hierarchy before kernels run.
func (b Builder) WithDMAThroughL2(enable bool) Builder {
	b.dmaThroughL2 = enable
	return b
}

// WithDMAThroughL2MaxBytes sets a per-transfer size threshold above which host
// DMA bypasses L2 and targets DRAM directly. Requires WithDMAThroughL2(true).
// Use on APUs so small buffers stay cache-warm while large copies do not fill
// L2 before bandwidth-bound kernels run.
func (b Builder) WithDMAThroughL2MaxBytes(maxBytes uint64) Builder {
	b.dmaThroughL2MaxBytes = maxBytes
	return b
}

// WithALUBuilder sets the ALU factory for each CU (e.g. GCN3 vs CDNA3).
func (b Builder) WithALUBuilder(f func() emu.ALU) Builder {
	b.aluBuilder = f
	return b
}

// WithDecoderBuilder sets the instruction decoder factory for each CU.
// GFX9+ (GCN5) needs IsCDNA3=true so FLAT/GLOBAL SADDR=0 is a valid scalar base.
func (b Builder) WithDecoderBuilder(f func() emu.Decoder) Builder {
	b.decoderBuilder = f
	return b
}

func (b *Builder) dmaHybrid() bool {
	return b.dmaThroughL2 && b.dmaThroughL2MaxBytes > 0
}

// WithMMU sets the MMU that can provide the ultimate address translation.
func (b Builder) WithMMU(mmu *mmu.Comp) Builder {
	b.mmu = mmu
	return b
}

// WithGlobalStorage sets the global storage that backs the memories of all
// the devices.
func (b Builder) WithGlobalStorage(
	globalStorage *mem.Storage,
) Builder {
	b.globalStorage = globalStorage
	return b
}

// WithRDMAAddressMapper sets the RDMA address mapper.
func (b Builder) WithRDMAAddressMapper(
	mapper mem.AddressToPortMapper,
) gpubuilder.GPUBuilder {
	b.rdmaAddressMapper = mapper
	return b
}

// WithDriverPort sets the driver port that the command processor responds
// to.
func (b Builder) WithDriverPort(
	port messaging.RemotePort,
) gpubuilder.GPUBuilder {
	b.driverPort = port
	return b
}

// Build builds the hardware platform.
func (b Builder) Build(name string) *gpubuilder.GPU {
	b.name = name

	b.l1AddressMapper = mem.NewInterleavedAddressPortMapper(
		1 << b.log2MemoryBankInterleavingSize,
	)
	b.l1AddressMapper.LowAddress = b.memAddrOffset
	b.l1AddressMapper.HighAddress = b.memAddrOffset + b.dramSize
	b.l1AddressMapper.UseAddressSpaceLimitation = true

	b.l1TLBAddressMapper = &mem.SinglePortMapper{}

	// Build order is bottom-up so the mappers the shader arrays snapshot at
	// build time (the v5 caches inline the mapper contents into their Spec)
	// are fully populated before the shader arrays are built.
	b.buildDRAMControllers()
	b.buildL2Caches()
	b.buildCP()
	b.buildL2TLB()
	b.buildSAs()

	b.connectCP()
	b.connectL2AndDRAM()
	b.connectL1ToL2()
	b.connectL1TLBToL2TLB()

	b.populateGPU()

	return b.gpu
}

// buildPort creates a port instance for a declared port and assigns it to
// the component.
func (b *Builder) buildPort(
	comp messaging.Component,
	name string,
	bufSize int,
) messaging.Port {
	port := modeling.MakePortBuilder().
		WithRegistrar(b.simulation).
		WithComponent(comp).
		WithSpec(modeling.PortSpec{BufSize: bufSize}).
		Build(name)
	comp.AssignPort(name, port)

	return port
}

func (b *Builder) populateGPU() {
	b.gpu = &gpubuilder.GPU{
		Name:                 b.name,
		CommandProcessor:     b.cp,
		CommandProcessorPort: b.cp.GetPortByName("ToDriver"),
		RDMARequestPort:      b.rdmaEngine.GetPortByName("RDMARequestOutside"),
		RDMADataPort:         b.rdmaEngine.GetPortByName("RDMADataOutside"),
	}

	for _, l2TLB := range b.l2TLBs {
		b.gpu.TranslationPorts = append(b.gpu.TranslationPorts,
			l2TLB.GetPortByName("Bottom"))
	}
}

func (b *Builder) connectCP() {
	b.internalConn = directconnection.MakeBuilder().
		WithRegistrar(b.simulation).
		WithSpec(directconnection.Spec{Freq: b.freq}).
		Build(b.name + ".InternalConn")

	b.internalConn.PlugIn(b.cp.GetPortByName("ToDMA"))
	b.internalConn.PlugIn(b.cp.GetPortByName("ToCaches"))
	b.internalConn.PlugIn(b.cp.GetPortByName("ToCUs"))
	b.internalConn.PlugIn(b.cp.GetPortByName("ToTLBs"))
	b.internalConn.PlugIn(b.cp.GetPortByName("ToAddressTranslators"))
	b.internalConn.PlugIn(b.cp.GetPortByName("ToRDMA"))

	rdmaCtrlPort := b.rdmaEngine.GetPortByName("Ctrl")
	b.cp.State.RDMA = rdmaCtrlPort.AsRemote()
	b.internalConn.PlugIn(rdmaCtrlPort)

	dmaToCPPort := b.dmaEngine.GetPortByName("ToCP")
	b.cp.State.DMAEngine = dmaToCPPort.AsRemote()
	b.internalConn.PlugIn(dmaToCPPort)

	b.connectCPWithCUs()
	b.connectCPWithAddressTranslators()
	b.connectCPWithTLBs()
	b.connectCPWithCaches()
	b.connectCPWithDRAMControllers()
}

func (b *Builder) connectCPWithCUs() {
	registered := 0
	for _, sa := range b.sas {
		for _, cuComp := range sa.CUs {
			b.internalConn.PlugIn(cuComp.GetPortByName(cu.DispatchPortName))
			b.internalConn.PlugIn(cuComp.GetPortByName(cu.CtrlPortName))

			if b.activeCUCount > 0 && registered >= b.activeCUCount {
				continue
			}

			cp.RegisterCU(b.cp, cu.DispatcherView{CU: cuComp})
			registered++
		}
	}
}

// connectCPWithAddressTranslators wires the Control ports of the address
// translators and the reorder buffers to the CP. The ROB list is new in v5:
// the CP requires the ROB control ports split from the AT control ports
// (v4 mixed them into the AT list).
func (b *Builder) connectCPWithAddressTranslators() {
	addAT := func(at messaging.Component) {
		ctrlPort := at.GetPortByName("Control")
		b.cp.State.AddressTranslators = append(
			b.cp.State.AddressTranslators, ctrlPort.AsRemote())
		b.internalConn.PlugIn(ctrlPort)
	}
	addROB := func(robComp messaging.Component) {
		ctrlPort := robComp.GetPortByName("Control")
		b.cp.State.ROBs = append(b.cp.State.ROBs, ctrlPort.AsRemote())
		b.internalConn.PlugIn(ctrlPort)
	}

	for _, sa := range b.sas {
		for i := range b.numCUPerShaderArray {
			addAT(sa.L1VATs[i])
			addROB(sa.L1VROBs[i])
		}

		addAT(sa.L1SAT)
		addROB(sa.L1SROB)

		addAT(sa.L1IAT)
		addROB(sa.L1IROB)
	}
}

func (b *Builder) connectCPWithTLBs() {
	addTLB := func(tlbComp messaging.Component) {
		ctrlPort := tlbComp.GetPortByName("Control")
		b.cp.State.TLBs = append(b.cp.State.TLBs, ctrlPort.AsRemote())
		b.internalConn.PlugIn(ctrlPort)
	}

	for _, sa := range b.sas {
		for i := range b.numCUPerShaderArray {
			addTLB(sa.L1VTLBs[i])
		}

		addTLB(sa.L1STLB)
		addTLB(sa.L1ITLB)
	}

	for _, l2TLB := range b.l2TLBs {
		addTLB(l2TLB)
	}
}

func (b *Builder) connectCPWithCaches() {
	for _, sa := range b.sas {
		for i := range b.numCUPerShaderArray {
			ctrlPort := sa.L1VCaches[i].GetPortByName("Control")
			b.cp.State.L1VCaches = append(
				b.cp.State.L1VCaches, ctrlPort.AsRemote())
			b.internalConn.PlugIn(ctrlPort)
		}

		l1sCtrlPort := sa.L1SCache.GetPortByName("Control")
		b.cp.State.L1SCaches = append(
			b.cp.State.L1SCaches, l1sCtrlPort.AsRemote())
		b.internalConn.PlugIn(l1sCtrlPort)

		l1iCtrlPort := sa.L1ICache.GetPortByName("Control")
		b.cp.State.L1ICaches = append(
			b.cp.State.L1ICaches, l1iCtrlPort.AsRemote())
		b.internalConn.PlugIn(l1iCtrlPort)
	}

	for _, c := range b.l2Caches {
		ctrlPort := c.GetPortByName("Control")
		b.cp.State.L2Caches = append(b.cp.State.L2Caches, ctrlPort.AsRemote())
		b.internalConn.PlugIn(ctrlPort)
	}
}

// connectCPWithDRAMControllers records the Control ports of the DRAM
// controllers in the CP state. The CP currently sends no commands to them,
// but the ports must be connected so the control protocol is reachable.
func (b *Builder) connectCPWithDRAMControllers() {
	for _, dramComp := range b.drams {
		ctrlPort := dramComp.GetPortByName("Control")
		b.cp.State.DRAMControllers = append(
			b.cp.State.DRAMControllers, ctrlPort.AsRemote())
		b.internalConn.PlugIn(ctrlPort)
	}
}

func (b *Builder) connectL1ToL2() {
	l1ToL2Conn := directconnection.MakeBuilder().
		WithRegistrar(b.simulation).
		WithSpec(directconnection.Spec{Freq: b.freq}).
		Build(b.name + ".L1ToL2")

	l1ToL2Conn.PlugIn(b.rdmaEngine.GetPortByName("RDMARequestInside"))
	l1ToL2Conn.PlugIn(b.rdmaEngine.GetPortByName("RDMADataInside"))

	for _, l2 := range b.l2Caches {
		topPort := l2.GetPortByName("Top")
		l1ToL2Conn.PlugIn(topPort)
		if b.dmaThroughL2 {
			// Host DMA targets the L2 so copies populate the cache hierarchy
			// before kernels run (APU unified-memory behavior).
			b.dmaLocalDataSource.LowModules = append(
				b.dmaLocalDataSource.LowModules, topPort.AsRemote())
		}
	}

	for _, sa := range b.sas {
		for i := range b.numCUPerShaderArray {
			l1ToL2Conn.PlugIn(sa.L1VCaches[i].GetPortByName("Bottom"))
		}

		l1ToL2Conn.PlugIn(sa.L1SCache.GetPortByName("Bottom"))
		// The instruction path egress to L2 is the L1I address translator's
		// bottom port (the L1I cache sits above its AT).
		l1ToL2Conn.PlugIn(sa.L1IAT.GetPortByName("Bottom"))
	}

	if b.dmaThroughL2 {
		l1ToL2Conn.PlugIn(b.dmaEngine.GetPortByName("ToMem"))
	}
}

func (b *Builder) connectL2AndDRAM() {
	b.l2ToDramConnection = directconnection.MakeBuilder().
		WithRegistrar(b.simulation).
		WithSpec(directconnection.Spec{Freq: b.freq}).
		Build(b.name + ".L2ToDRAM")

	for _, l2 := range b.l2Caches {
		b.l2ToDramConnection.PlugIn(l2.GetPortByName("Bottom"))
	}

	for _, dramComp := range b.drams {
		topPort := dramComp.GetPortByName("Top")
		b.l2ToDramConnection.PlugIn(topPort)
		switch {
		case b.dmaHybrid():
			b.dmaDirectDRAMSource.LowModules = append(
				b.dmaDirectDRAMSource.LowModules, topPort.AsRemote())
		case !b.dmaThroughL2:
			b.dmaLocalDataSource.LowModules = append(
				b.dmaLocalDataSource.LowModules, topPort.AsRemote())
		}
	}

	if !b.dmaThroughL2 {
		b.l2ToDramConnection.PlugIn(b.dmaEngine.GetPortByName("ToMem"))
	}
	if b.dmaHybrid() {
		b.l2ToDramConnection.PlugIn(b.dmaEngine.GetPortByName("ToMemDRAM"))
	}
}

func (b *Builder) connectL1TLBToL2TLB() {
	tlbConn := directconnection.MakeBuilder().
		WithRegistrar(b.simulation).
		WithSpec(directconnection.Spec{Freq: b.freq}).
		Build(b.name + ".L1TLBToL2TLB")

	tlbConn.PlugIn(b.l2TLBs[0].GetPortByName("Top"))

	for _, sa := range b.sas {
		for i := range b.numCUPerShaderArray {
			tlbConn.PlugIn(sa.L1VTLBs[i].GetPortByName("Bottom"))
		}

		tlbConn.PlugIn(sa.L1STLB.GetPortByName("Bottom"))
		tlbConn.PlugIn(sa.L1ITLB.GetPortByName("Bottom"))
	}
}

func (b *Builder) buildSAs() {
	saBuilder := shaderarray.MakeBuilder().
		WithSimulation(b.simulation).
		WithFreq(b.freq).
		WithGPUID(b.gpuID).
		WithNumCUs(b.numCUPerShaderArray).
		WithLog2CacheLineSize(b.log2CacheLineSize).
		WithLog2PageSize(b.log2PageSize).
		WithL1AddressMapper(b.l1AddressMapper).
		WithL1TLBAddressMapper(b.l1TLBAddressMapper)

	if b.l1vCacheSize > 0 {
		saBuilder = saBuilder.WithL1VCacheSize(b.l1vCacheSize)
	}
	if b.l1vBankLatency > 0 {
		saBuilder = saBuilder.WithL1VBankLatency(b.l1vBankLatency)
	}
	if b.registerScoreboard {
		saBuilder = saBuilder.WithRegisterScoreboard(true)
	}
	if b.scoreboardVALULatency > 0 {
		saBuilder = saBuilder.WithScoreboardVALULatency(b.scoreboardVALULatency)
	}
	saBuilder = saBuilder.WithVALUTiming(b.valuTiming)
	if b.ldsPipelineLatency > 0 {
		saBuilder = saBuilder.WithLDSPipelineLatency(b.ldsPipelineLatency)
	}
	if b.ldsIssueInterval > 0 {
		saBuilder = saBuilder.WithLDSThroughput(
			b.ldsIssueInterval,
			b.ldsMaxInFlight,
		)
	}
	if b.ldsBankCount > 0 {
		saBuilder = saBuilder.WithLDSBanking(
			b.ldsBankCount,
			b.ldsBankWidth,
			b.ldsBankConflictPenalty,
		)
	}
	if b.barrierLatency > 0 {
		saBuilder = saBuilder.WithBarrierLatency(b.barrierLatency)
	}
	if b.maxCoalescingPenalty > 0 {
		saBuilder = saBuilder.WithMaxCoalescingPenalty(
			b.maxCoalescingPenalty,
		)
	}
	if b.splitLineLoadPenalty > 0 {
		saBuilder = saBuilder.WithSplitLineLoadPenalty(
			b.splitLineLoadPenalty,
		).WithSplitLineLoadMaxDwords(b.splitLineLoadMaxDwords)
	}
	if b.dependentLoadIssuePenalty > 0 {
		saBuilder = saBuilder.WithDependentLoadIssueWindow(
			b.dependentLoadIssuePenalty,
			b.dependentLoadMinAge,
			b.dependentLoadMaxAge,
		).
			WithDependentLoadMaxDwords(b.dependentLoadMaxDwords).
			WithDependentLoadFlatOnly(b.dependentLoadFlatOnly)
	}
	if b.maxWriteCoalescingPenalty > 0 {
		saBuilder = saBuilder.WithMaxWriteCoalescingPenalty(
			b.maxWriteCoalescingPenalty,
		)
	}
	if b.maxWideWriteStridePenalty > 0 {
		saBuilder = saBuilder.WithMaxWideWriteStridePenalty(
			b.maxWideWriteStridePenalty,
		)
		if b.maxWideWriteStrideFarPenalty > 0 {
			saBuilder = saBuilder.WithMaxWideWriteStrideFarPenalty(
				b.maxWideWriteStrideFarPenalty,
				b.maxWideWriteStrideFarMinDistanceLines,
			)
		}
	}
	saBuilder = saBuilder.WithVMemReturnFanoutLaneDwordsPerCycle(
		b.vmemReturnFanoutLaneDwordsPerCycle,
	)
	saBuilder = saBuilder.WithVMemLoadReturnLaneDwordsPerCycle(
		b.vmemLoadReturnLaneDwordsPerCycle,
	)
	if b.vecMemTransPipelineWidth > 0 {
		saBuilder = saBuilder.WithVecMemTransPipelineWidth(
			b.vecMemTransPipelineWidth,
		)
	}
	if b.numSinglePrecisionUnits > 0 {
		saBuilder = saBuilder.WithNumSinglePrecisionUnits(b.numSinglePrecisionUnits)
	}
	if b.aluBuilder != nil {
		saBuilder = saBuilder.WithALUBuilder(b.aluBuilder)
	}
	if b.decoderBuilder != nil {
		saBuilder = saBuilder.WithDecoderBuilder(b.decoderBuilder)
	}

	for i := 0; i < b.numShaderArray; i++ {
		saName := fmt.Sprintf("%s.SA[%d]", b.name, i)
		sa := saBuilder.Build(saName)

		b.sas = append(b.sas, sa)
	}
}

func (b *Builder) buildL2Caches() {
	byteSize := b.l2CacheSize / uint64(b.numMemoryBank)

	spec := writeback.DefaultSpec()
	spec.Freq = b.freq
	spec.Log2BlockSize = b.log2CacheLineSize
	spec.WayAssociativity = 16
	spec.TotalByteSize = byteSize
	spec.NumMSHREntry = 64
	spec.NumReqPerCycle = 16
	if b.l2NumReqPerCycle > 0 {
		spec.NumReqPerCycle = b.l2NumReqPerCycle
	}
	if b.l2BankLatency > 0 {
		spec.BankLatency = b.l2BankLatency
	}

	for i := 0; i < b.numMemoryBank; i++ {
		cacheName := fmt.Sprintf("%s.L2Cache[%d]", b.name, i)
		l2 := writeback.MakeBuilder().
			WithRegistrar(b.simulation).
			WithSpec(spec).
			WithResources(writeback.Resources{
				AddressToPortMapper: &mem.SinglePortMapper{
					Port: b.drams[i].GetPortByName("Top").AsRemote(),
				},
			}).
			Build(cacheName)

		portBufSize := 2 * spec.NumReqPerCycle
		b.buildPort(l2, "Top", portBufSize)
		b.buildPort(l2, "Bottom", portBufSize)
		b.buildPort(l2, "Control", portBufSize)

		b.l2Caches = append(b.l2Caches, l2)

		b.l1AddressMapper.LowModules = append(
			b.l1AddressMapper.LowModules,
			l2.GetPortByName("Top").AsRemote(),
		)
	}
}

// buildDRAMControllers builds the memory controllers. The default is the ideal
// memory controller with a fixed latency; WithDetailedDRAM and WithBankedDRAM
// swap in cycle-accurate or banked DRAM models.
func (b *Builder) buildDRAMControllers() {
	b.dmaLocalDataSource = mem.NewInterleavedAddressPortMapper(
		1 << b.log2MemoryBankInterleavingSize)
	if b.dmaHybrid() {
		b.dmaDirectDRAMSource = mem.NewInterleavedAddressPortMapper(
			1 << b.log2MemoryBankInterleavingSize)
	}

	switch b.dramBackend {
	case dramDetailed:
		b.buildDetailedDRAMControllers()
		return
	case dramBanked:
		b.buildBankedDRAMControllers()
		return
	}

	spec := idealmemcontroller.DefaultSpec()
	spec.Freq = b.freq
	spec.Latency = 100
	if b.memoryLatency > 0 {
		spec.Latency = b.memoryLatency
	}
	spec.Width = 1
	if b.memoryWidth > 0 {
		spec.Width = b.memoryWidth
	}

	for i := 0; i < b.numMemoryBank; i++ {
		dramName := fmt.Sprintf("%s.DRAM[%d]", b.name, i)
		dramComp := idealmemcontroller.MakeBuilder().
			WithRegistrar(b.simulation).
			WithSpec(spec).
			WithResources(idealmemcontroller.Resources{
				Storage: b.globalStorage,
			}).
			Build(dramName)

		b.buildPort(dramComp, "Top", memCtrlPortBufSize)
		b.buildPort(dramComp, "Control", memCtrlPortBufSize)

		b.drams = append(b.drams, dramComp)
	}
}

func (b *Builder) buildDetailedDRAMControllers() {
	spec := b.ddr4DRAMSpec()

	for i := 0; i < b.numMemoryBank; i++ {
		dramName := fmt.Sprintf("%s.DRAM[%d]", b.name, i)
		dramComp := dram.MakeBuilder().
			WithRegistrar(b.simulation).
			WithSpec(spec).
			WithResources(dram.Resources{
				Storage: b.globalStorage,
			}).
			Build(dramName)

		b.buildPort(dramComp, "Top", detailedDRAMPortBufSize)
		b.buildPort(dramComp, "Control", ctrlPortBufSize)

		b.drams = append(b.drams, dramComp)
	}
}

func (b *Builder) buildBankedDRAMControllers() {
	memBankSize := b.dramSize / uint64(b.numMemoryBank)

	memFreq := b.dramMemFreq
	if memFreq == 0 {
		memFreq = 1 * timing.GHz
	}
	numBanks := b.dramNumInternalBanks
	if numBanks <= 0 {
		numBanks = 16
	}
	pipelineWidth := b.dramBankPipelineWidth
	if pipelineWidth <= 0 {
		pipelineWidth = 1
	}
	pipelineDepth := b.dramBankPipelineDepth
	if pipelineDepth <= 0 {
		pipelineDepth = 14
	}
	stageLatency := b.dramStageLatency
	if stageLatency <= 0 {
		stageLatency = 7
	}

	for i := 0; i < b.numMemoryBank; i++ {
		dramName := fmt.Sprintf("%s.DRAM[%d]", b.name, i)

		spec := simplebankedmemory.DefaultSpec()
		spec.Freq = memFreq
		spec.NumBanks = numBanks
		spec.BankPipelineWidth = pipelineWidth
		spec.BankPipelineDepth = pipelineDepth
		spec.StageLatency = stageLatency
		spec.PostPipelineBufSize = 128
		spec.BankSelectorKind = "interleaved"
		spec.BankSelectorLog2InterleaveSize = 6
		spec.BankAddrConvKind = "interleaving"
		spec.BankAddrInterleavingSize = 1 << b.log2MemoryBankInterleavingSize
		spec.BankAddrTotalNumOfElements = b.numMemoryBank
		spec.BankAddrCurrentElementIndex = i
		spec.Capacity = memBankSize

		dramComp := simplebankedmemory.MakeBuilder().
			WithRegistrar(b.simulation).
			WithSpec(spec).
			WithResources(simplebankedmemory.Resources{
				Storage: b.globalStorage,
			}).
			Build(dramName)

		b.buildPort(dramComp, "Top", detailedDRAMPortBufSize)
		b.buildPort(dramComp, "Control", ctrlPortBufSize)

		b.drams = append(b.drams, dramComp)
	}
}

// ddr4DRAMSpec returns a DDR4-3200 controller spec sized for one memory
// channel. Renoir-class APUs use dual 64-bit DDR4 channels; model each channel
// as one DRAM controller (numMemoryBank=2).
func (b *Builder) ddr4DRAMSpec() dram.Spec {
	memBankSize := b.dramSize / uint64(b.numMemoryBank)
	if b.dramSize%uint64(b.numMemoryBank) != 0 {
		panic("GPU memory size is not a multiple of the number of memory banks")
	}

	dramCol := 1024
	dramRow := 32768
	dramDeviceWidth := 8
	dramBankSize := dramCol * dramRow * dramDeviceWidth
	dramBank := 4
	dramBankGroup := 4
	dramBusWidth := 64
	dramDevicePerRank := dramBusWidth / dramDeviceWidth
	dramRankSize := dramBankSize * dramDevicePerRank * dramBank
	dramRank := int(memBankSize * 8 / uint64(dramRankSize))
	if dramRank < 1 {
		dramRank = 1
	}

	spec := dram.DDR4Spec
	// DDR4-3200 (1600 MHz memory clock). Peak BW per channel:
	// 1600 MHz * 64 bits * 2 (DDR) / 8 = 25.6 GB/s.
	spec.Freq = 1600 * timing.MHz
	spec.TCL = 22
	spec.TCWL = 16
	spec.TRCD = 22
	spec.TRP = 22
	spec.TRAS = 52
	spec.TCCDL = 8
	spec.TCCDS = 4
	spec.TRTP = 12
	spec.TWTRL = 12
	spec.TWTRS = 4
	spec.TWR = 24
	spec.TRRDL = 8
	spec.TRRDS = 6
	spec.TFAW = 32
	spec.BusWidth = dramBusWidth
	spec.DeviceWidth = dramDeviceWidth
	spec.NumChannel = 1
	spec.NumRank = dramRank
	spec.NumBankGroup = dramBankGroup
	spec.NumBank = dramBank
	spec.NumCol = dramCol
	spec.NumRow = dramRow
	spec.TransactionQueueSize = 64
	spec.CommandQueueCapacity = 16
	spec.ReadQueueSize = 32
	spec.WriteQueueSize = 32

	return spec
}

// hbmDRAMSpec returns the spec of a detailed HBM memory controller that
// matches the v4 dram.HBM configuration. It starts from the v5 HBM2Spec
// preset (the closest preset to v4's dram.HBM protocol) and applies the
// exact geometry and timing numbers the v4 configuration used. It is not
// used by default (the v4 configuration used the ideal memory controller),
// but is kept so the detailed model can be swapped in.
//
//nolint:unused
func (b *Builder) hbmDRAMSpec() dram.Spec {
	memBankSize := 4 * mem.GB / uint64(b.numMemoryBank)
	if 4*mem.GB%uint64(b.numMemoryBank) != 0 {
		panic("GPU memory size is not a multiple of the number of memory banks")
	}

	dramCol := 64
	dramRow := 16384
	dramDeviceWidth := 128
	dramBankSize := dramCol * dramRow * dramDeviceWidth
	dramBank := 4
	dramBankGroup := 4
	dramBusWidth := 256
	dramDevicePerRank := dramBusWidth / dramDeviceWidth
	dramRankSize := dramBankSize * dramDevicePerRank * dramBank
	dramRank := int(memBankSize * 8 / uint64(dramRankSize))

	spec := dram.HBM2Spec
	spec.Freq = 500 * timing.MHz
	spec.BurstLength = 4
	spec.DeviceWidth = dramDeviceWidth
	spec.BusWidth = dramBusWidth
	spec.NumChannel = 1
	spec.NumRank = dramRank
	spec.NumBankGroup = dramBankGroup
	spec.NumBank = dramBank
	spec.NumCol = dramCol
	spec.NumRow = dramRow
	spec.CommandQueueCapacity = 8
	spec.TransactionQueueSize = 32
	applyHBMTimings(&spec)

	return spec
}

// applyHBMTimings applies the v4 dram.HBM timing parameters onto spec.
//
//nolint:unused
func applyHBMTimings(spec *dram.Spec) {
	spec.TCL = 7
	spec.TCWL = 2
	spec.TRCDRD = 7
	spec.TRCDWR = 7
	spec.TRP = 7
	spec.TRAS = 17
	spec.TREFI = 1950
	spec.TRRDS = 2
	spec.TRRDL = 3
	spec.TWTRS = 3
	spec.TWTRL = 4
	spec.TWR = 8
	spec.TCCDS = 1
	spec.TCCDL = 1
	spec.TRTRS = 0
	spec.TRTP = 3
	spec.TPPD = 2
}

func (b *Builder) buildRDMAEngine() {
	name := fmt.Sprintf("%s.RDMA", b.name)

	spec := rdma.DefaultSpec()
	spec.Freq = 1 * timing.GHz

	b.rdmaEngine = rdma.MakeBuilder().
		WithRegistrar(b.simulation).
		WithSpec(spec).
		WithResources(rdma.Resources{
			LocalModules:           b.l1AddressMapper,
			RemoteRDMAAddressTable: b.rdmaAddressMapper,
		}).
		Build(name)

	b.buildPort(b.rdmaEngine, "RDMARequestInside", rdmaPortBufSize)
	b.buildPort(b.rdmaEngine, "RDMARequestOutside", rdmaPortBufSize)
	b.buildPort(b.rdmaEngine, "RDMADataInside", rdmaPortBufSize)
	b.buildPort(b.rdmaEngine, "RDMADataOutside", rdmaPortBufSize)
	b.buildPort(b.rdmaEngine, "Ctrl", rdmaPortBufSize)

	b.l1AddressMapper.ModuleForOtherAddresses =
		b.rdmaEngine.GetPortByName("RDMARequestInside").AsRemote()
}

func (b *Builder) buildDMAEngine() {
	resources := cp.DMAResources{
		LocalDataSource: b.dmaLocalDataSource,
	}
	if b.dmaHybrid() {
		resources.DirectDRAMSource = b.dmaDirectDRAMSource
		resources.L2BypassThreshold = b.dmaThroughL2MaxBytes
	}

	b.dmaEngine = cp.MakeDMAEngineBuilder().
		WithRegistrar(b.simulation).
		WithSpec(cp.DefaultDMASpec()).
		WithResources(resources).
		Build(fmt.Sprintf("%s.DMA", b.name))

	b.buildPort(b.dmaEngine, "ToCP", dmaToCPBufSize)
	b.buildPort(b.dmaEngine, "ToMem", dmaToMemBufSize)
	if b.dmaHybrid() {
		b.buildPort(b.dmaEngine, "ToMemDRAM", dmaToMemBufSize)
	}
}

func (b *Builder) buildCP() {
	spec := cp.DefaultSpec()
	spec.Freq = b.freq
	if b.cpAlg != "" {
		spec.Alg = b.cpAlg
	}
	if b.cpNumDies > 0 {
		spec.NumDies = b.cpNumDies
	}
	if b.cpWavefrontDispatchCycles > 0 {
		spec.WavefrontDispatchCycles = b.cpWavefrontDispatchCycles
	}
	if b.cpConstantKernelOverhead != unsetCPInt {
		spec.ConstantKernelOverhead = b.cpConstantKernelOverhead
	}
	if b.cpConstantKernelLaunchOverhead != unsetCPInt {
		spec.ConstantKernelLaunchOverhead =
			b.cpConstantKernelLaunchOverhead
	}
	if b.cpSubsequentKernelLaunchOverhead != unsetCPInt {
		spec.SubsequentKernelLaunchOverhead =
			b.cpSubsequentKernelLaunchOverhead
	}
	if b.cpWGScalingThreshold != unsetCPInt {
		spec.WGScalingThreshold = b.cpWGScalingThreshold
	}

	b.cp = cp.MakeBuilder().
		WithRegistrar(b.simulation).
		WithSpec(spec).
		WithVisTracer(b.simulation.GetVisTracer()).
		WithMonitor(b.simulation.GetMonitor()).
		WithDriver(b.driverPort).
		Build(b.name + ".CommandProcessor")

	b.buildPort(b.cp, "ToDriver", cpPortBufSize)
	b.buildPort(b.cp, "ToDMA", cpPortBufSize)
	b.buildPort(b.cp, "ToCUs", cpPortBufSize)
	b.buildPort(b.cp, "ToTLBs", cpPortBufSize)
	b.buildPort(b.cp, "ToAddressTranslators", cpPortBufSize)
	b.buildPort(b.cp, "ToCaches", cpPortBufSize)
	b.buildPort(b.cp, "ToRDMA", cpPortBufSize)

	b.buildDMAEngine()
	b.buildRDMAEngine()
}

func (b *Builder) buildL2TLB() {
	numWays := 64

	spec := tlb.DefaultSpec()
	spec.Freq = b.freq
	spec.NumWays = numWays
	spec.NumSets = int(b.dramSize / (1 << b.log2PageSize) / uint64(numWays))
	spec.MSHRSize = 64
	spec.NumReqPerCycle = 1024
	spec.Log2PageSize = b.log2PageSize

	l2TLB := tlb.MakeBuilder().
		WithRegistrar(b.simulation).
		WithSpec(spec).
		WithResources(tlb.Resources{
			TranslationProviderMapper: &mem.SinglePortMapper{
				Port: b.mmu.GetPortByName("Top").AsRemote(),
			},
		}).
		Build(fmt.Sprintf("%s.L2TLB", b.name))

	b.buildPort(l2TLB, "Top", l2TLBPortBufSize)
	b.buildPort(l2TLB, "Bottom", l2TLBPortBufSize)
	b.buildPort(l2TLB, "Control", ctrlPortBufSize)

	b.l2TLBs = append(b.l2TLBs, l2TLB)

	b.l1TLBAddressMapper.Port = l2TLB.GetPortByName("Top").AsRemote()
}
