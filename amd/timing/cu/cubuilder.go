package cu

import (
	"fmt"

	"github.com/sarchlab/akita/v5/mem/memprotocol"
	"github.com/sarchlab/akita/v5/modeling"
	"github.com/sarchlab/akita/v5/queueing"
	"github.com/sarchlab/akita/v5/timing"
	"github.com/sarchlab/mgpusim/v5/amd/emu/gcn3"
	"github.com/sarchlab/mgpusim/v5/amd/insts"
)

var defaultSpec = Spec{
	Freq:                                  1 * timing.GHz,
	SIMDCount:                             4,
	WfPoolSize:                            10,
	VGPRCounts:                            []int{16384, 16384, 16384, 16384},
	SGPRCount:                             3200,
	LDSBytes:                              64 * 1024,
	Log2CachelineSize:                     6,
	NumSinglePrecisionUnits:               16,
	VecMemInstPipelineStages:              6,
	VecMemTransPipelineStages:             10,
	VecMemTransPipelineWidth:              1,
	MemPipelineBufferSize:                 8,
	MaxCoalescingPenalty:                  0,
	SplitLineLoadPenalty:                  0,
	SplitLineLoadMaxDwords:                0,
	DependentLoadIssuePenalty:             0,
	DependentLoadMinAge:                   0,
	DependentLoadMaxAge:                   0,
	DependentLoadMaxDwords:                0,
	DependentLoadFlatOnly:                 false,
	MaxWriteCoalescingPenalty:             0,
	MaxWideWriteStridePenalty:             0,
	MaxWideWriteStrideFarPenalty:          0,
	MaxWideWriteStrideFarMinDistanceLines: 0,
	VMemReturnFanoutLaneDwordsPerCycle:    0,
	VMemLoadReturnLaneDwordsPerCycle:      0,
	VMemWideLoadReturnLaneDwordsPerCycle:  0,
	VMemCUWideReturnUnitsPerCycle:         0,
	VMemCUWideReturnConcurrentWaves:       0,
	VMemCUWideReturnBurstConcurrentWaves:  0,
	VMemCUWideReturnBurstAssistInterval:   0,
	RegisterScoreboard:                    false,
	LDSPipelineLatency:                    14,
	LDSIssueInterval:                      0,
	LDSMaxInFlight:                        1,
	LDSBankCount:                          0,
	LDSBankWidth:                          4,
	LDSBankConflictPenalty:                0,
	BarrierLatency:                        0,
	InFlightVectorMemAccessLimit:          512,
	InstBufByteSize:                       256,
}

// DefaultSpec returns a copy of the default compute-unit configuration.
// Callers obtain it, tweak the fields they care about, and pass it to
// WithSpec.
func DefaultSpec() Spec {
	return defaultSpec
}

// A Builder can construct a fully functional Compute Unit. Configuration is
// supplied as a whole through WithSpec; shared references (decoder, ALU,
// vector memory address mapper) through WithResources; wiring through
// WithRegistrar. The component declares its "Top", "Ctrl", "InstMem",
// "ScalarMem", and "VectorMem" ports; the port instances are supplied
// externally after Build with AssignPort. The destinations of instruction
// and scalar memory accesses are set after Build through comp.State.InstMem
// and comp.State.ScalarMem.
type Builder struct {
	spec      Spec
	resources Resources
	registrar modeling.Registrar
}

// MakeBuilder returns a builder seeded with the default spec.
func MakeBuilder() Builder {
	return Builder{spec: defaultSpec}
}

// WithRegistrar wires the builder to a registrar (a *simulation.Simulation
// in platform assembly, or modeling.NewStandaloneRegistrar(engine) in
// isolated tests).
func (b Builder) WithRegistrar(reg modeling.Registrar) Builder {
	b.registrar = reg
	return b
}

// WithSpec sets the entire configuration. Start from DefaultSpec() and tweak.
func (b Builder) WithSpec(spec Spec) Builder {
	b.spec = spec
	return b
}

// WithVMemReturnFanoutLaneDwordsPerCycle sets the per-instruction bandwidth
// for broadcasting returned dwords to duplicate vector-lane destinations.
// Zero disables the model.
func (b Builder) WithVMemReturnFanoutLaneDwordsPerCycle(n int) Builder {
	b.spec.VMemReturnFanoutLaneDwordsPerCycle = n
	return b
}

// WithVMemLoadReturnLaneDwordsPerCycle sets the per-instruction bandwidth
// for retiring all lane-dwords returned by a vector load. Zero disables the
// model.
func (b Builder) WithVMemLoadReturnLaneDwordsPerCycle(n int) Builder {
	b.spec.VMemLoadReturnLaneDwordsPerCycle = n
	return b
}

// WithVMemWideLoadReturnLaneDwordsPerCycle sets the per-wave bandwidth for
// retiring lane-dwords beyond one dword per active lane. Zero disables it.
func (b Builder) WithVMemWideLoadReturnLaneDwordsPerCycle(n int) Builder {
	b.spec.VMemWideLoadReturnLaneDwordsPerCycle = n
	return b
}

// WithVMemCUWideReturnUnitsPerCycle sets the work budget of the CU-wide
// work-conserving wide-load return FIFO. Zero disables it.
func (b Builder) WithVMemCUWideReturnUnitsPerCycle(n int) Builder {
	b.spec.VMemCUWideReturnUnitsPerCycle = n
	return b
}

// WithVMemCUWideReturnConcurrentWaves sets the number of distinct wave
// return FIFOs that the CU-wide model may service per cycle. Zero selects one
// wave when that model is enabled.
func (b Builder) WithVMemCUWideReturnConcurrentWaves(n int) Builder {
	b.spec.VMemCUWideReturnConcurrentWaves = n
	return b
}

// WithVMemCUWideReturnBurstConcurrentWaves enables burst-sensitive service
// and sets the number of burst wave queues selected per cycle. Zero preserves
// static concurrent-wave scheduling.
func (b Builder) WithVMemCUWideReturnBurstConcurrentWaves(n int) Builder {
	b.spec.VMemCUWideReturnBurstConcurrentWaves = n
	return b
}

// WithVMemCUWideReturnBurstAssistInterval sets the contended-tick cadence for
// intermittently servicing a second burst wave. Zero disables the assist.
func (b Builder) WithVMemCUWideReturnBurstAssistInterval(n int) Builder {
	b.spec.VMemCUWideReturnBurstAssistInterval = n
	return b
}

// WithResources sets the shared references of the compute unit. Decoder and
// ALU default to insts.NewDisassembler() and gcn3.NewALU(nil) when left nil.
func (b Builder) WithResources(resources Resources) Builder {
	b.resources = resources
	return b
}

// Build returns a newly constructed compute unit according to the
// configuration.
func (b Builder) Build(name string) *Comp {
	if b.registrar == nil {
		panic("cu: WithRegistrar is required")
	}

	b.mustHaveValidSpec()
	b.fillResourceDefaults()

	comp := modeling.NewBuilder[Spec, State, Resources]().
		WithEngine(b.registrar.GetEngine()).
		WithFreq(b.spec.Freq).
		WithSpec(b.spec).
		WithResources(b.resources).
		Build(name)
	comp.State = State{}

	cuMW := &ComputeUnit{
		comp:                  comp,
		engine:                b.registrar.GetEngine(),
		wfCompletionHandlerID: name + ".WfCompletion",
		wfDispatchHandlerID:   name + ".WfDispatch",
		Decoder:               b.resources.Decoder,
		wftime:                make(map[uint64]timing.VTimeInPicoSec),
	}
	cuMW.InFlightVectorMemAccessLimit = b.spec.InFlightVectorMemAccessLimit
	cuMW.vmemReturnFanoutLaneDwordsPerCycle =
		b.spec.VMemReturnFanoutLaneDwordsPerCycle
	cuMW.vmemLoadReturnLaneDwordsPerCycle =
		b.spec.VMemLoadReturnLaneDwordsPerCycle
	cuMW.vmemWideLoadReturnLaneDwordsPerCycle =
		b.spec.VMemWideLoadReturnLaneDwordsPerCycle
	cuMW.vmemCUWideReturnUnitsPerCycle =
		b.spec.VMemCUWideReturnUnitsPerCycle
	cuMW.vmemCUWideReturnConcurrentWaves =
		b.spec.VMemCUWideReturnConcurrentWaves
	cuMW.vmemCUWideReturnBurstConcurrentWaves =
		b.spec.VMemCUWideReturnBurstConcurrentWaves
	cuMW.vmemCUWideReturnBurstAssistInterval =
		b.spec.VMemCUWideReturnBurstAssistInterval

	wfDispatcher := NewWfDispatcher(cuMW)
	wfDispatcher.scoreboardEnabled = b.spec.RegisterScoreboard
	wfDispatcher.memoryDependencyTrackingEnabled =
		b.spec.DependentLoadIssuePenalty > 0
	cuMW.WfDispatcher = wfDispatcher

	for i := 0; i < numWfPools; i++ {
		cuMW.WfPools = append(cuMW.WfPools, NewWavefrontPool(b.spec.WfPoolSize))
	}

	b.equipScheduler(cuMW)
	b.equipScalarUnits(cuMW)
	b.equipSIMDUnits(cuMW, name)
	b.equipLDSUnit(cuMW)
	b.equipVectorMemoryUnit(cuMW, name)
	b.equipRegisterFiles(cuMW)

	comp.AddMiddleware(cuMW)

	comp.DeclarePort(DispatchPortName)
	comp.DeclarePort(CtrlPortName)
	comp.DeclarePort(InstMemPortName, memprotocol.Requester)
	comp.DeclarePort(ScalarMemPortName, memprotocol.Requester)
	comp.DeclarePort(VectorMemPortName, memprotocol.Requester)

	if hr, ok := b.registrar.GetEngine().(timing.HandlerRegistrar); ok {
		hr.RegisterHandler(cuMW.wfCompletionHandlerID, cuMW)
		hr.RegisterHandler(cuMW.wfDispatchHandlerID, cuMW)
	}

	b.registrar.RegisterComponent(comp)

	return comp
}

func (b *Builder) mustHaveValidSpec() {
	if len(b.spec.VGPRCounts) != b.spec.SIMDCount {
		panic("cu: VGPRCounts must have a length that equals to the SIMDCount")
	}
	if b.spec.DependentLoadIssuePenalty < 0 {
		panic("cu: DependentLoadIssuePenalty cannot be negative")
	}
	if b.spec.DependentLoadIssuePenalty > 0 &&
		b.spec.DependentLoadMaxAge < 1 {
		panic("cu: DependentLoadMaxAge must be positive when dependency tracking is enabled")
	}
	if b.spec.DependentLoadMinAge < 0 ||
		b.spec.DependentLoadMinAge > b.spec.DependentLoadMaxAge {
		panic("cu: dependent-load age window is invalid")
	}
	if b.spec.DependentLoadMaxDwords < 0 {
		panic("cu: DependentLoadMaxDwords cannot be negative")
	}
	if b.spec.SplitLineLoadMaxDwords < 0 {
		panic("cu: SplitLineLoadMaxDwords cannot be negative")
	}
	if b.spec.MaxWideWriteStridePenalty < 0 ||
		b.spec.MaxWideWriteStrideFarPenalty < 0 {
		panic("cu: wide-write stride penalties cannot be negative")
	}
	if b.spec.MaxWideWriteStrideFarPenalty > 0 &&
		b.spec.MaxWideWriteStridePenalty == 0 {
		panic("cu: wide-write far penalty requires a near penalty")
	}
	if b.spec.MaxWideWriteStrideFarPenalty > 0 &&
		b.spec.MaxWideWriteStrideFarMinDistanceLines < 2 {
		panic("cu: wide-write far penalty requires a distance of at least 2 lines")
	}
	if b.spec.MaxWideWriteStrideFarPenalty == 0 &&
		b.spec.MaxWideWriteStrideFarMinDistanceLines != 0 {
		panic("cu: wide-write far distance requires a far penalty")
	}
	if b.spec.VMemReturnFanoutLaneDwordsPerCycle < 0 {
		panic("cu: VMemReturnFanoutLaneDwordsPerCycle cannot be negative")
	}
	if b.spec.VMemLoadReturnLaneDwordsPerCycle < 0 {
		panic("cu: VMemLoadReturnLaneDwordsPerCycle cannot be negative")
	}
	if b.spec.VMemWideLoadReturnLaneDwordsPerCycle < 0 {
		panic("cu: VMemWideLoadReturnLaneDwordsPerCycle cannot be negative")
	}
	if b.spec.VMemCUWideReturnUnitsPerCycle < 0 {
		panic("cu: VMemCUWideReturnUnitsPerCycle cannot be negative")
	}
	if b.spec.VMemCUWideReturnConcurrentWaves < 0 {
		panic("cu: VMemCUWideReturnConcurrentWaves cannot be negative")
	}
	if b.spec.VMemCUWideReturnBurstConcurrentWaves < 0 {
		panic("cu: VMemCUWideReturnBurstConcurrentWaves cannot be negative")
	}
	if b.spec.VMemCUWideReturnBurstAssistInterval < 0 {
		panic("cu: VMemCUWideReturnBurstAssistInterval cannot be negative")
	}
	if b.spec.VMemCUWideReturnBurstAssistInterval > 0 &&
		b.spec.VMemCUWideReturnUnitsPerCycle == 0 {
		panic("cu: VMemCUWideReturnBurstAssistInterval requires the CU-wide return model")
	}
	if b.spec.VMemCUWideReturnBurstAssistInterval > 0 &&
		b.spec.VMemCUWideReturnBurstConcurrentWaves != 1 {
		panic("cu: VMemCUWideReturnBurstAssistInterval requires burst concurrency one")
	}
	if b.spec.VMemCUWideReturnBurstAssistInterval > 0 &&
		b.spec.VMemCUWideReturnConcurrentWaves > 0 {
		panic("cu: VMemCUWideReturnBurstAssistInterval requires static concurrency zero")
	}
	if b.spec.VMemCUWideReturnBurstConcurrentWaves > 0 &&
		b.spec.VMemCUWideReturnUnitsPerCycle == 0 {
		panic("cu: VMemCUWideReturnBurstConcurrentWaves requires the CU-wide return model")
	}
	if b.spec.VMemCUWideReturnBurstConcurrentWaves > 0 &&
		b.spec.VMemCUWideReturnConcurrentWaves > 0 {
		panic("cu: static and burst CU-wide concurrent-wave controls are mutually exclusive")
	}
	if b.spec.VMemCUWideReturnUnitsPerCycle == 0 &&
		b.spec.VMemCUWideReturnConcurrentWaves > 0 {
		panic("cu: VMemCUWideReturnConcurrentWaves requires the CU-wide return model")
	}
	if b.spec.VMemCUWideReturnUnitsPerCycle > 0 &&
		b.spec.VMemCUWideReturnConcurrentWaves == 0 &&
		b.spec.VMemCUWideReturnBurstConcurrentWaves == 0 {
		b.spec.VMemCUWideReturnConcurrentWaves = 1
	}
	configuredReturnModels := 0
	for _, bandwidth := range []int{
		b.spec.VMemReturnFanoutLaneDwordsPerCycle,
		b.spec.VMemLoadReturnLaneDwordsPerCycle,
		b.spec.VMemWideLoadReturnLaneDwordsPerCycle,
		b.spec.VMemCUWideReturnUnitsPerCycle,
	} {
		if bandwidth > 0 {
			configuredReturnModels++
		}
	}
	if configuredReturnModels > 1 {
		panic("cu: vector-memory return bandwidth models are mutually exclusive")
	}
}

func (b *Builder) fillResourceDefaults() {
	if b.resources.Decoder == nil {
		b.resources.Decoder = insts.NewDisassembler()
	}

	if b.resources.ALU == nil {
		b.resources.ALU = gcn3.NewALU(nil)
	}
}

func (b *Builder) equipScheduler(cu *ComputeUnit) {
	fetchArbitor := new(FetchArbiter)
	fetchArbitor.InstBufByteSize = b.spec.InstBufByteSize
	issueArbitor := new(IssueArbiter)
	issueArbitor.scoreboardEnabled = b.spec.RegisterScoreboard
	issueArbitor.dependentLoadIssuePenalty =
		b.spec.DependentLoadIssuePenalty
	issueArbitor.dependentLoadMinAge = b.spec.DependentLoadMinAge
	issueArbitor.dependentLoadMaxAge = b.spec.DependentLoadMaxAge
	issueArbitor.dependentLoadMaxDwords = b.spec.DependentLoadMaxDwords
	issueArbitor.dependentLoadFlatOnly = b.spec.DependentLoadFlatOnly
	scheduler := NewScheduler(cu, fetchArbitor, issueArbitor)
	scheduler.scoreboardEnabled = b.spec.RegisterScoreboard
	scheduler.scoreboardVALULatency = b.spec.ScoreboardVALULatency
	cu.Scheduler = scheduler
}

func (b *Builder) equipScalarUnits(cu *ComputeUnit) {
	cu.BranchUnit = NewBranchUnit(cu, b.resources.ALU)

	scalarDecoder := NewDecodeUnit(cu)
	scalarDecoder.stage = "decode_scalar"
	cu.ScalarDecoder = scalarDecoder
	scalarUnit := NewScalarUnit(cu, b.resources.ALU)
	scalarUnit.log2CachelineSize = b.spec.Log2CachelineSize
	cu.ScalarUnit = scalarUnit
	for i := 0; i < b.spec.SIMDCount; i++ {
		scalarDecoder.AddExecutionUnit(scalarUnit)
	}
}

func (b *Builder) equipSIMDUnits(cu *ComputeUnit, name string) {
	vectorDecoder := NewDecodeUnit(cu)
	vectorDecoder.stage = "decode_vector"
	cu.VectorDecoder = vectorDecoder
	for i := 0; i < b.spec.SIMDCount; i++ {
		simdName := fmt.Sprintf(name+".SIMD%d", i)
		simdUnit := NewSIMDUnit(cu, simdName, b.resources.ALU)
		simdUnit.NumSinglePrecisionUnit = b.spec.NumSinglePrecisionUnits
		simdUnit.scoreboardEnabled = b.spec.RegisterScoreboard
		if b.spec.RegisterScoreboard {
			capacity := b.spec.VALUMaxInFlight
			if capacity < 1 {
				capacity = 1
			}
			simdUnit.pipelineCapacity = capacity
			simdUnit.pipelineSlots =
				make([]*simdPipelineSlot, 0, capacity)
		}
		vectorDecoder.AddExecutionUnit(simdUnit)
		cu.SIMDUnit = append(cu.SIMDUnit, simdUnit)
	}
}

func (b *Builder) equipLDSUnit(cu *ComputeUnit) {
	ldsDecoder := NewDecodeUnit(cu)
	ldsDecoder.stage = "decode_lds"
	cu.LDSDecoder = ldsDecoder

	ldsUnit := NewLDSUnit(cu, b.resources.ALU)
	cu.LDSUnit = ldsUnit

	for i := 0; i < b.spec.SIMDCount; i++ {
		ldsDecoder.AddExecutionUnit(ldsUnit)
	}
}

func (b *Builder) equipVectorMemoryUnit(cu *ComputeUnit, name string) {
	vectorMemDecoder := NewDecodeUnit(cu)
	vectorMemDecoder.stage = "decode_vmem"
	cu.VectorMemDecoder = vectorMemDecoder

	coalescer := &defaultCoalescer{
		log2CacheLineSize: b.spec.Log2CachelineSize,
	}
	vectorMemoryUnit := NewVectorMemoryUnit(cu, coalescer)
	vectorMemoryUnit.maxCoalescingPenalty = b.spec.MaxCoalescingPenalty
	vectorMemoryUnit.splitLineLoadPenalty = b.spec.SplitLineLoadPenalty
	vectorMemoryUnit.splitLineLoadMaxDwords = b.spec.SplitLineLoadMaxDwords
	vectorMemoryUnit.maxWriteCoalescingPenalty =
		b.spec.MaxWriteCoalescingPenalty
	vectorMemoryUnit.maxWideWriteStridePenalty =
		b.spec.MaxWideWriteStridePenalty
	vectorMemoryUnit.maxWideWriteStrideFarPenalty =
		b.spec.MaxWideWriteStrideFarPenalty
	vectorMemoryUnit.maxWideWriteStrideFarMinDistanceLines =
		b.spec.MaxWideWriteStrideFarMinDistanceLines
	cu.VectorMemUnit = vectorMemoryUnit

	vectorMemoryUnit.postInstructionPipelineBuffer =
		queueing.NewBuffer[vectorMemInst](
			name+".VectorMemoryUnit.PostInstPipelineBuffer",
			4*b.spec.SIMDCount)
	// v4 used CyclePerStage=1, so the v5 stage count equals the v4 stage
	// count and the total latency is preserved.
	vectorMemoryUnit.instructionPipeline = queueing.NewPipeline[vectorMemInst](
		b.spec.SIMDCount, b.spec.VecMemInstPipelineStages)

	pipelineWidth := b.spec.VecMemTransPipelineWidth
	if pipelineWidth < 1 {
		pipelineWidth = 1
	}
	bufSize := b.spec.MemPipelineBufferSize
	if bufSize < 8 {
		bufSize = 8
	}
	vectorMemoryUnit.postTransactionPipelineBuffer =
		queueing.NewBuffer[VectorMemAccessInfo](
			name+".VectorMemoryUnit.PostTransPipelineBuffer", bufSize)
	vectorMemoryUnit.transactionPipeline =
		queueing.NewPipeline[VectorMemAccessInfo](
			pipelineWidth, b.spec.VecMemTransPipelineStages)

	for i := 0; i < b.spec.SIMDCount; i++ {
		vectorMemDecoder.AddExecutionUnit(vectorMemoryUnit)
	}
}

func (b *Builder) equipRegisterFiles(cu *ComputeUnit) {
	sRegFile := NewSimpleRegisterFile(uint64(b.spec.SGPRCount*4), 0)
	cu.SRegFile = sRegFile

	for i := 0; i < b.spec.SIMDCount; i++ {
		vRegFile := NewSimpleRegisterFile(uint64(b.spec.VGPRCounts[i]*4), 1024)
		cu.VRegFile = append(cu.VRegFile, vRegFile)
	}
}
