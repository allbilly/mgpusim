package cu

import (
	"github.com/sarchlab/akita/v5/mem"
	"github.com/sarchlab/akita/v5/messaging"
	"github.com/sarchlab/akita/v5/modeling"
	"github.com/sarchlab/akita/v5/timing"
	"github.com/sarchlab/mgpusim/v5/amd/emu"
)

// VALUTiming describes wave-level issue intervals and result latencies for
// major VALU instruction classes. Zero values retain the legacy latency
// derived from NumSinglePrecisionUnits.
type VALUTiming struct {
	DefaultIssueInterval         int `json:"default_issue_interval"`
	DefaultResultLatency         int `json:"default_result_latency"`
	FMAIssueInterval             int `json:"fma_issue_interval"`
	FMAResultLatency             int `json:"fma_result_latency"`
	IntegerMultiplyIssueInterval int `json:"integer_multiply_issue_interval"`
	IntegerMultiplyResultLatency int `json:"integer_multiply_result_latency"`
	TranscendentalIssueInterval  int `json:"transcendental_issue_interval"`
	TranscendentalResultLatency  int `json:"transcendental_result_latency"`
	FP64IssueInterval            int `json:"fp64_issue_interval"`
	FP64ResultLatency            int `json:"fp64_result_latency"`
	MaxInFlight                  int `json:"max_in_flight"`
}

// Port names of the ComputeUnit. The port instances are created externally
// (by the platform configuration or the test setup) and supplied with
// AssignPort after Build.
const (
	// DispatchPortName receives MapWGReqs from the dispatcher and sends
	// WGCompletionMsgs back (v4: ToACE / "Top").
	DispatchPortName = "Top"

	// CtrlPortName receives pipeline flush/restart requests from the command
	// processor (v4: ToCP / "Ctrl").
	CtrlPortName = "Ctrl"

	// InstMemPortName sends instruction-fetch reads (v4: ToInstMem).
	InstMemPortName = "InstMem"

	// ScalarMemPortName sends scalar memory reads (v4: ToScalarMem).
	ScalarMemPortName = "ScalarMem"

	// VectorMemPortName sends vector memory reads/writes (v4: ToVectorMem).
	VectorMemPortName = "VectorMem"
)

// numWfPools is the number of wavefront pools in a compute unit. The v4
// implementation hard-coded 4 pools; this is preserved.
const numWfPools = 4

// Spec is the immutable configuration of a timing ComputeUnit.
type Spec struct {
	Freq timing.Freq `json:"freq"`

	// SIMDCount is the number of SIMD units in the compute unit.
	SIMDCount int `json:"simd_count"`

	// WfPoolSize is the number of wavefronts each wavefront pool can hold.
	WfPoolSize int `json:"wf_pool_size"`

	// VGPRCounts is the number of vector registers per SIMD unit. Its length
	// must equal SIMDCount.
	VGPRCounts []int `json:"vgpr_counts"`

	// SGPRCount is the number of scalar registers in the compute unit.
	SGPRCount int `json:"sgpr_count"`

	// LDSBytes is the size of the local data share in bytes.
	LDSBytes int `json:"lds_bytes"`

	// Log2CachelineSize is the cacheline size as a power of 2.
	Log2CachelineSize uint64 `json:"log2_cacheline_size"`

	// NumSinglePrecisionUnits is the number of single-precision units per
	// SIMD. GCN3 uses 16, CDNA3 uses 32.
	NumSinglePrecisionUnits int `json:"num_single_precision_units"`

	// VecMemInstPipelineStages is the number of stages in the vector memory
	// instruction pipeline (v4 used CyclePerStage=1, so the total latency in
	// cycles equals the stage count).
	VecMemInstPipelineStages int `json:"vec_mem_inst_pipeline_stages"`

	// VecMemTransPipelineStages is the number of stages in the vector memory
	// transaction pipeline.
	VecMemTransPipelineStages int `json:"vec_mem_trans_pipeline_stages"`

	// VecMemTransPipelineWidth is the number of transactions that can enter
	// the transaction pipeline per cycle.
	VecMemTransPipelineWidth int `json:"vec_mem_trans_pipeline_width"`

	// MemPipelineBufferSize is the capacity of the post-pipeline buffer for
	// vector memory transactions.
	MemPipelineBufferSize int `json:"mem_pipeline_buffer_size"`

	// MaxCoalescingPenalty is the maximum low-utilization penalty for reads.
	// MaxWriteCoalescingPenalty optionally overrides it for partial-line
	// writes, which can require write combining or read-modify-write traffic.
	// MaxWideWriteStridePenalty is a calibrated per-gap cost for non-adjacent
	// cache-line transactions emitted by one dynamic wide-store instruction.
	// The optional far tier replaces, rather than adds to, this cost when the
	// aligned-line distance reaches MaxWideWriteStrideFarMinDistanceLines.
	MaxCoalescingPenalty int `json:"max_coalescing_penalty"`
	// SplitLineLoadPenalty is a calibrated alignment overhead for contiguous,
	// unaligned vector loads spanning multiple cache lines. It is not an
	// extra-transaction counter.
	SplitLineLoadPenalty                  int  `json:"split_line_load_penalty"`
	SplitLineLoadMaxDwords                int  `json:"split_line_load_max_dwords"`
	DependentLoadIssuePenalty             int  `json:"dependent_load_issue_penalty"`
	DependentLoadMinAge                   int  `json:"dependent_load_min_age"`
	DependentLoadMaxAge                   int  `json:"dependent_load_max_age"`
	DependentLoadMaxDwords                int  `json:"dependent_load_max_dwords"`
	DependentLoadFlatOnly                 bool `json:"dependent_load_flat_only"`
	MaxWriteCoalescingPenalty             int  `json:"max_write_coalescing_penalty"`
	MaxWideWriteStridePenalty             int  `json:"max_wide_write_stride_penalty"`
	MaxWideWriteStrideFarPenalty          int  `json:"max_wide_write_stride_far_penalty"`
	MaxWideWriteStrideFarMinDistanceLines int  `json:"max_wide_write_stride_far_min_distance_lines"`

	// VMemReturnFanoutLaneDwordsPerCycle models the per-instruction bandwidth
	// for broadcasting a returned source dword to duplicate lane destinations.
	// Zero disables the model and preserves legacy immediate retirement.
	VMemReturnFanoutLaneDwordsPerCycle int `json:"vmem_return_fanout_lane_dwords_per_cycle"`

	// VMemLoadReturnLaneDwordsPerCycle models the per-instruction bandwidth
	// for retiring all lane-dwords returned by a vector load. Zero disables
	// the model and preserves legacy immediate retirement. It is mutually
	// exclusive with VMemReturnFanoutLaneDwordsPerCycle.
	VMemLoadReturnLaneDwordsPerCycle int `json:"vmem_load_return_lane_dwords_per_cycle"`

	// VMemWideLoadReturnLaneDwordsPerCycle models the per-wave bandwidth for
	// retiring only the lane-dwords beyond one dword per active lane. Zero
	// disables the model and preserves legacy immediate retirement. It is
	// mutually exclusive with the other vector-memory return models.
	VMemWideLoadReturnLaneDwordsPerCycle int `json:"vmem_wide_load_return_lane_dwords_per_cycle"`

	// VMemCUWideReturnUnitsPerCycle models a single work-conserving wide-load
	// return path shared by every wave and SIMD in a CU. Zero disables it. It
	// is mutually exclusive with all other vector-memory return models.
	VMemCUWideReturnUnitsPerCycle int `json:"vmem_cu_wide_return_units_per_cycle"`

	// VMemCUWideReturnConcurrentWaves is the number of distinct wave return
	// FIFOs that the CU-wide model can service in one cycle. Zero selects one
	// wave for backward compatibility when the CU-wide model is enabled.
	VMemCUWideReturnConcurrentWaves int `json:"vmem_cu_wide_return_concurrent_waves"`

	// VMemCUWideReturnBurstConcurrentWaves enables burst-sensitive scheduling.
	// Singleton waves are always serviced, while at most this many waves with
	// multiple modeled-wide loads outstanding are serviced each cycle. Zero
	// preserves the static concurrent-wave scheduler.
	VMemCUWideReturnBurstConcurrentWaves int `json:"vmem_cu_wide_return_burst_concurrent_waves"`

	// VMemCUWideReturnBurstAssistInterval intermittently grants a second burst
	// wave when burst concurrency is exactly one. Zero disables the assist.
	VMemCUWideReturnBurstAssistInterval int `json:"vmem_cu_wide_return_burst_assist_interval"`

	// RegisterScoreboard enables the register scoreboard and SIMD pipelining
	// feature.
	RegisterScoreboard bool `json:"register_scoreboard"`

	// ScoreboardVALULatency overrides VALU scoreboard busy cycles when > 0.
	// Use when writeback latency exceeds SIMD execution latency so dependent
	// VALU instructions stall (e.g. GCN3-class GPUs at 4 cyc/instr).
	ScoreboardVALULatency int `json:"scoreboard_valu_latency"`

	// Flat VALU timing fields keep the Spec checkpoint-compatible while
	// separating issue throughput from dependent-result latency.
	VALUDefaultIssueInterval         int `json:"valu_default_issue_interval"`
	VALUDefaultResultLatency         int `json:"valu_default_result_latency"`
	VALUFMAIssueInterval             int `json:"valu_fma_issue_interval"`
	VALUFMAResultLatency             int `json:"valu_fma_result_latency"`
	VALUIntegerMultiplyIssueInterval int `json:"valu_integer_multiply_issue_interval"`
	VALUIntegerMultiplyResultLatency int `json:"valu_integer_multiply_result_latency"`
	VALUTranscendentalIssueInterval  int `json:"valu_transcendental_issue_interval"`
	VALUTranscendentalResultLatency  int `json:"valu_transcendental_result_latency"`
	VALUFP64IssueInterval            int `json:"valu_fp64_issue_interval"`
	VALUFP64ResultLatency            int `json:"valu_fp64_result_latency"`
	VALUMaxInFlight                  int `json:"valu_max_in_flight"`

	// LDSPipelineLatency is the number of cycles the LDS execution stage holds
	// a wavefront before writeback.
	LDSPipelineLatency int `json:"lds_pipeline_latency"`

	// LDSIssueInterval controls how often the LDS unit can accept a new
	// wavefront instruction. LDSMaxInFlight separates this throughput from
	// dependent-result latency. Zero values preserve legacy serialization.
	LDSIssueInterval int `json:"lds_issue_interval"`
	LDSMaxInFlight   int `json:"lds_max_in_flight"`

	// LDSBankCount and LDSBankWidth describe the physical LDS banking.
	// LDSBankConflictPenalty is charged for each additional distinct address
	// mapped to the busiest bank in either half-wave.
	LDSBankCount           int `json:"lds_bank_count"`
	LDSBankWidth           int `json:"lds_bank_width"`
	LDSBankConflictPenalty int `json:"lds_bank_conflict_penalty"`

	// BarrierLatency is the release latency after the final wavefront in a
	// work-group reaches S_BARRIER.
	BarrierLatency int `json:"barrier_latency"`

	// InFlightVectorMemAccessLimit caps the number of outstanding vector
	// memory transactions.
	InFlightVectorMemAccessLimit int `json:"in_flight_vector_mem_access_limit"`

	// InstBufByteSize is the per-wavefront instruction buffer size that the
	// fetch arbiter fills up to.
	InstBufByteSize int `json:"inst_buf_byte_size"`
}

// State is the pure, serializable runtime state of a timing ComputeUnit.
//
// The complex runtime structures (wavefront pools, in-flight access records,
// sub-unit pipeline contents) hold pointers and live on the ComputeUnit
// middleware instead. // TODO(akita5): state purity
type State struct {
	// InstMem is the port instruction fetches are sent to.
	InstMem messaging.RemotePort `json:"inst_mem"`

	// ScalarMem is the port scalar memory accesses are sent to.
	ScalarMem messaging.RemotePort `json:"scalar_mem"`

	// Running indicates that at least one work-group has been mapped.
	Running bool `json:"running"`

	IsFlushing                   bool `json:"is_flushing"`
	IsPaused                     bool `json:"is_paused"`
	IsSendingOutShadowBufferReqs bool `json:"is_sending_out_shadow_buffer_reqs"`
	IsHandlingWfCompletionEvent  bool `json:"is_handling_wf_completion_event"`

	// HasFlushReq, FlushReqID, and FlushReqSrc record the pipeline flush
	// request currently being served (v4: currentFlushReq).
	HasFlushReq bool                 `json:"has_flush_req"`
	FlushReqID  uint64               `json:"flush_req_id"`
	FlushReqSrc messaging.RemotePort `json:"flush_req_src"`

	// HasPendingCPRsp, PendingCPRspTo, and PendingCPRspDst record the flush
	// response waiting to be sent to the command processor (v4: toSendToCP).
	HasPendingCPRsp bool                 `json:"has_pending_cp_rsp"`
	PendingCPRspTo  uint64               `json:"pending_cp_rsp_to"`
	PendingCPRspDst messaging.RemotePort `json:"pending_cp_rsp_dst"`
}

// Resources holds the shared references that a timing ComputeUnit needs.
type Resources struct {
	// Decoder decodes raw instruction bytes. Defaults to
	// insts.NewDisassembler() when nil at Build time.
	Decoder emu.Decoder

	// ALU executes the instructions. Defaults to gcn3.NewALU(nil) when nil at
	// Build time.
	ALU emu.ALU

	// VectorMemModules maps addresses to the ports that serve them.
	VectorMemModules mem.AddressToPortMapper
}

// SetVALUTiming copies a timing profile into the checkpointable flat fields.
func (s *Spec) SetVALUTiming(t VALUTiming) {
	s.VALUDefaultIssueInterval = t.DefaultIssueInterval
	s.VALUDefaultResultLatency = t.DefaultResultLatency
	s.VALUFMAIssueInterval = t.FMAIssueInterval
	s.VALUFMAResultLatency = t.FMAResultLatency
	s.VALUIntegerMultiplyIssueInterval = t.IntegerMultiplyIssueInterval
	s.VALUIntegerMultiplyResultLatency = t.IntegerMultiplyResultLatency
	s.VALUTranscendentalIssueInterval = t.TranscendentalIssueInterval
	s.VALUTranscendentalResultLatency = t.TranscendentalResultLatency
	s.VALUFP64IssueInterval = t.FP64IssueInterval
	s.VALUFP64ResultLatency = t.FP64ResultLatency
	s.VALUMaxInFlight = t.MaxInFlight
}

// VALUTimingSpec reconstructs the convenient timing profile view.
func (s Spec) VALUTimingSpec() VALUTiming {
	return VALUTiming{
		DefaultIssueInterval:         s.VALUDefaultIssueInterval,
		DefaultResultLatency:         s.VALUDefaultResultLatency,
		FMAIssueInterval:             s.VALUFMAIssueInterval,
		FMAResultLatency:             s.VALUFMAResultLatency,
		IntegerMultiplyIssueInterval: s.VALUIntegerMultiplyIssueInterval,
		IntegerMultiplyResultLatency: s.VALUIntegerMultiplyResultLatency,
		TranscendentalIssueInterval:  s.VALUTranscendentalIssueInterval,
		TranscendentalResultLatency:  s.VALUTranscendentalResultLatency,
		FP64IssueInterval:            s.VALUFP64IssueInterval,
		FP64ResultLatency:            s.VALUFP64ResultLatency,
		MaxInFlight:                  s.VALUMaxInFlight,
	}
}

// Comp is the timing ComputeUnit component.
type Comp = modeling.Component[Spec, State, Resources]

// MiddlewareOf returns the ComputeUnit middleware attached to a timing
// compute-unit component. It is used by external code (e.g., the ISA
// debugger) that needs access to the register files and other complex
// runtime state.
func MiddlewareOf(comp *Comp) *ComputeUnit {
	for _, mw := range comp.Middlewares() {
		if cuMW, ok := mw.(*ComputeUnit); ok {
			return cuMW
		}
	}

	panic("cu: component does not carry a ComputeUnit middleware")
}

// DispatcherView adapts a timing ComputeUnit to the interface that the
// command processor's resource pool expects from a dispatchable CU.
type DispatcherView struct {
	CU *Comp
}

// DispatchingPort returns the port that the dispatcher can use to dispatch
// work-groups to the CU.
func (v DispatcherView) DispatchingPort() messaging.RemotePort {
	return v.CU.GetPortByName(DispatchPortName).AsRemote()
}

// ControlPort returns the port that can receive controlling messages from
// the Command Processor.
func (v DispatcherView) ControlPort() messaging.RemotePort {
	return v.CU.GetPortByName(CtrlPortName).AsRemote()
}

// WfPoolSizes returns an array of the numbers of wavefronts that each SIMD
// unit can execute.
func (v DispatcherView) WfPoolSizes() []int {
	sizes := make([]int, numWfPools)
	for i := range sizes {
		sizes[i] = v.CU.Spec().WfPoolSize
	}

	return sizes
}

// VRegCounts returns an array of the numbers of vector registers in each
// SIMD unit.
func (v DispatcherView) VRegCounts() []int {
	return v.CU.Spec().VGPRCounts
}

// SRegCount returns the number of scalar registers in the Compute Unit.
func (v DispatcherView) SRegCount() int {
	return v.CU.Spec().SGPRCount
}

// LDSBytes returns the number of bytes in the LDS of the CU.
func (v DispatcherView) LDSBytes() int {
	return v.CU.Spec().LDSBytes
}
