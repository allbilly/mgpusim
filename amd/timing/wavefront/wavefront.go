package wavefront

import (
	"log"
	"math"
	"sync"

	"github.com/sarchlab/akita/v5/mem/vm"
	"github.com/sarchlab/akita/v5/timing"
	"github.com/sarchlab/mgpusim/v5/amd/insts"
	"github.com/sarchlab/mgpusim/v5/amd/kernels"
)

// RegFileAccessor provides access to register files for a wavefront.
type RegFileAccessor interface {
	ReadReg(reg *insts.Reg, regCount int, laneID int, waveOffset int) []byte
	WriteReg(reg *insts.Reg, regCount int, laneID int, waveOffset int, data []byte)
}

// WfState marks what state that wavefront it in.
type WfState int

// A list of all possible WfState
const (
	WfDispatching      WfState = iota // Dispatching in progress, not ready to run
	WfReady                           // Allow the scheduler to schedule instruction
	WfRunning                         // Instruction in fight
	WfCompleted                       // Wavefront completed
	WfAtBarrier                       // Wavefront at barrier
	WfSampledCompleted                // Wavefront completed at Sampling
)

// A Wavefront in the timing package contains the information of the progress
// of a wavefront
type Wavefront struct {
	*kernels.Wavefront
	sync.RWMutex

	WG *WorkGroup

	pid            vm.PID
	State          WfState
	inst           *Inst                 // The instruction that is being executed
	LastFetchTime  timing.VTimeInPicoSec // The time that the last instruction was fetched
	CompletedLanes int                   // The number of lanes that is completed in the SIMD unit

	InstBuffer        []byte
	InstBufferStartPC uint64
	IsFetching        bool
	InstToIssue       *Inst

	SIMDID     int
	SRegOffset int
	VRegOffset int
	LDSOffset  int

	pc   uint64
	exec uint64
	vcc  uint64
	M0   uint32
	scc  byte

	FlatScratchLo uint32
	FlatScratchHi uint32

	RegAccessor RegFileAccessor

	OutstandingScalarMemAccess int
	OutstandingVectorMemAccess int

	MemoryDependency  *MemoryDependencyState
	lastWideWriteLine uint64
	hasWideWriteLine  bool

	// InFlightInsts counts this wavefront's instruction tasks currently in
	// flight (issued but not yet completed). When it is zero the wavefront has
	// nothing executing — a real gap — and its fetch/issue stalls are
	// attributed with milestones; while it is non-zero a concurrent prefetch is
	// silent (the wavefront is making progress, so it is not blocked).
	InFlightInsts int

	// ScoreboardData holds per-wavefront register scoreboard state.
	// When register scoreboard is enabled, this contains a *cu.Scoreboard
	// (stored as interface{} to avoid circular imports).
	ScoreboardData interface{}
}

// MemoryDependencyState is allocated only on platforms that model short
// vector-load-to-address dependencies.
type MemoryDependencyState struct {
	InstructionIssueSequence uint64
	MemoryDerivedVGPR        [256]uint64
	DependencyStallPC        uint64
	DependencyStallRemaining int
	DependencyStallActive    bool
}

// NewWavefront creates a new Wavefront of the timing package, wrapping the
// wavefront from the kernels package.
func NewWavefront(raw *kernels.Wavefront) *Wavefront {
	wf := new(Wavefront)
	wf.Wavefront = raw

	wf.InstBuffer = make([]byte, 0, 256)

	return wf
}

// Inst return the instruction that is being simulated
func (wf *Wavefront) Inst() *insts.Inst {
	if wf.inst == nil {
		return nil
	}
	return wf.inst.Inst
}

// DynamicInst returns the insts with an ID
func (wf *Wavefront) DynamicInst() *Inst {
	return wf.inst
}

// SetDynamicInst sets the dynamic inst to execute
func (wf *Wavefront) SetDynamicInst(i *Inst) {
	wf.inst = i
}

// EnableMemoryDependencyTracking enables recent vector-load provenance.
func (wf *Wavefront) EnableMemoryDependencyTracking() {
	if wf.MemoryDependency == nil {
		wf.MemoryDependency = new(MemoryDependencyState)
	}
}

// TrackIssuedInstruction propagates recent vector-load provenance through
// VGPR-producing instructions. The timestamp is an instruction sequence,
// allowing the timing model to recognize short load-to-address dependency
// chains without tagging a benchmark or opcode sequence.
func (wf *Wavefront) TrackIssuedInstruction(inst *insts.Inst) {
	state := wf.MemoryDependency
	if state == nil {
		return
	}
	// A dynamic instruction reaching the issue point re-arms dependency
	// detection, including EXEC=0 instructions that perform no lane work.
	state.DependencyStallActive = false
	state.DependencyStallRemaining = 0
	if wf.EXEC() == 0 {
		return
	}
	state.InstructionIssueSequence++

	if inst.ExeUnit == insts.ExeUnitVMem {
		if isVectorMemoryLoad(inst) && wf.EXEC() == ^uint64(0) {
			wf.setMemoryDerivedStamp(inst.Dst, 0)
		}
		return
	}

	stamp := uint64(0)
	for _, operand := range []*insts.Operand{
		inst.Src0, inst.Src1, inst.Src2,
		inst.Addr, inst.Data, inst.Base, inst.Offset,
	} {
		if candidate := wf.memoryDerivedStamp(operand); candidate > stamp {
			stamp = candidate
		}
	}
	if wf.EXEC() != ^uint64(0) {
		if previous := wf.memoryDerivedStamp(inst.Dst); previous > stamp {
			stamp = previous
		}
	}
	wf.setMemoryDerivedStamp(inst.Dst, stamp)
}

// StallRecentLoadAddress delays a short load-to-address dependency on this
// wave only. Returning true means the instruction must not issue this cycle;
// other ready waves remain eligible.
func (wf *Wavefront) StallRecentLoadAddress(
	inst *insts.Inst,
	penalty, maxAge int,
) bool {
	if penalty <= 0 || wf.EXEC() == 0 || !isVectorMemoryLoad(inst) {
		return false
	}
	state := wf.MemoryDependency
	if state == nil {
		return false
	}

	if !state.DependencyStallActive || state.DependencyStallPC != wf.PC() {
		state.DependencyStallActive = true
		state.DependencyStallPC = wf.PC()
		state.DependencyStallRemaining = 0
		if wf.AddressDependsOnRecentLoad(inst, uint64(maxAge)) {
			state.DependencyStallRemaining = penalty
		}
	}
	if state.DependencyStallRemaining <= 0 {
		return false
	}
	state.DependencyStallRemaining--
	return true
}

// MarkMemoryLoadDestination marks the destination of a completed vector load.
func (wf *Wavefront) MarkMemoryLoadDestination(inst *insts.Inst) {
	state := wf.MemoryDependency
	if state == nil {
		return
	}
	stamp := state.InstructionIssueSequence
	if stamp == 0 {
		stamp = 1
	}
	if wf.EXEC() != ^uint64(0) {
		if previous := wf.memoryDerivedStamp(inst.Dst); previous > stamp {
			stamp = previous
		}
	}
	wf.setMemoryDerivedStamp(inst.Dst, stamp)
}

// AddressDependsOnRecentLoad reports whether a vector-memory address reads a
// VGPR derived from a recently completed vector load.
func (wf *Wavefront) AddressDependsOnRecentLoad(
	inst *insts.Inst,
	maxAge uint64,
) bool {
	state := wf.MemoryDependency
	if state == nil {
		return false
	}
	stamp := uint64(0)
	for _, operand := range []*insts.Operand{inst.Addr, inst.Base, inst.Offset} {
		if candidate := wf.memoryDerivedStamp(operand); candidate > stamp {
			stamp = candidate
		}
	}
	if stamp == 0 || state.InstructionIssueSequence < stamp {
		return false
	}
	return state.InstructionIssueSequence-stamp <= maxAge
}

func (wf *Wavefront) memoryDerivedStamp(operand *insts.Operand) uint64 {
	state := wf.MemoryDependency
	if state == nil {
		return 0
	}
	if operand == nil || operand.OperandType != insts.RegOperand ||
		operand.Register == nil || !operand.Register.IsVReg() {
		return 0
	}
	base := operand.Register.RegIndex()
	count := operand.RegCount
	if count < 1 {
		count = 1
	}
	stamp := uint64(0)
	for i := 0; i < count && base+i < len(state.MemoryDerivedVGPR); i++ {
		if state.MemoryDerivedVGPR[base+i] > stamp {
			stamp = state.MemoryDerivedVGPR[base+i]
		}
	}
	return stamp
}

func (wf *Wavefront) setMemoryDerivedStamp(
	operand *insts.Operand,
	stamp uint64,
) {
	state := wf.MemoryDependency
	if state == nil {
		return
	}
	if operand == nil || operand.OperandType != insts.RegOperand ||
		operand.Register == nil || !operand.Register.IsVReg() {
		return
	}
	base := operand.Register.RegIndex()
	count := operand.RegCount
	if count < 1 {
		count = 1
	}
	for i := 0; i < count && base+i < len(state.MemoryDerivedVGPR); i++ {
		state.MemoryDerivedVGPR[base+i] = stamp
	}
}

func isVectorMemoryLoad(inst *insts.Inst) bool {
	switch inst.FormatType {
	case insts.FLAT:
		return inst.Opcode >= 16 && inst.Opcode <= 23
	case insts.MUBUF:
		return inst.Opcode >= 20 && inst.Opcode <= 23
	default:
		return false
	}
}

// ManagedInst returns the wrapped Inst
func (wf *Wavefront) ManagedInst() *Inst {
	return wf.inst
}

// PID returns pid
func (wf *Wavefront) PID() vm.PID {
	return wf.pid
}

// SetPID sets pid
func (wf *Wavefront) SetPID(pid vm.PID) {
	wf.pid = pid
}

// PC returns the program counter
func (wf *Wavefront) PC() uint64 {
	return wf.pc
}

// SetPC sets the program counter
func (wf *Wavefront) SetPC(v uint64) {
	wf.pc = v
}

// EXEC returns the exec mask
func (wf *Wavefront) EXEC() uint64 {
	return wf.exec
}

// SetEXEC sets the exec mask
func (wf *Wavefront) SetEXEC(v uint64) {
	wf.exec = v
}

// WideWriteStridePenalty tracks store locality per wavefront, preventing
// unrelated waves from perturbing one another's timing.
func (wf *Wavefront) WideWriteStridePenalty(
	current, lineBytes uint64,
	penalty int,
) int {
	result := 0
	if penalty > 0 && wf.hasWideWriteLine &&
		!wideWriteLinesAreLocal(wf.lastWideWriteLine, current, lineBytes) {
		result = penalty
	}
	wf.lastWideWriteLine = current
	wf.hasWideWriteLine = true
	return result
}

// ResetWideWriteTracking ends the current wide-store stream.
func (wf *Wavefront) ResetWideWriteTracking() {
	wf.lastWideWriteLine = 0
	wf.hasWideWriteLine = false
}

func wideWriteLinesAreLocal(previous, current, lineBytes uint64) bool {
	if lineBytes == 0 {
		return true
	}
	previous -= previous % lineBytes
	current -= current % lineBytes
	if previous == current {
		return true
	}
	return (current > previous && current-previous == lineBytes) ||
		(previous > current && previous-current == lineBytes)
}

// VCC returns the vector condition code
func (wf *Wavefront) VCC() uint64 {
	return wf.vcc
}

// SetVCC sets the vector condition code
func (wf *Wavefront) SetVCC(v uint64) {
	wf.vcc = v
}

// SCC returns the scalar condition code
func (wf *Wavefront) SCC() byte {
	return wf.scc
}

// SetSCC sets the scalar condition code
func (wf *Wavefront) SetSCC(v byte) {
	wf.scc = v
}

// ReadOperand reads the value of an operand using the RegFileAccessor
func (wf *Wavefront) ReadOperand(operand *insts.Operand, laneID int) uint64 {
	switch operand.OperandType {
	case insts.RegOperand:
		waveOffset := wf.SRegOffset
		if operand.Register.IsVReg() {
			waveOffset = wf.VRegOffset
		}
		buf := wf.RegAccessor.ReadReg(operand.Register, operand.RegCount, laneID, waveOffset)
		if len(buf) < 8 {
			padded := make([]byte, 8)
			copy(padded, buf)
			buf = padded
		}
		return insts.BytesToUint64(buf)
	case insts.IntOperand:
		return uint64(operand.IntValue)
	case insts.FloatOperand:
		return uint64(math.Float32bits(float32(operand.FloatValue)))
	case insts.LiteralConstant:
		return uint64(operand.LiteralConstant)
	default:
		log.Panicf("Unsupported operand type: %s", operand.String())
		return 0
	}
}

// WriteOperand writes a value to an operand using the RegFileAccessor
func (wf *Wavefront) WriteOperand(operand *insts.Operand, laneID int, value uint64) {
	if operand.OperandType != insts.RegOperand {
		log.Panicf("Cannot write to non-register operand: %s", operand.String())
	}

	numBytes := operand.Register.ByteSize
	if operand.RegCount >= 2 {
		numBytes *= operand.RegCount
	}

	waveOffset := wf.SRegOffset
	if operand.Register.IsVReg() {
		waveOffset = wf.VRegOffset
	}

	data := insts.Uint64ToBytes(value)
	wf.RegAccessor.WriteReg(operand.Register, operand.RegCount, laneID, waveOffset, data[:numBytes])
}

// ReadOperandBytes reads the raw bytes of an operand
func (wf *Wavefront) ReadOperandBytes(operand *insts.Operand, laneID int, byteCount int) []byte {
	switch operand.OperandType {
	case insts.RegOperand:
		waveOffset := wf.SRegOffset
		if operand.Register.IsVReg() {
			waveOffset = wf.VRegOffset
		}
		buf := wf.RegAccessor.ReadReg(operand.Register, operand.RegCount, laneID, waveOffset)
		if len(buf) > byteCount {
			return buf[:byteCount]
		}
		return buf
	case insts.IntOperand:
		data := insts.Uint64ToBytes(uint64(operand.IntValue))
		return data[:byteCount]
	case insts.FloatOperand:
		data := insts.Uint64ToBytes(uint64(math.Float32bits(float32(operand.FloatValue))))
		return data[:byteCount]
	case insts.LiteralConstant:
		data := insts.Uint64ToBytes(uint64(operand.LiteralConstant))
		return data[:byteCount]
	default:
		log.Panicf("Unsupported operand type: %s", operand.String())
		return nil
	}
}

// WriteOperandBytes writes raw bytes to an operand
func (wf *Wavefront) WriteOperandBytes(operand *insts.Operand, laneID int, data []byte) {
	if operand.OperandType != insts.RegOperand {
		log.Panicf("Cannot write to non-register operand: %s", operand.String())
	}

	waveOffset := wf.SRegOffset
	if operand.Register.IsVReg() {
		waveOffset = wf.VRegOffset
	}

	wf.RegAccessor.WriteReg(operand.Register, operand.RegCount, laneID, waveOffset, data)
}
