package cu

import (
	"strings"

	"github.com/sarchlab/mgpusim/v5/amd/insts"
)

// Scoreboard tracks per-wavefront register availability for hazard detection.
// Each entry stores the number of cycles until the register becomes available.
type Scoreboard struct {
	VGPRBusyUntil [256]int // cycle counter for each VGPR
	SGPRBusyUntil [102]int // cycle counter for each SGPR
	SCCBusyUntil  int
	VCCBusyUntil  int
	EXECBusyUntil int

	// Runtime-only sparse indexes avoid scanning all 358 registers for every
	// resident wavefront on every cycle. Direct array mutation remains
	// supported for tests and checkpoint restoration through managed=false.
	activeVGPR  []int
	activeSGPR  []int
	vgprTracked [256]bool
	sgprTracked [102]bool
	managed     bool
}

// NewScoreboard creates a new Scoreboard with all counters at zero.
func NewScoreboard() *Scoreboard {
	return &Scoreboard{}
}

// Tick decrements all non-zero counters by 1 each cycle.
func (s *Scoreboard) Tick() {
	if s.managed {
		s.tickActiveRegisters()
		s.tickSpecialRegisters()
		return
	}

	for i := range s.VGPRBusyUntil {
		if s.VGPRBusyUntil[i] > 0 {
			s.VGPRBusyUntil[i]--
		}
	}

	for i := range s.SGPRBusyUntil {
		if s.SGPRBusyUntil[i] > 0 {
			s.SGPRBusyUntil[i]--
		}
	}

	s.tickSpecialRegisters()
}

func (s *Scoreboard) tickActiveRegisters() {
	activeVGPR := s.activeVGPR[:0]
	for _, reg := range s.activeVGPR {
		s.VGPRBusyUntil[reg]--
		if s.VGPRBusyUntil[reg] > 0 {
			activeVGPR = append(activeVGPR, reg)
		} else {
			s.vgprTracked[reg] = false
		}
	}
	s.activeVGPR = activeVGPR

	activeSGPR := s.activeSGPR[:0]
	for _, reg := range s.activeSGPR {
		s.SGPRBusyUntil[reg]--
		if s.SGPRBusyUntil[reg] > 0 {
			activeSGPR = append(activeSGPR, reg)
		} else {
			s.sgprTracked[reg] = false
		}
	}
	s.activeSGPR = activeSGPR
}

func (s *Scoreboard) tickSpecialRegisters() {
	if s.SCCBusyUntil > 0 {
		s.SCCBusyUntil--
	}
	if s.VCCBusyUntil > 0 {
		s.VCCBusyUntil--
	}
	if s.EXECBusyUntil > 0 {
		s.EXECBusyUntil--
	}
}

// MarkBusy marks destination registers of the instruction as busy for the
// given number of cycles. It examines Dst (VGPR/SGPR), SDst, and implicit
// SCC/VCC writes.
func (s *Scoreboard) MarkBusy(inst *insts.Inst, latency int) {
	if latency <= 0 {
		return
	}
	s.enableSparseTracking()

	s.markOperandBusy(inst.Dst, latency)
	s.markOperandBusy(inst.SDst, latency)

	// Scalar ALU instructions implicitly write SCC.
	if inst.ExeUnit == insts.ExeUnitScalar {
		s.SCCBusyUntil = max(s.SCCBusyUntil, latency)
	}

	// VOPC comparison instructions implicitly write VCC.
	if inst.Format != nil && inst.FormatType == insts.VOPC {
		s.VCCBusyUntil = max(s.VCCBusyUntil, latency)
	}
}

// enableSparseTracking builds the runtime-only indexes once. This also
// preserves counters restored from a checkpoint before the first new issue.
func (s *Scoreboard) enableSparseTracking() {
	if s.managed {
		return
	}

	for reg, busyUntil := range s.VGPRBusyUntil {
		if busyUntil > 0 {
			s.vgprTracked[reg] = true
			s.activeVGPR = append(s.activeVGPR, reg)
		}
	}
	for reg, busyUntil := range s.SGPRBusyUntil {
		if busyUntil > 0 {
			s.sgprTracked[reg] = true
			s.activeSGPR = append(s.activeSGPR, reg)
		}
	}
	s.managed = true
}

func (s *Scoreboard) markOperandBusy(op *insts.Operand, latency int) {
	if op == nil || op.OperandType != insts.RegOperand || op.Register == nil {
		return
	}

	reg := op.Register
	regCount := op.RegCount
	if regCount < 1 {
		regCount = 1
	}

	if reg.IsVReg() {
		base := reg.RegIndex()
		for i := 0; i < regCount && base+i < 256; i++ {
			index := base + i
			s.VGPRBusyUntil[index] = max(s.VGPRBusyUntil[index], latency)
			if !s.vgprTracked[index] {
				s.vgprTracked[index] = true
				s.activeVGPR = append(s.activeVGPR, index)
			}
		}
		return
	}

	if reg.IsSReg() {
		base := reg.RegIndex()
		for i := 0; i < regCount && base+i < 102; i++ {
			index := base + i
			s.SGPRBusyUntil[index] = max(s.SGPRBusyUntil[index], latency)
			if !s.sgprTracked[index] {
				s.sgprTracked[index] = true
				s.activeSGPR = append(s.activeSGPR, index)
			}
		}
		return
	}

	switch reg.RegType {
	case insts.SCC:
		s.SCCBusyUntil = max(s.SCCBusyUntil, latency)
	case insts.VCC, insts.VCCLO, insts.VCCHI:
		s.VCCBusyUntil = max(s.VCCBusyUntil, latency)
	case insts.EXEC, insts.EXECLO, insts.EXECHI:
		s.EXECBusyUntil = max(s.EXECBusyUntil, latency)
	}
}

// HasHazard checks if any source operand reads a register that is still busy.
func (s *Scoreboard) HasHazard(inst *insts.Inst) bool {
	operands := []*insts.Operand{
		inst.Src0, inst.Src1, inst.Src2,
		inst.Addr, inst.Data, inst.Base, inst.Offset,
	}

	for _, op := range operands {
		if s.operandHasHazard(op) {
			return true
		}
	}

	return false
}

func (s *Scoreboard) operandHasHazard(op *insts.Operand) bool {
	if op == nil || op.OperandType != insts.RegOperand || op.Register == nil {
		return false
	}

	reg := op.Register
	regCount := op.RegCount
	if regCount < 1 {
		regCount = 1
	}

	if reg.IsVReg() {
		base := reg.RegIndex()
		for i := 0; i < regCount && base+i < 256; i++ {
			if s.VGPRBusyUntil[base+i] > 0 {
				return true
			}
		}
		return false
	}

	if reg.IsSReg() {
		base := reg.RegIndex()
		for i := 0; i < regCount && base+i < 102; i++ {
			if s.SGPRBusyUntil[base+i] > 0 {
				return true
			}
		}
		return false
	}

	switch reg.RegType {
	case insts.SCC:
		return s.SCCBusyUntil > 0
	case insts.VCC, insts.VCCLO, insts.VCCHI:
		return s.VCCBusyUntil > 0
	case insts.EXEC, insts.EXECLO, insts.EXECHI:
		return s.EXECBusyUntil > 0
	}

	return false
}

// AnyBusy returns true if any register counter is still > 0.
func (s *Scoreboard) AnyBusy() bool {
	if s.managed {
		return len(s.activeVGPR) > 0 ||
			len(s.activeSGPR) > 0 ||
			s.SCCBusyUntil > 0 ||
			s.VCCBusyUntil > 0 ||
			s.EXECBusyUntil > 0
	}

	for _, v := range s.VGPRBusyUntil {
		if v > 0 {
			return true
		}
	}
	for _, v := range s.SGPRBusyUntil {
		if v > 0 {
			return true
		}
	}
	return s.SCCBusyUntil > 0 || s.VCCBusyUntil > 0 || s.EXECBusyUntil > 0
}

// Clear resets all counters to 0.
func (s *Scoreboard) Clear() {
	s.VGPRBusyUntil = [256]int{}
	s.SGPRBusyUntil = [102]int{}
	s.SCCBusyUntil = 0
	s.VCCBusyUntil = 0
	s.EXECBusyUntil = 0
	s.activeVGPR = nil
	s.activeSGPR = nil
	s.vgprTracked = [256]bool{}
	s.sgprTracked = [102]bool{}
	s.managed = false
}

// GetScoreboardLatency returns the scoreboard latency for an instruction based
// on its execution unit. LDS/VMem instructions return 0 (not tracked by
// scoreboard; handled by s_waitcnt).
func GetScoreboardLatency(inst *insts.Inst) int {
	switch inst.ExeUnit {
	case insts.ExeUnitVALU:
		if strings.Contains(inst.InstName, "f64") {
			return 8
		}
		return 4
	case insts.ExeUnitScalar:
		return 2
	case insts.ExeUnitBranch:
		return 3
	default:
		return 0 // LDS/VMem/Special - don't track
	}
}
