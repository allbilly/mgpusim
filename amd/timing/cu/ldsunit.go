package cu

import (
	"github.com/sarchlab/mgpusim/v5/amd/emu"
	"github.com/sarchlab/mgpusim/v5/amd/timing/wavefront"
)

// A LDSUnit performs Scalar operations
type LDSUnit struct {
	cu *ComputeUnit

	alu emu.ALU

	toRead            *wavefront.Wavefront
	inFlight          []ldsPipelineEntry
	issueIntervalLeft int

	isIdle bool
}

type ldsPipelineEntry struct {
	wave       *wavefront.Wavefront
	cyclesLeft int
}

// NewLDSUnit creates a new Scalar unit, injecting the dependency of
// the compute unit.
func NewLDSUnit(
	cu *ComputeUnit,
	alu emu.ALU,
) *LDSUnit {
	u := new(LDSUnit)
	u.cu = cu
	u.alu = alu
	return u
}

// CanAcceptWave checks if the buffer of the read stage is occupied or not
func (u *LDSUnit) CanAcceptWave() bool {
	return u.toRead == nil
}

// IsIdle checks idleness
func (u *LDSUnit) IsIdle() bool {
	u.isIdle = u.toRead == nil && len(u.inFlight) == 0
	return u.isIdle
}

// AcceptWave moves one wavefront into the read buffer of the Scalar unit
func (u *LDSUnit) AcceptWave(wave *wavefront.Wavefront) {
	u.toRead = wave
}

// Run executes three pipeline stages that are controlled by the LDSUnit
func (u *LDSUnit) Run() bool {
	madeProgress := false
	madeProgress = u.advancePipeline() || madeProgress
	madeProgress = u.issue() || madeProgress
	return madeProgress
}

func (u *LDSUnit) advancePipeline() bool {
	madeProgress := false
	if u.issueIntervalLeft > 0 {
		u.issueIntervalLeft--
		madeProgress = true
	}

	remaining := u.inFlight[:0]
	for _, entry := range u.inFlight {
		entry.cyclesLeft--
		madeProgress = true
		if entry.cyclesLeft > 0 {
			remaining = append(remaining, entry)
			continue
		}

		u.cu.logInstTask(entry.wave, entry.wave.DynamicInst(), true)
		u.cu.UpdatePCAndSetReady(entry.wave)
	}
	u.inFlight = remaining
	return madeProgress
}

func (u *LDSUnit) issue() bool {
	if u.toRead == nil || u.issueIntervalLeft > 0 {
		return false
	}

	spec := u.cu.comp.Spec()
	maxInFlight := spec.LDSMaxInFlight
	if maxInFlight <= 0 {
		maxInFlight = 1
	}
	if len(u.inFlight) >= maxInFlight {
		return false
	}

	wave := u.toRead
	conflictCycles := LDSBankConflictCycles(
		wave.Inst(),
		wave.EXEC(),
		func(lane int) uint32 {
			return uint32(wave.ReadOperand(wave.Inst().Addr, lane))
		},
		spec.LDSBankCount,
		spec.LDSBankWidth,
		spec.LDSBankConflictPenalty,
	)
	b128ExtraCycles := LDSB128ServiceExtraCycles(
		wave.Inst(), spec.LDSB128ServiceExtraCycles)
	u.alu.SetLDS(wave.WG.LDS)
	u.alu.Run(wave)

	resultLatency := spec.LDSPipelineLatency + conflictCycles +
		b128ExtraCycles
	if resultLatency < 1 {
		resultLatency = 1
	}
	u.inFlight = append(u.inFlight, ldsPipelineEntry{
		wave:       wave,
		cyclesLeft: resultLatency,
	})

	issueInterval := spec.LDSIssueInterval
	if issueInterval <= 0 {
		issueInterval = spec.LDSPipelineLatency
	}
	u.issueIntervalLeft = issueInterval + conflictCycles + b128ExtraCycles
	u.toRead = nil
	return true
}

// Flush clears the unit
func (u *LDSUnit) Flush() {
	u.toRead = nil
	u.inFlight = nil
	u.issueIntervalLeft = 0
}
