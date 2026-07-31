package cu

import (
	"strings"

	"github.com/sarchlab/mgpusim/v5/amd/insts"
)

type valuClass int

const (
	valuDefault valuClass = iota
	valuBitwise
	valuFMA
	valuIntegerMultiply
	valuTranscendental
	valuFP64
)

func classifyVALU(inst *insts.Inst) valuClass {
	name := strings.ToLower(inst.InstName)

	if strings.Contains(name, "f64") {
		return valuFP64
	}

	for _, token := range []string{
		"rcp", "rsq", "sqrt", "log", "exp", "sin", "cos",
	} {
		if strings.Contains(name, token) {
			return valuTranscendental
		}
	}

	for _, token := range []string{"_xor_", "_and_", "_or_"} {
		if strings.Contains(name, token) {
			return valuBitwise
		}
	}

	for _, token := range []string{"fma", "mad", "mac"} {
		if strings.Contains(name, token) {
			return valuFMA
		}
	}

	if strings.Contains(name, "mul") &&
		!strings.Contains(name, "f16") &&
		!strings.Contains(name, "f32") {
		return valuIntegerMultiply
	}

	return valuDefault
}

// GetVALUTiming returns wave-level issue interval and result latency in cycles.
func GetVALUTiming(
	inst *insts.Inst,
	numSinglePrecisionUnits int,
	spec VALUTiming,
) (issueInterval, resultLatency int) {
	legacy := 64 / numSinglePrecisionUnits
	if strings.Contains(strings.ToLower(inst.InstName), "f64") {
		legacy = 64 / (numSinglePrecisionUnits / 2)
	}

	issueInterval = spec.DefaultIssueInterval
	resultLatency = spec.DefaultResultLatency

	switch classifyVALU(inst) {
	case valuBitwise:
		issueInterval = firstPositive(
			spec.BitwiseIssueInterval, issueInterval)
		resultLatency = firstPositive(
			spec.BitwiseResultLatency, resultLatency)
	case valuFMA:
		issueInterval = firstPositive(spec.FMAIssueInterval, issueInterval)
		resultLatency = firstPositive(spec.FMAResultLatency, resultLatency)
	case valuIntegerMultiply:
		issueInterval = firstPositive(
			spec.IntegerMultiplyIssueInterval, issueInterval)
		resultLatency = firstPositive(
			spec.IntegerMultiplyResultLatency, resultLatency)
	case valuTranscendental:
		issueInterval = firstPositive(
			spec.TranscendentalIssueInterval, issueInterval)
		resultLatency = firstPositive(
			spec.TranscendentalResultLatency, resultLatency)
	case valuFP64:
		issueInterval = firstPositive(spec.FP64IssueInterval, issueInterval)
		resultLatency = firstPositive(spec.FP64ResultLatency, resultLatency)
	}

	issueInterval = firstPositive(issueInterval, legacy)
	resultLatency = firstPositive(resultLatency, legacy)

	return issueInterval, resultLatency
}

func firstPositive(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}
