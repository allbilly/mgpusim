package cu

import (
	"strings"

	"github.com/sarchlab/mgpusim/v5/amd/insts"
)

type valuClass int

const (
	valuDefault valuClass = iota
	valuFP32Add
	valuFP32Multiply
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

	if isScalarFMA(name) {
		return valuFMA
	}

	if strings.HasPrefix(name, "v_add_f32") {
		return valuFP32Add
	}

	if strings.HasPrefix(name, "v_mul_f32") ||
		strings.HasPrefix(name, "v_mul_legacy_f32") {
		return valuFP32Multiply
	}

	if containsAny(name, "mul", "mad", "mac") &&
		!strings.Contains(name, "f16") &&
		!strings.Contains(name, "f32") {
		return valuIntegerMultiply
	}

	return valuDefault
}

func isScalarFMA(name string) bool {
	for _, prefix := range []string{
		"v_fma_f16", "v_mac_f16", "v_mad_f16", "v_madak_f16",
		"v_madmk_f16",
		"v_fma_f32", "v_fmac_f32", "v_mac_f32", "v_mad_f32",
		"v_mad_legacy_f32", "v_madak_f32", "v_madmk_f32",
	} {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}

	return false
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
	case valuFP32Add:
		issueInterval = firstPositive(
			spec.FP32AddIssueInterval, issueInterval)
		resultLatency = firstPositive(
			spec.FP32AddResultLatency, resultLatency)
	case valuFP32Multiply:
		issueInterval = firstPositive(
			spec.FP32MultiplyIssueInterval, issueInterval)
		resultLatency = firstPositive(
			spec.FP32MultiplyResultLatency, resultLatency)
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

func containsAny(value string, tokens ...string) bool {
	for _, token := range tokens {
		if strings.Contains(value, token) {
			return true
		}
	}

	return false
}

func firstPositive(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}
