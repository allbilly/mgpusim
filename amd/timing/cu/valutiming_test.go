package cu

import (
	"testing"

	"github.com/sarchlab/mgpusim/v5/amd/insts"
)

func TestGetVALUTimingClassOverrides(t *testing.T) {
	spec := VALUTiming{
		DefaultIssueInterval:         4,
		DefaultResultLatency:         4,
		FMAIssueInterval:             4,
		FMAResultLatency:             5,
		IntegerMultiplyIssueInterval: 4,
		IntegerMultiplyResultLatency: 6,
		TranscendentalIssueInterval:  4,
		TranscendentalResultLatency:  16,
		FP64IssueInterval:            8,
		FP64ResultLatency:            8,
	}

	tests := []struct {
		name       string
		issue, lat int
	}{
		{"v_add_f32", 4, 4},
		{"v_fma_f32", 4, 5},
		{"v_mul_lo_u32", 4, 6},
		{"v_sqrt_f32", 4, 16},
		{"v_add_f64", 8, 8},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			inst := insts.NewInst()
			inst.InstName = test.name
			issue, latency := GetVALUTiming(inst, 16, spec)
			if issue != test.issue || latency != test.lat {
				t.Fatalf("got (%d,%d), want (%d,%d)",
					issue, latency, test.issue, test.lat)
			}
		})
	}
}

func TestGetVALUTimingLegacyFallback(t *testing.T) {
	inst := insts.NewInst()
	inst.InstName = "v_add_f32"
	issue, latency := GetVALUTiming(inst, 16, VALUTiming{})
	if issue != 4 || latency != 4 {
		t.Fatalf("got (%d,%d), want legacy (4,4)", issue, latency)
	}
}
