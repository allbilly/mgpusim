package cu

import (
	"testing"

	"github.com/sarchlab/mgpusim/v5/amd/insts"
)

func TestGetVALUTimingClassOverrides(t *testing.T) {
	spec := VALUTiming{
		DefaultIssueInterval:         4,
		DefaultResultLatency:         4,
		FP32AddIssueInterval:         7,
		FP32AddResultLatency:         8,
		FP32MultiplyIssueInterval:    9,
		FP32MultiplyResultLatency:    10,
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
		{"v_add_f32", 7, 8},
		{"v_mul_f32", 9, 10},
		{"v_fma_f32", 4, 5},
		{"v_mul_lo_u32", 4, 6},
		{"v_mad_u64_u32", 4, 6},
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

func TestClassifyVALUSeparatesFP32ArithmeticAndIntegerMAD(t *testing.T) {
	tests := []struct {
		name string
		want valuClass
	}{
		{"v_add_f32", valuFP32Add},
		{"v_add_f32_e32", valuFP32Add},
		{"v_mul_f32", valuFP32Multiply},
		{"v_mul_legacy_f32", valuFP32Multiply},
		{"v_fma_f32", valuFMA},
		{"v_fmac_f32_e32", valuFMA},
		{"v_fma_f16", valuFMA},
		{"v_mad_u64_u32", valuIntegerMultiply},
		{"v_pk_add_f32", valuDefault},
		{"v_pk_mul_f32", valuDefault},
		{"v_pk_fma_f32", valuDefault},
		{"ds_add_f32", valuDefault},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			inst := insts.NewInst()
			inst.InstName = test.name
			if got := classifyVALU(inst); got != test.want {
				t.Fatalf("got class %d, want %d", got, test.want)
			}
		})
	}
}

func TestSpecVALUTimingRoundTrip(t *testing.T) {
	want := VALUTiming{
		DefaultIssueInterval:         1,
		DefaultResultLatency:         2,
		FP32AddIssueInterval:         3,
		FP32AddResultLatency:         4,
		FP32MultiplyIssueInterval:    5,
		FP32MultiplyResultLatency:    6,
		FMAIssueInterval:             7,
		FMAResultLatency:             8,
		IntegerMultiplyIssueInterval: 9,
		IntegerMultiplyResultLatency: 10,
		TranscendentalIssueInterval:  11,
		TranscendentalResultLatency:  12,
		FP64IssueInterval:            13,
		FP64ResultLatency:            14,
		MaxInFlight:                  15,
	}

	var spec Spec
	spec.SetVALUTiming(want)
	if got := spec.VALUTimingSpec(); got != want {
		t.Fatalf("got %+v, want %+v", got, want)
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
