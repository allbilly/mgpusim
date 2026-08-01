package fp32throughput

import "testing"

func TestKernelSymbol(t *testing.T) {
	tests := []struct {
		benchmark Benchmark
		want      string
	}{
		{Benchmark{}, "fp32_fma_kernel"},
		{Benchmark{Operation: OperationFMA}, "fp32_fma_kernel"},
		{Benchmark{Dependent: true}, "fp32_fma_dependent_kernel"},
		{Benchmark{Operation: OperationMul}, "fp32_mul_kernel"},
		{Benchmark{Operation: OperationAdd}, "fp32_add_kernel"},
	}
	for _, test := range tests {
		if got := test.benchmark.symbol(); got != test.want {
			t.Errorf("symbol %q, want %q", got, test.want)
		}
	}
}

func TestFmaCountPreservesExistingRounding(t *testing.T) {
	for _, test := range []struct {
		input int
		want  int32
	}{
		{0, 256},
		{1, 4},
		{4, 4},
		{67, 64},
		{4096, 4096},
	} {
		b := &Benchmark{FmasPerThread: test.input}
		if got := b.fmasPerThread(); got != test.want {
			t.Errorf("fmasPerThread(%d) = %d, want %d", test.input, got, test.want)
		}
	}
}
