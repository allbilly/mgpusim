package runner

import "testing"

func TestCacheHitRateReportingEnabled(t *testing.T) {
	originalReportAll := *reportAll
	originalHitRate := *cacheHitRateReportFlag
	originalLatency := *cacheLatencyReportFlag
	t.Cleanup(func() {
		*reportAll = originalReportAll
		*cacheHitRateReportFlag = originalHitRate
		*cacheLatencyReportFlag = originalLatency
	})

	tests := []struct {
		name      string
		reportAll bool
		hitRate   bool
		latency   bool
		want      bool
	}{
		{name: "disabled", want: false},
		{name: "hit-rate flag", hitRate: true, want: true},
		{name: "report all", reportAll: true, want: true},
		{
			name:    "latency alone does not enable hit rate",
			latency: true,
			want:    false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			*reportAll = test.reportAll
			*cacheHitRateReportFlag = test.hitRate
			*cacheLatencyReportFlag = test.latency

			if got := cacheHitRateReportingEnabled(); got != test.want {
				t.Fatalf("cacheHitRateReportingEnabled() = %t, want %t", got, test.want)
			}
		})
	}
}
