package main

import (
	"flag"

	"github.com/sarchlab/mgpusim/v5/amd/benchmarks/microbench/scratchspill"
	"github.com/sarchlab/mgpusim/v5/amd/samples/runner"
)

var variant = flag.String("variant", scratchspill.VariantScratch4,
	"Probe variant: control4 or scratch4.")
var workgroups = flag.Int("workgroups", 16,
	"Number of independent 8x8 one-wave work-groups.")

func main() {
	flag.Parse()
	runner := new(runner.Runner).Init()
	benchmark := scratchspill.NewBenchmark(runner.Driver())
	benchmark.Arch = runner.ArchType
	benchmark.Variant = *variant
	benchmark.Workgroups = *workgroups
	runner.AddBenchmark(benchmark)
	runner.Run()
}
