package main

import (
	"flag"

	"github.com/sarchlab/mgpusim/v5/amd/benchmarks/microbench/storestride"
	"github.com/sarchlab/mgpusim/v5/amd/samples/runner"
)

var strideLines = flag.Int("stride-lines", 1,
	"Distance in 64-byte cache lines between full-line stores.")
var allocationStrideLines = flag.Int("allocation-stride-lines", 128,
	"Fixed maximum stride used to keep allocation size invariant.")
var repeats = flag.Int("repeats", 32,
	"Dynamic wide-store instructions executed by each work-group.")
var workgroups = flag.Int("workgroups", 1,
	"Number of independent one-wave work-groups.")

func main() {
	flag.Parse()
	runner := new(runner.Runner).Init()
	benchmark := storestride.NewBenchmark(runner.Driver())
	benchmark.Arch = runner.ArchType
	benchmark.StrideLines = *strideLines
	benchmark.AllocationStrideLines = *allocationStrideLines
	benchmark.Repeats = *repeats
	benchmark.Workgroups = *workgroups
	runner.AddBenchmark(benchmark)
	runner.Run()
}
