package main

import (
	"flag"

	"github.com/sarchlab/mgpusim/v5/amd/benchmarks/microbench/vmemloadshape"
	"github.com/sarchlab/mgpusim/v5/amd/samples/runner"
)

var width = flag.Int("width-dwords", 4, "Load width in dwords: 1, 2, or 4.")
var mode = flag.String("mode", vmemloadshape.ModeSerial,
	"Dependency mode: serial or independent4.")
var aliasLanes = flag.Int("alias-lanes", 8,
	"Number of interleaved lanes sharing one address: 1, 2, 4, or 8.")
var arrayBytes = flag.Int("array-bytes", 8*1024,
	"Power-of-two vector footprint in bytes (8 KiB for L1, 64 KiB for L2).")
var repeats = flag.Int("repeats", 1024,
	"Loads per lane; independent4 requires a multiple of four.")
var workgroups = flag.Int("workgroups", 16, "Number of 64-lane work-groups.")

func main() {
	flag.Parse()
	runner := new(runner.Runner).Init()
	benchmark := vmemloadshape.NewBenchmark(runner.Driver())
	benchmark.Arch = runner.ArchType
	benchmark.WidthDwords = *width
	benchmark.Mode = *mode
	benchmark.AliasLanes = *aliasLanes
	benchmark.ArrayBytes = *arrayBytes
	benchmark.Repeats = *repeats
	benchmark.Workgroups = *workgroups
	runner.AddBenchmark(benchmark)
	runner.Run()
}
