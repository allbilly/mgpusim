package main

import (
	"flag"

	"github.com/sarchlab/mgpusim/v5/amd/benchmarks/microbench/fp32throughput"
	"github.com/sarchlab/mgpusim/v5/amd/samples/runner"
)

var numBlocks = flag.Int("num-blocks", 4, "The number of thread blocks to launch.")
var fmasPerThread = flag.Int("fmas", 256,
	"The number of FMA iterations per thread (rounded down to a multiple of 4).")
var threadsPerBlock = flag.Int("threads-per-block", 256,
	"The work-group (block) size; must be <= 1024.")
var dependent = flag.Bool("dependent", false,
	"Use one accumulator to expose dependent FP32-FMA latency (gfx90c only).")
var operation = flag.String("operation", fp32throughput.OperationFMA,
	"Arithmetic operation for the four-chain probe: fma, mul, or add.")

func main() {
	flag.Parse()

	runner := new(runner.Runner).Init()

	benchmark := fp32throughput.NewBenchmark(runner.Driver())
	benchmark.NumBlocks = *numBlocks
	benchmark.FmasPerThread = *fmasPerThread
	benchmark.ThreadsPerBlock = *threadsPerBlock
	benchmark.Dependent = *dependent
	benchmark.Operation = *operation
	benchmark.Arch = runner.ArchType

	runner.AddBenchmark(benchmark)

	runner.Run()
}
