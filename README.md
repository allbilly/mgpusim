# MGPUSIM



![GitHub Discussions](https://img.shields.io/github/discussions/sarchlab/mgpusim)


[![MGPUSim Test](https://github.com/sarchlab/mgpusim/actions/workflows/mgpusim_test.yml/badge.svg)](https://github.com/sarchlab/mgpusim/actions/workflows/mgpusim_test.yml)

[![Go Reference](https://pkg.go.dev/badge/github.com/sarchlab/mgpusim.svg)](https://pkg.go.dev/github.com/sarchlab/mgpusim)
[![Go Report Card](https://goreportcard.com/badge/github.com/sarchlab/mgpusim/v4)](https://goreportcard.com/report/github.com/sarchlab/mgpusim/v4)

MGPUSim Documents can be found [here](https://akitasim.dev/docs/mgpusim/intro). Please raise issues if you need documentation on a specific aspect. 


MGPUSim is a high-flexibility, high-performance, high-accuracy GPU simulator. It models GPUs that run the AMD GCN3 instruction sets. One main feature of MGPUSim is the support for multi-GPU simulation (you can still use it for single-GPU architecture research).

## <span style="color:red">⚠️ Important Note on NVIDIA Simulation</span>

<span style="color:red">**Warning**: NVIDIA GPU simulation is under ongoing development and is not ready for use. Currently, only AMD GCN3-based GPU simulation is stable and supported.</span>

## Getting Started

- Install the most recent version of Go from golang.org.
- Clone this repository, assuming the path is `[mgpusim_home]`.
- Change your current directory to `[mgpusim_home]/samples/fir`.
- Compile the simulator with the benchmark with `go build`. The compiler will generate an executable file called `fir` (on Linux or Mac OS) or `fir.exe` (on Windows) for you.
- Run the simulation with `./fir -timing --report-all` to run the simulation.
- Check the generated `metrics.csv` file for high-level metrics output.

## Develop with Modified Version of Akita (or other depending libraries)

If a modification to Akita is required, you can clone Akita next to the MGPUSim directory in your system. Then, you can modify the `go.mod` file to include the following line. 

```
replace github.com/sarchlab/akita/v4 => ../akita
```

This line will direct the go compiler to use your local version of Akita rather than the official release of Akita. 

## Benchmark Support

| AMD APP SDK           | DNN Mark   | HeteroMark | Polybench | Rodinia          | SHOC      |
| --------------------- | ---------- | ---------- | --------- | ---------------- | --------- |
| Bitonic Sort          | MaxPooling | AES        | ATAX      | Needleman-Wunsch | BFS       |
| Fast Walsh Transform  | ReLU       | FIR        | BICG      |                  | FFT       |
| Floyd-Warshall        |            | KMeans     |           |                  | SPMV      |
| Matrix Multiplication |            | PageRank   |           |                  | Stencil2D |
| Matrix Transpose      |            |            |           |                  |           |
| NBody                 |            |            |           |                  |           |
| Simple Covolution     |            |            |           |                  |           |

## ISCA 2019 Application Suite: Sim vs Hardware (gfx90c)

Calibration against a **Renoir gfx90c APU** (Ryzen 7 4700U: 7 active CUs,
1600 MHz) using the ten-application suite from the
[ISCA 2019 MGPUSim paper](https://doi.org/10.1145/3307650.3322230). Against the
accepted legacy pinned table, all nine scored benchmarks pass the strict
`abs(HW/Sim - 1) < 10%` gate; scored MARE is **7.33%**. Corrected matrix
multiplication remains unscored because its target predates the source/HSACO
correction.

### Configuration

| Setting | Simulator | Hardware |
|---------|-----------|----------|
| Architecture | `-arch gcn5 -gpu gfx90c` | gfx90c (Renoir iGPU, 7 CU @ 1600 MHz) |
| Mode | `-timing -disable-rtm -verify` | legacy native HIP kernels |
| ROCm | — | `docker.io/rocm/dev-ubuntu-24.04:7.1.1` |
| Timing | Driver `kernel_time`, verification enabled | `hipEvent` kernel timing |
| Power | — | pinned `power_dpm_force_performance_level=high` |

### Problem sizes

| Benchmark | Flags |
|-----------|-------|
| vectoradd | `-width 65536 -height 1` |
| relu | `-length 65536` |
| matrixmult | `-x 128 -y 128 -z 128` |
| matrixtranspose | `-width 512` |
| bitonicsort | `-length 4096` |
| aes | `-length 4096` |
| fir | `-length 8192 -taps 16` |
| kmeans | `-points 4096 -features 16 -clusters 5 -max-iter 1` |
| pagerank | `-node 512 -sparsity 0.5 -iterations 2` |
| nw | `-length 128` |

### Results (kernel time, µs)

| Benchmark | Sim (µs) | HW (µs) | Error | Status |
|-----------|---------:|--------:|------:|--------|
| vectoradd | 27.261 | 29.364 | +7.71% | Pass |
| relu | 14.572 | 13.123 | -9.94% | Pass |
| matrixmult | 42.229 | 39.107 | -7.39% | Not scored (stale HW target) |
| matrixtranspose | 129.067 | 140.773 | +9.07% | Pass |
| bitonicsort | 745.124 | 811.360 | +8.89% | Pass |
| aes | 15.476 | 16.962 | +9.60% | Pass |
| fir | 11.316 | 12.003 | +6.07% | Pass |
| kmeans | 40.439 | 39.220 | -3.01% | Pass |
| pagerank | 126.412 | 130.638 | +3.34% | Pass |
| nw | 134.211 | 123.052 | -8.31% | Pass |

**Sim metric:** total `kernel_time` reported by the driver (all kernel launches in the benchmark).

**HW metric:** average GPU kernel time per iteration via `hipEvent`, using HIP kernels compiled from MGPUSim `native/*.cpp` sources (see notes below).

### Reproduce

```bash
# Simulator (all 10 benchmarks)
bash gpu_perf_scripts/calibration/gfx90c/run_sim.sh

# Hardware (requires /dev/kfd and /dev/dri; ROCm via podman; no sudo)
bash gpu_perf_scripts/calibration/gfx90c/build_and_run.sh

# Compare
python3 gpu_perf_scripts/calibration/gfx90c/compare.py
```

Harness: `gpu_perf_scripts/calibration/gfx90c/` (`isca10_bench.cpp`, `build_and_run.sh`, `run_sim.sh`).

### Methodology notes

All ten simulator runs use the current exact gfx90c HSACOs, deterministic
fixtures, matching launch geometry, and output verification. The legacy
hardware table was collected earlier with warmed, many-iteration native HIP
kernels, so the 7.33% result is a calibration score rather than a fully
matched validation. Every target should be reacquired with the pinned
exact-HSACO single-shot harness; matrix multiplication is additionally invalid
because its source changed. Signed error is `HW/Sim - 1`. ReLU, AES, and
transpose have narrow default-size margins, and unpinned size sweeps cannot
establish absolute cross-size accuracy.

The timing preset is in
`amd/samples/runner/timingconfig/gfx90c/builder.go`. Detailed size sweeps,
counter evidence, rejected candidates, and validation provenance are in
`gpu_perf_scripts/calibration/gfx90c/microbenchmark_results.md`. The reusable
paired-latency/throughput and geometric-sweep procedure is in
`how_to_microbenchmark_amd.md`.

## Default Performance Metrics Supported

You can run a simulation with the `--report-all` argument to enable all the performance metrics.

- Total execution time
- Total kernel time
- Per-GPU kernel time
- Instruction count on each Compute Unit
- Average request latency on all the cache components
- Number of read-misses, read-mshr-hits, read-hits, write-misses, write-mshr-hits, and write hits on all the cache components
- Number of incoming transactions and outgoing transactions on all the RDMA components.
- Number of transactions on each DRAM controller.

## How to Prepare Your Own Experiment

- Create a new repository repo. Typically we create one repo for each project, which may contain multiple experiments.
- Create a folder in your repo for each experiment. Run `go init [git repo path]/[directory_name]` to initialize the folder as a new go module. For example, if your git repository is hosted at `https://github.com/syifan/fancy_project` and your experiment folder is named as `exp1`, your module path should be `github.com/syifan/fancy_project/exp1`.
- Copy all the files under the directory `samples/experiment` to your experiment folder. In the `main.go` file, change the benchmark and the problem size to run. Or you can use an argument to select which benchmark to run. The file `runner.go`, `platform.go`, `r9nano.go`, and `shaderarray.go` serve as configuration files. So you need to change them according to your need.
- It is also possible to modify an existing component or adding a new component. You should copy the folder that includes the component you want to modify to your repo first. Then, modify the configuration scripts to link the system with your new component. You can try to add some print commands to see if your local component is used. Finally, you can start to modify the component code.

## Contributing

- If you find any bug related to the simulator (e.g., simulator is not accurately modeling some behavior or the simulator is not getting the correct emulation result), please raise an issue in the issue tab.
- If you want a new feature (e.g., you need to implement some new instructions or you want to model some new components), please also raise an issue.
- If you want to add a feature or fix a bug, create a pull request.
- There is no particular style requirement other than the default Go style requirement. Please run `gofmt`, `goimports`, or `goreturns` before making your merge request ready. Also, running `golangci-lint run` in the root directory will point you out most of the styling errors.

## Citation

If you use MGPUSim in your research, please cite our ISCA '19 paper. 

```bibtex
@inproceedings{sun19mgpusim, 
    author = {Sun, Yifan and Baruah, Trinayan and Mojumder, Saiful A. and Dong, Shi and Gong, Xiang and Treadway, Shane and Bao, Yuhui and Hance, Spencer and McCardwell, Carter and Zhao, Vincent and Barclay, Harrison and Ziabari, Amir Kavyan and Chen, Zhongliang and Ubal, Rafael and Abell\'{a}n, Jos\'{e} L. and Kim, John and Joshi, Ajay and Kaeli, David}, 
    title = {MGPUSim: Enabling Multi-GPU Performance Modeling and Optimization}, 
    year = {2019}, 
    isbn = {9781450366694}, 
    publisher = {Association for Computing Machinery}, 
    address = {New York, NY, USA}, 
    url = {https://doi.org/10.1145/3307650.3322230}, 
    doi = {10.1145/3307650.3322230}, 
    booktitle = {Proceedings of the 46th International Symposium on Computer Architecture}, 
    pages = {197–209}, 
    numpages = {13}, 
    keywords = {simulation, multi-GPU systems, memory management}, 
    location = {Phoenix, Arizona}, 
    series = {ISCA '19} 
}
```

Papers that use MGPUSim:

* Dynamic GMMU Bypass for Address Translation in Multi-GPU Systems
* Valkyrie: Leveraging Inter-TLB Locality to Enhance GPU Performance
* MGPU-TSM: A Multi-GPU System with Truly Shared Memory
* Griffin: Hardware-Software Support for Efficient Page Migration in Multi-GPU Systems
* HALCONE: A Hardware-Level Timestamp-based Cache Coherence Scheme for Multi-GPU systems
* Priority-Based PCIe Scheduling for Multi-Tenant Multi-GPU Systems
* Exploiting Adaptive Data Compression to Improve Performance and Energy-efficiency of Compute Workloads in Multi-GPU Systems


## License

MIT © Project Akita Developers.
