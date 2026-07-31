// gfx90c ISCA-10 hardware timing harness.
// Loads the exact HSACO files embedded by the Go benchmarks and reports
// average hipEvent kernel time (µs) for the same launch sequences as the sim.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "aes_tables.inc"

#define HIP_CHECK(x)                                                           \
  do {                                                                         \
    hipError_t _e = (x);                                                       \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr, "HIP %s:%d: %s\n", __FILE__, __LINE__,                   \
              hipGetErrorString(_e));                                          \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

static std::string repo_path(const char *relative) {
  const char *root = std::getenv("MGPUSIM_ROOT");
  if (!root || !*root) {
    fprintf(stderr, "MGPUSIM_ROOT must point to the repository root\n");
    std::exit(2);
  }
  return std::string(root) + "/" + relative;
}

template <typename T>
static std::vector<T> read_fixture(const char *relative, size_t count) {
  std::string path = repo_path(relative);
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    fprintf(stderr, "cannot open fixture %s\n", path.c_str());
    std::exit(2);
  }
  const std::streamsize expected =
      static_cast<std::streamsize>(count * sizeof(T));
  if (file.tellg() != expected) {
    fprintf(stderr, "fixture %s has the wrong size\n", path.c_str());
    std::exit(2);
  }
  std::vector<T> values(count);
  file.seekg(0);
  if (!file.read(reinterpret_cast<char *>(values.data()), expected)) {
    fprintf(stderr, "cannot read fixture %s\n", path.c_str());
    std::exit(2);
  }
  return values;
}

class ModuleKernel {
public:
  ModuleKernel(const char *hsaco, const char *name) {
    std::string path = repo_path(hsaco);
    HIP_CHECK(hipModuleLoad(&module_, path.c_str()));
    HIP_CHECK(hipModuleGetFunction(&function_, module_, name));
  }

  ModuleKernel(const ModuleKernel &) = delete;
  ModuleKernel &operator=(const ModuleKernel &) = delete;

  ~ModuleKernel() {
    if (module_)
      (void)hipModuleUnload(module_);
  }

  void launch(dim3 grid, dim3 block, unsigned shared_mem, void **args) const {
    HIP_CHECK(hipModuleLaunchKernel(function_, grid.x, grid.y, grid.z, block.x,
                                    block.y, block.z, shared_mem, nullptr, args,
                                    nullptr));
  }

private:
  hipModule_t module_ = nullptr;
  hipFunction_t function_ = nullptr;
};

static float elapsed_us(hipEvent_t a, hipEvent_t b) {
  float ms = 0;
  HIP_CHECK(hipEventElapsedTime(&ms, a, b));
  return ms * 1000.0f;
}

static int warmup_iters = 0;
static bool report_components = false;
static int cache_array_bytes = 16 * 1024;
static int cache_num_accesses = 131072;
static int cache_active_lanes = 0;
static int store_stride_lines = 1;
static int store_allocation_stride_lines = 128;
static int store_repeats = 32;
static int store_workgroups = 1;
static std::string scratch_variant = "scratch4";
static int scratch_workgroups = 16;
static int relu_length = 65536;
static int matrix_size = 128;
static int transpose_width = 512;
static int aes_length = 4096;
static int fir_length = 8192;
static int fir_taps = 16;
static int kmeans_npoints = 4096;
static int kmeans_nfeatures = 16;
static int kmeans_nclusters = 5;
static int pagerank_nodes = 512;
static int nw_length = 128;
static bool kmeans_preinitialize_swap = false;

template <typename F>
static float time_iters(int iters, F &&launch) {
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  for (int i = 0; i < warmup_iters; i++)
    launch();
  if (warmup_iters > 0)
    HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipEventRecord(start));
  for (int i = 0; i < iters; i++)
    launch();
  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));
  float us = elapsed_us(start, stop) / float(iters);
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return us;
}

static void bench_vectoradd(int iters) {
  ModuleKernel kernel(
      "amd/benchmarks/amdappsdk/vectoradd/kernels_gfx90c.hsaco",
      "_Z15vectoradd_floatPfPKfS1_ii");
  const int width = 65536, height = 1;
  const int n = width * height;
  float *a, *b, *c;
  HIP_CHECK(hipMalloc(&a, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&b, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&c, n * sizeof(float)));
  HIP_CHECK(hipMemset(b, 1, n * sizeof(float)));
  HIP_CHECK(hipMemset(c, 2, n * sizeof(float)));
  // Match sim launch: 1D grid, WG=64 (not native 16x16).
  dim3 block(64, 1, 1);
  dim3 grid((n + 63) / 64, 1, 1);
  float us = time_iters(iters, [&] {
    void *args[] = {&a, &b, &c, (void *)&width, (void *)&height};
    kernel.launch(grid, block, 0, args);
  });
  printf("vectoradd %.3f\n", us);
  HIP_CHECK(hipFree(a));
  HIP_CHECK(hipFree(b));
  HIP_CHECK(hipFree(c));
}

static void bench_relu(int iters) {
  ModuleKernel kernel(
      "amd/benchmarks/dnn/layer_benchmarks/relu/kernels_gfx90c.hsaco",
      "ReLUForward");
  const int n = relu_length;
  if (n < 1) {
    fprintf(stderr, "ReLU length must be positive\n");
    std::exit(2);
  }
  float *in, *out;
  HIP_CHECK(hipMalloc(&in, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&out, n * sizeof(float)));
  HIP_CHECK(hipMemset(in, 1, n * sizeof(float)));
  dim3 block(64);
  dim3 grid((n + 63) / 64);
  float us = time_iters(iters, [&] {
    void *args[] = {(void *)&n, &in, &out};
    kernel.launch(grid, block, 0, args);
  });
  printf("relu %.3f\n", us);
  HIP_CHECK(hipFree(in));
  HIP_CHECK(hipFree(out));
}

static void bench_matrixmult(int iters) {
  ModuleKernel kernel(
      "amd/benchmarks/amdappsdk/matrixmultiplication/kernels_gfx90c.hsaco",
      "mmmKernel_local");
  const int N = matrix_size;
  if (N != 32 && N != 64 && N != 128) {
    fprintf(stderr, "matrix size must be 32, 64, or 128\n");
    std::exit(2);
  }
  const std::string fixture_prefix =
      "gpu_perf_scripts/calibration/gfx90c/build/matrixmult_" +
      std::to_string(N) + "_";
  const size_t elements = size_t(N) * N;
  const std::vector<float> host_a =
      read_fixture<float>((fixture_prefix + "a.f32").c_str(), elements);
  const std::vector<float> host_b =
      read_fixture<float>((fixture_prefix + "b.f32").c_str(), elements);
  std::vector<float> host_c(elements,
                            std::numeric_limits<float>::quiet_NaN());
  float4 *A, *B, *C;
  const size_t bytes = elements * sizeof(float);
  HIP_CHECK(hipMalloc(&A, bytes));
  HIP_CHECK(hipMalloc(&B, bytes));
  HIP_CHECK(hipMalloc(&C, bytes));
  HIP_CHECK(hipMemcpy(A, host_a.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B, host_b.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(C, host_c.data(), bytes, hipMemcpyHostToDevice));
  // global = (N/4, N/4), local 8x8
  dim3 block(8, 8);
  dim3 grid(N / 4 / 8, N / 4 / 8);
  float us = time_iters(iters, [&] {
    void *args[] = {&A, &B, &C, (void *)&N};
    kernel.launch(grid, block, 0, args);
  });
  HIP_CHECK(hipMemcpy(host_c.data(), C, bytes, hipMemcpyDeviceToHost));

  constexpr double abs_tolerance = 1e-3;
  constexpr double rel_tolerance = 1e-4;
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      double expected = 0.0;
      for (int k = 0; k < N; ++k) {
        expected += double(host_a[size_t(row) * N + k]) *
                    double(host_b[size_t(k) * N + col]);
      }
      const double actual = host_c[size_t(row) * N + col];
      const double abs_error = std::abs(actual - expected);
      const double rel_error = abs_error / std::max(std::abs(expected), 1.0);
      max_abs_error = std::max(max_abs_error, abs_error);
      max_rel_error = std::max(max_rel_error, rel_error);
      const double tolerance =
          abs_tolerance + rel_tolerance * std::abs(expected);
      if (!std::isfinite(actual) || abs_error > tolerance) {
        fprintf(stderr,
                "matrixmult N=%d mismatch at [%d, %d]: expected %.9g, "
                "got %.9g, abs error %.3g exceeds %.3g\n",
                N, row, col, expected, actual, abs_error, tolerance);
        std::exit(3);
      }
    }
  }
  fprintf(stderr,
          "matrixmult N=%d verification Passed! max_abs_error=%.3g "
          "max_rel_error=%.3g\n",
          N, max_abs_error, max_rel_error);
  printf("matrixmult %.3f\n", us);
  HIP_CHECK(hipFree(A));
  HIP_CHECK(hipFree(B));
  HIP_CHECK(hipFree(C));
}

static void bench_matrixtranspose(int iters) {
  ModuleKernel kernel(
      "amd/benchmarks/amdappsdk/matrixtranspose/kernels_gfx90c.hsaco",
      "matrixTranspose");
  const int width = transpose_width;
  if (width < 64 || width % 64 != 0) {
    fprintf(stderr, "transpose width must be a positive multiple of 64\n");
    std::exit(2);
  }
  const int blockSize = 16;
  const int elems = 4;
  const int wiWidth = width / elems;
  const int wiHeight = width / elems;
  const int numBlocks = wiWidth / blockSize;
  size_t bytes = size_t(width) * width * sizeof(float);
  float4 *in, *out;
  HIP_CHECK(hipMalloc(&in, bytes));
  HIP_CHECK(hipMalloc(&out, bytes));
  HIP_CHECK(hipMemset(in, 1, bytes));
  dim3 block(blockSize, blockSize);
  dim3 grid(numBlocks, numBlocks);
  size_t lds = size_t(blockSize) * blockSize * elems * sizeof(float4);
  float us = time_iters(iters, [&] {
    float4 *block_ptr = nullptr;
    unsigned width_arg = wiWidth, height_arg = wiHeight;
    unsigned blocks_arg = numBlocks, zero = 0;
    void *args[] = {&out,       &in,         &block_ptr,
                    &width_arg, &height_arg, &blocks_arg,
                    &zero,      &zero};
    kernel.launch(grid, block, lds, args);
  });
  printf("matrixtranspose %.3f\n", us);
  HIP_CHECK(hipFree(in));
  HIP_CHECK(hipFree(out));
}

static void bench_bitonicsort(int iters) {
  ModuleKernel kernel(
      "amd/benchmarks/amdappsdk/bitonicsort/kernels_gfx90c.hsaco",
      "BitonicSort");
  const int length = 4096;
  unsigned *d;
  HIP_CHECK(hipMalloc(&d, length * sizeof(unsigned)));
  std::vector<unsigned> h(length);
  for (int i = 0; i < length; i++)
    h[i] = length - i;
  HIP_CHECK(hipMemcpy(d, h.data(), length * sizeof(unsigned),
                      hipMemcpyHostToDevice));
  int numStages = 0;
  for (int t = length; t > 1; t >>= 1)
    numStages++;
  dim3 block(64);
  dim3 grid((length / 2 + 63) / 64);
  // One "iteration" = full sort (all stages/passes), matching sim kernel_time.
  float us = time_iters(iters, [&] {
    for (int stage = 0; stage < numStages; stage++) {
      for (int pass = 0; pass < stage + 1; pass++) {
        unsigned stage_arg = stage, pass_arg = pass, direction = 1;
        void *args[] = {&d, &stage_arg, &pass_arg, &direction};
        kernel.launch(grid, block, 0, args);
      }
    }
  });
  printf("bitonicsort %.3f\n", us);
  HIP_CHECK(hipFree(d));
}

static void bench_aes(int iters) {
  ModuleKernel kernel("amd/benchmarks/heteromark/aes/kernels_gfx90c.hsaco",
                      "Encrypt");
  const int length = aes_length; // bytes
  if (length < 1024 || length % 1024 != 0) {
    fprintf(stderr, "AES length must be a positive multiple of 1024 bytes\n");
    std::exit(2);
  }
  unsigned char *input, *sdev;
  unsigned int *ek;
  HIP_CHECK(hipMalloc(&input, length));
  HIP_CHECK(hipMalloc(&ek, sizeof(expanded_key)));
  HIP_CHECK(hipMalloc(&sdev, sizeof(sbox)));
  HIP_CHECK(hipMemcpy(ek, expanded_key, sizeof(expanded_key),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(sdev, sbox, sizeof(sbox), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(input, 0xab, length));
  int numWi = length / 16;
  dim3 block(64);
  dim3 grid((numWi + 63) / 64);
  float us = time_iters(iters, [&] {
    void *args[] = {&input, &ek, &sdev};
    kernel.launch(grid, block, 0, args);
  });
  printf("aes %.3f\n", us);
  HIP_CHECK(hipFree(input));
  HIP_CHECK(hipFree(ek));
  HIP_CHECK(hipFree(sdev));
}

static void bench_fir(int iters) {
  ModuleKernel kernel("amd/benchmarks/heteromark/fir/kernels_gfx90c.hsaco",
                      "FIR");
  const int length = fir_length;
  const unsigned taps = unsigned(fir_taps);
  if (length < 1 || taps < 1) {
    fprintf(stderr, "FIR length and taps must be positive\n");
    std::exit(2);
  }
  float *out, *coeff, *in, *hist;
  HIP_CHECK(hipMalloc(&out, length * sizeof(float)));
  HIP_CHECK(hipMalloc(&coeff, taps * sizeof(float)));
  HIP_CHECK(hipMalloc(&in, length * sizeof(float)));
  HIP_CHECK(hipMalloc(&hist, taps * sizeof(float)));
  HIP_CHECK(hipMemset(coeff, 1, taps * sizeof(float)));
  HIP_CHECK(hipMemset(in, 1, length * sizeof(float)));
  HIP_CHECK(hipMemset(hist, 0, taps * sizeof(float)));
  dim3 block(256);
  dim3 grid(length / 256);
  float us = time_iters(iters, [&] {
    void *args[] = {&out, &coeff, &in, &hist, (void *)&taps};
    kernel.launch(grid, block, 0, args);
  });
  printf("fir %.3f\n", us);
  HIP_CHECK(hipFree(out));
  HIP_CHECK(hipFree(coeff));
  HIP_CHECK(hipFree(in));
  HIP_CHECK(hipFree(hist));
}

static uint64_t splitmix64_next(uint64_t &state) {
  state += UINT64_C(0x9E3779B97F4A7C15);
  uint64_t z = state;
  z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
  return z ^ (z >> 31);
}

static void bench_cache_latency(int iters) {
  const bool vector_probe = cache_active_lanes > 0;
  if (cache_active_lanes < 0 || cache_active_lanes > 64) {
    fprintf(stderr, "active lanes must be between 0 and 64\n");
    std::exit(2);
  }
  ModuleKernel kernel(
      "amd/benchmarks/microbench/cachelatency/kernels_gfx90c.hsaco",
      vector_probe ? "vector_pointer_chase_kernel" : "pointer_chase_kernel");
  constexpr int cacheline_bytes = 64;
  constexpr int stride = cacheline_bytes / sizeof(uint32_t);
  int n = std::max(cache_array_bytes / int(sizeof(uint32_t)), stride * 2);
  int nodes = n / stride;
  n = nodes * stride;

  std::vector<uint32_t> chain(n, 0);
  std::vector<uint32_t> starts;
  if (vector_probe) {
    const int nodes_per_lane = nodes / cache_active_lanes;
    if (nodes_per_lane < 2) {
      fprintf(stderr, "vector cache_latency requires at least two cache lines "
                      "per active lane\n");
      std::exit(2);
    }
    starts.resize(cache_active_lanes);
    for (int lane = 0; lane < cache_active_lanes; ++lane) {
      std::vector<uint32_t> permutation(nodes_per_lane);
      for (int i = 0; i < nodes_per_lane; ++i)
        permutation[i] = uint32_t(i);
      uint64_t state =
          uint64_t(uint32_t(42) + uint32_t(lane)) * UINT64_C(2654435761) +
          UINT64_C(0x9E3779B97F4A7C15);
      for (int i = nodes_per_lane - 1; i > 0; --i) {
        int j = int(splitmix64_next(state) % uint64_t(i + 1));
        std::swap(permutation[i], permutation[j]);
      }
      auto node_index = [&](uint32_t ordinal) {
        return (ordinal * uint32_t(cache_active_lanes) + uint32_t(lane)) *
               uint32_t(stride);
      };
      for (int i = 0; i < nodes_per_lane; ++i) {
        uint32_t node = node_index(permutation[i]);
        uint32_t next =
            node_index(permutation[(i + 1) % nodes_per_lane]);
        chain[node] = next;
      }
      starts[lane] = node_index(permutation[0]);
    }
  } else {
    std::vector<uint32_t> permutation(nodes);
    for (int i = 0; i < nodes; ++i)
      permutation[i] = uint32_t(i);
    uint64_t state = uint64_t(42) * UINT64_C(2654435761) +
                     UINT64_C(0x9E3779B97F4A7C15);
    for (int i = nodes - 1; i > 0; --i) {
      int j = int(splitmix64_next(state) % uint64_t(i + 1));
      std::swap(permutation[i], permutation[j]);
    }
    for (int i = 0; i < nodes; ++i) {
      uint32_t node = permutation[i] * stride;
      uint32_t next = permutation[(i + 1) % nodes] * stride;
      chain[node] = next;
    }
    starts.push_back(permutation[0] * stride);
  }
  uint32_t accesses = uint32_t(std::max(cache_num_accesses, 1));
  std::vector<uint32_t> expected = starts;
  for (uint32_t &value : expected)
    for (uint32_t i = 0; i < accesses; ++i)
      value = chain[value];

  uint32_t *device_chain, *device_starts = nullptr, *device_result;
  HIP_CHECK(hipMalloc(&device_chain, chain.size() * sizeof(uint32_t)));
  if (vector_probe)
    HIP_CHECK(hipMalloc(&device_starts, starts.size() * sizeof(uint32_t)));
  HIP_CHECK(hipMalloc(&device_result, expected.size() * sizeof(uint32_t)));
  HIP_CHECK(hipMemcpy(device_chain, chain.data(),
                      chain.size() * sizeof(uint32_t),
                      hipMemcpyHostToDevice));
  if (vector_probe)
    HIP_CHECK(hipMemcpy(device_starts, starts.data(),
                        starts.size() * sizeof(uint32_t),
                        hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(device_result, 0,
                      expected.size() * sizeof(uint32_t)));

  auto launch = [&] {
    if (vector_probe) {
      uint32_t active_lanes = uint32_t(cache_active_lanes);
      void *args[] = {&device_chain, &device_starts, &accesses, &active_lanes,
                      &device_result};
      kernel.launch(dim3(1), dim3(64), 0, args);
    } else {
      uint32_t start_index = starts[0];
      void *args[] = {&device_chain, &start_index, &accesses, &device_result};
      kernel.launch(dim3(1), dim3(1), 0, args);
    }
  };
  float us = time_iters(iters, launch);

  std::vector<uint32_t> actual(expected.size());
  HIP_CHECK(hipMemcpy(actual.data(), device_result,
                      actual.size() * sizeof(uint32_t),
                      hipMemcpyDeviceToHost));
  for (size_t lane = 0; lane < expected.size(); ++lane) {
    if (actual[lane] != expected[lane]) {
      fprintf(stderr,
              "cache_latency lane %zu mismatch: expected %u, got %u\n", lane,
              expected[lane], actual[lane]);
      std::exit(3);
    }
  }

  printf(vector_probe ? "cache_latency_vector %.3f\n"
                      : "cache_latency %.3f\n",
         us);
  printf(vector_probe ? "cache_latency_vector_ns_per_access %.6f\n"
                      : "cache_latency_ns_per_access %.6f\n",
         double(us) * 1000.0 / double(accesses));
  if (device_starts)
    HIP_CHECK(hipFree(device_starts));
  HIP_CHECK(hipFree(device_chain));
  HIP_CHECK(hipFree(device_result));
}

static void bench_storestride(int iters) {
  if (store_stride_lines < 1 || store_allocation_stride_lines < 1 ||
      store_stride_lines > store_allocation_stride_lines ||
      store_repeats < 1 || store_workgroups < 1) {
    fprintf(stderr, "invalid store-stride geometry\n");
    std::exit(2);
  }
  ModuleKernel kernel(
      "amd/benchmarks/microbench/storestride/kernels_gfx90c.hsaco",
      "full_line_store_stride_kernel");
  const uint64_t epoch_span_lines =
      uint64_t(15) * uint64_t(store_allocation_stride_lines) + 1;
  const uint64_t epochs =
      uint64_t(store_repeats) * uint64_t(store_workgroups);
  if (epochs > UINT32_MAX / epoch_span_lines ||
      epochs * epoch_span_lines > UINT32_MAX / 4) {
    fprintf(stderr, "store-stride vector index exceeds 32 bits\n");
    std::exit(2);
  }
  const uint64_t output_words = epochs * epoch_span_lines * 16;
  std::vector<uint32_t> expected(output_words, 0);

  for (int workgroup = 0; workgroup < store_workgroups; ++workgroup) {
    for (int repeat = 0; repeat < store_repeats; ++repeat) {
      const uint64_t epoch =
          uint64_t(workgroup * store_repeats + repeat);
      for (int lane = 0; lane < 64; ++lane) {
        const uint64_t line = epoch * epoch_span_lines +
                              uint64_t(lane >> 2) * store_stride_lines;
        const uint64_t word = line * 16 + uint64_t(lane & 3) * 4;
        const uint32_t tag = uint32_t(epoch * 64 + uint64_t(lane) + 1);
        expected[word + 0] = tag;
        expected[word + 1] = tag ^ 0x13579bdfu;
        expected[word + 2] = tag ^ 0x2468ace0u;
        expected[word + 3] = tag ^ 0xa5a5a5a5u;
      }
    }
  }

  uint32_t *output;
  HIP_CHECK(hipMalloc(&output, output_words * sizeof(uint32_t)));
  HIP_CHECK(hipMemset(output, 0, output_words * sizeof(uint32_t)));
  uint32_t stride = uint32_t(store_stride_lines);
  uint32_t allocation_stride = uint32_t(store_allocation_stride_lines);
  uint32_t repeats = uint32_t(store_repeats);
  float us = time_iters(iters, [&] {
    void *args[] = {&output, &stride, &allocation_stride, &repeats};
    kernel.launch(dim3(store_workgroups), dim3(64), 0, args);
  });

  std::vector<uint32_t> actual(output_words);
  HIP_CHECK(hipMemcpy(actual.data(), output,
                      output_words * sizeof(uint32_t),
                      hipMemcpyDeviceToHost));
  for (uint64_t word = 0; word < output_words; ++word) {
    if (actual[word] != expected[word]) {
      fprintf(stderr,
              "storestride word %llu mismatch: expected %#x, got %#x\n",
              static_cast<unsigned long long>(word), expected[word],
              actual[word]);
      std::exit(3);
    }
  }
  printf("storestride %.3f\n", us);
  printf("storestride_ns_per_repeat %.6f\n",
         double(us) * 1000.0 / double(store_repeats));
  HIP_CHECK(hipFree(output));
}

static void bench_scratchspill(int iters) {
  if (scratch_variant != "control4" && scratch_variant != "scratch4") {
    fprintf(stderr, "scratch variant must be control4 or scratch4\n");
    std::exit(2);
  }
  if (scratch_workgroups < 1 || scratch_workgroups > INT32_MAX / 32) {
    fprintf(stderr, "invalid private-spill work-group count\n");
    std::exit(2);
  }
  const char *symbol = scratch_variant == "control4"
                           ? "private_control4_kernel"
                           : "private_scratch4_kernel";
  ModuleKernel kernel(
      "amd/benchmarks/microbench/scratchspill/kernels_gfx90c.hsaco",
      symbol);
  constexpr int width_a = 32;
  constexpr int rows = 32;
  const int cols = scratch_workgroups * 32;
  const size_t a_count = size_t(rows) * width_a;
  const size_t b_count = size_t(width_a) * cols;
  const size_t c_count = size_t(rows) * cols;
  std::vector<float> host_a(a_count), host_b(b_count), expected(c_count);
  for (int row = 0; row < rows; ++row)
    for (int k = 0; k < width_a; ++k)
      host_a[size_t(row) * width_a + k] = float((row + k) % 7 - 3);
  for (int k = 0; k < width_a; ++k)
    for (int col = 0; col < cols; ++col)
      host_b[size_t(k) * cols + col] = float((k * 3 + col) % 5 - 2);
  for (int row = 0; row < rows; ++row)
    for (int col = 0; col < cols; ++col)
      for (int k = 0; k < width_a; ++k)
        expected[size_t(row) * cols + col] +=
            host_a[size_t(row) * width_a + k] *
            host_b[size_t(k) * cols + col];

  float *matrix_a, *matrix_b, *matrix_c;
  HIP_CHECK(hipMalloc(&matrix_a, a_count * sizeof(float)));
  HIP_CHECK(hipMalloc(&matrix_b, b_count * sizeof(float)));
  HIP_CHECK(hipMalloc(&matrix_c, c_count * sizeof(float)));
  HIP_CHECK(hipMemcpy(matrix_a, host_a.data(), a_count * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(matrix_b, host_b.data(), b_count * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(matrix_c, 0, c_count * sizeof(float)));
  int width_arg = width_a;
  float us = time_iters(iters, [&] {
    void *args[] = {&matrix_a, &matrix_b, &matrix_c, &width_arg};
    kernel.launch(dim3(scratch_workgroups, 1), dim3(8, 8), 0, args);
  });
  std::vector<float> actual(c_count);
  HIP_CHECK(hipMemcpy(actual.data(), matrix_c, c_count * sizeof(float),
                      hipMemcpyDeviceToHost));
  for (size_t i = 0; i < c_count; ++i) {
    if (std::fabs(actual[i] - expected[i]) > 1e-4f) {
      fprintf(stderr,
              "scratchspill %s element %zu mismatch: expected %g, got %g\n",
              scratch_variant.c_str(), i, expected[i], actual[i]);
      std::exit(3);
    }
  }
  printf("scratchspill_%s %.3f\n", scratch_variant.c_str(), us);
  printf("scratchspill_ns_per_workgroup %.6f\n",
         double(us) * 1000.0 / double(scratch_workgroups));
  HIP_CHECK(hipFree(matrix_a));
  HIP_CHECK(hipFree(matrix_b));
  HIP_CHECK(hipFree(matrix_c));
}

static void bench_kmeans(int iters) {
  ModuleKernel swap_kernel(
      "amd/benchmarks/heteromark/kmeans/kernels_gfx90c.hsaco",
      "kmeans_kernel_swap");
  ModuleKernel compute_kernel(
      "amd/benchmarks/heteromark/kmeans/kernels_gfx90c.hsaco",
      "kmeans_kernel_compute");
  const int npoints = kmeans_npoints;
  const int nfeatures = kmeans_nfeatures;
  const int nclusters = kmeans_nclusters;
  float *feat, *feat_swap, *clusters;
  int *membership;
  HIP_CHECK(hipMalloc(&feat, npoints * nfeatures * sizeof(float)));
  HIP_CHECK(hipMalloc(&feat_swap, npoints * nfeatures * sizeof(float)));
  HIP_CHECK(hipMalloc(&clusters, nclusters * nfeatures * sizeof(float)));
  HIP_CHECK(hipMalloc(&membership, npoints * sizeof(int)));
  std::vector<float> host_features;
  if (npoints == 4096 && nfeatures == 16) {
    host_features = read_fixture<float>(
        "gpu_perf_scripts/calibration/gfx90c/build/kmeans_features.f32",
        npoints * nfeatures);
  } else {
    std::mt19937 generator(0);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    host_features.resize(size_t(npoints) * nfeatures);
    for (float &value : host_features)
      value = distribution(generator);
  }
  HIP_CHECK(hipMemcpy(feat, host_features.data(),
                      host_features.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  if (kmeans_preinitialize_swap) {
    std::vector<float> transposed(host_features.size());
    for (int point = 0; point < npoints; ++point)
      for (int feature = 0; feature < nfeatures; ++feature)
        transposed[size_t(feature) * npoints + point] =
            host_features[size_t(point) * nfeatures + feature];
    HIP_CHECK(hipMemcpy(feat_swap, transposed.data(),
                        transposed.size() * sizeof(float),
                        hipMemcpyHostToDevice));
  }
  HIP_CHECK(hipMemcpy(clusters, host_features.data(),
                      nclusters * nfeatures * sizeof(float),
                      hipMemcpyHostToDevice));
  dim3 block(64);
  dim3 grid((npoints + 63) / 64);
  auto launch_swap = [&] {
    void *swap_args[] = {&feat, &feat_swap, (void *)&npoints,
                         (void *)&nfeatures};
    swap_kernel.launch(grid, block, 0, swap_args);
  };
  auto launch_compute = [&] {
    int offset = 0, size = 0;
    void *compute_args[] = {
        &feat_swap,      &clusters,         &membership,
        (void *)&npoints, (void *)&nclusters, (void *)&nfeatures,
        &offset,         &size,
    };
    compute_kernel.launch(grid, block, 0, compute_args);
  };
  if (kmeans_preinitialize_swap) {
    float compute_us = time_iters(iters, launch_compute);
    printf("kmeans %.3f\n", compute_us);
    printf("kmeans_compute_preinitialized %.3f\n", compute_us);
    HIP_CHECK(hipFree(feat));
    HIP_CHECK(hipFree(feat_swap));
    HIP_CHECK(hipFree(clusters));
    HIP_CHECK(hipFree(membership));
    return;
  }

  // max-iter=1: one swap + one compute (matches sim).
  float us = time_iters(iters, [&] {
    launch_swap();
    launch_compute();
  });
  printf("kmeans %.3f\n", us);
  if (report_components) {
    printf("kmeans_swap %.3f\n", time_iters(iters, launch_swap));
    printf("kmeans_compute %.3f\n", time_iters(iters, launch_compute));
  }
  HIP_CHECK(hipFree(feat));
  HIP_CHECK(hipFree(feat_swap));
  HIP_CHECK(hipFree(clusters));
  HIP_CHECK(hipFree(membership));
}

static void bench_pagerank(int iters) {
  ModuleKernel kernel(
      "amd/benchmarks/heteromark/pagerank/kernels_gfx90c.hsaco",
      "PageRankUpdateGpu");
  const unsigned num_nodes = unsigned(pagerank_nodes);
  if (num_nodes != 128 && num_nodes != 256 && num_nodes != 512) {
    fprintf(stderr, "PageRank nodes must be 128, 256, or 512\n");
    std::exit(2);
  }
  const unsigned num_conn = num_nodes * num_nodes / 2;
  const std::string fixture_prefix =
      "gpu_perf_scripts/calibration/gfx90c/build/pagerank_" +
      std::to_string(num_nodes) + "_";
  std::vector<unsigned> row = read_fixture<unsigned>(
      (fixture_prefix + "row_offsets.u32").c_str(), num_nodes + 1);
  std::vector<unsigned> col = read_fixture<unsigned>(
      (fixture_prefix + "columns.u32").c_str(), num_conn);
  std::vector<float> val = read_fixture<float>(
      (fixture_prefix + "values.f32").c_str(), num_conn);
  std::vector<float> x(num_nodes, 1.0f / num_nodes), y(num_nodes, 0);
  unsigned *drow, *dcol;
  float *dval, *dx, *dy;
  HIP_CHECK(hipMalloc(&drow, row.size() * sizeof(unsigned)));
  HIP_CHECK(hipMalloc(&dcol, col.size() * sizeof(unsigned)));
  HIP_CHECK(hipMalloc(&dval, val.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&dx, x.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&dy, y.size() * sizeof(float)));
  HIP_CHECK(hipMemcpy(drow, row.data(), row.size() * sizeof(unsigned),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dcol, col.data(), col.size() * sizeof(unsigned),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dval, val.data(), val.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dx, x.data(), x.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  const int iterations = 2;
  dim3 block(64);
  dim3 grid(num_nodes); // one WG per row (64 lanes)
  float us = time_iters(iters, [&] {
    for (int it = 0; it < iterations; it++) {
      void *args[] = {(void *)&num_nodes, &drow, &dcol, &dval, &dx, &dy};
      kernel.launch(grid, block, 0, args);
      std::swap(dx, dy);
    }
  });
  printf("pagerank %.3f\n", us);
  HIP_CHECK(hipFree(drow));
  HIP_CHECK(hipFree(dcol));
  HIP_CHECK(hipFree(dval));
  HIP_CHECK(hipFree(dx));
  HIP_CHECK(hipFree(dy));
}

static void bench_nw(int iters) {
  ModuleKernel kernel1("amd/benchmarks/rodinia/nw/kernels_gfx90c.hsaco",
                       "nw_kernel1");
  ModuleKernel kernel2("amd/benchmarks/rodinia/nw/kernels_gfx90c.hsaco",
                       "nw_kernel2");
  // Match amd/benchmarks/rodinia/nw: blockSize=64.
  const int length = nw_length;
  if (length < 64 || length % 64 != 0) {
    fprintf(stderr, "NW length must be a positive multiple of 64\n");
    std::exit(2);
  }
  const int B = 64;
  const int cols = length + 1;
  const int rows = length + 1;
  const int penalty = 10;
  int *ref, *in, *out;
  size_t bytes = size_t(cols) * rows * sizeof(int);
  HIP_CHECK(hipMalloc(&ref, bytes));
  HIP_CHECK(hipMalloc(&in, bytes));
  HIP_CHECK(hipMalloc(&out, bytes));
  HIP_CHECK(hipMemset(ref, 1, bytes));
  HIP_CHECK(hipMemset(in, 0, bytes));
  const int workSize = cols - 1;
  const int blockWidth = workSize / B;
  float us = time_iters(iters, [&] {
    for (int blk = 1; blk <= workSize / B; blk++) {
      dim3 block(B);
      dim3 grid(blk); // Go gridSize is B*blk work-items with a B-lane WG.
      int zero = 0;
      void *args[] = {&ref,
                      &in,
                      &out,
                      (void *)&cols,
                      (void *)&penalty,
                      &blk,
                      (void *)&B,
                      (void *)&blockWidth,
                      (void *)&workSize,
                      &zero,
                      &zero};
      kernel1.launch(grid, block, 0, args);
    }
    // The Go gfx90c benchmark currently launches kernel2 for the same blk
    // range as kernel1.
    for (int blk = 1; blk <= workSize / B; blk++) {
      dim3 block(B);
      dim3 grid(blk);
      int zero = 0;
      void *args[] = {&ref,
                      &in,
                      &out,
                      (void *)&cols,
                      (void *)&penalty,
                      &blk,
                      (void *)&B,
                      (void *)&blockWidth,
                      (void *)&workSize,
                      &zero,
                      &zero};
      kernel2.launch(grid, block, 0, args);
    }
  });
  printf("nw %.3f\n", us);
  HIP_CHECK(hipFree(ref));
  HIP_CHECK(hipFree(in));
  HIP_CHECK(hipFree(out));
}

int main(int argc, char **argv) {
  // MGPUSim samples execute one cold launch sequence, so cold single-shot
  // measurement is the comparable default. Use --warmup and --iters
  // explicitly for a steady-state experiment.
  int iters = 1;
  int bitonic_iters = 1;
  std::string only;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--iters") && i + 1 < argc)
      iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--warmup") && i + 1 < argc)
      warmup_iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--only") && i + 1 < argc)
      only = argv[++i];
    else if (!strcmp(argv[i], "--components"))
      report_components = true;
    else if (!strcmp(argv[i], "--array-bytes") && i + 1 < argc)
      cache_array_bytes = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--num-accesses") && i + 1 < argc)
      cache_num_accesses = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--active-lanes") && i + 1 < argc)
      cache_active_lanes = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--store-stride-lines") && i + 1 < argc)
      store_stride_lines = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--store-allocation-stride-lines") &&
             i + 1 < argc)
      store_allocation_stride_lines = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--store-repeats") && i + 1 < argc)
      store_repeats = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--store-workgroups") && i + 1 < argc)
      store_workgroups = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--scratch-variant") && i + 1 < argc)
      scratch_variant = argv[++i];
    else if (!strcmp(argv[i], "--scratch-workgroups") && i + 1 < argc)
      scratch_workgroups = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--relu-length") && i + 1 < argc)
      relu_length = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--matrix-size") && i + 1 < argc)
      matrix_size = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--transpose-width") && i + 1 < argc)
      transpose_width = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--aes-length") && i + 1 < argc)
      aes_length = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--fir-length") && i + 1 < argc)
      fir_length = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--fir-taps") && i + 1 < argc)
      fir_taps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--points") && i + 1 < argc)
      kmeans_npoints = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--features") && i + 1 < argc)
      kmeans_nfeatures = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--clusters") && i + 1 < argc)
      kmeans_nclusters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--pagerank-nodes") && i + 1 < argc)
      pagerank_nodes = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--nw-length") && i + 1 < argc)
      nw_length = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--preinitialize-swap"))
      kmeans_preinitialize_swap = true;
  }
  bitonic_iters = iters;
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  fprintf(stderr, "device=%s arch=%d.%d CUs=%d clock=%dMHz\n", prop.name,
          prop.major, prop.minor, prop.multiProcessorCount, prop.clockRate / 1000);

  auto run = [&](const char *name, auto fn, int n) {
    if (!only.empty() && only != name)
      return;
    fn(n);
  };
  run("vectoradd", bench_vectoradd, iters);
  run("relu", bench_relu, iters);
  run("matrixmult", bench_matrixmult, iters);
  run("matrixtranspose", bench_matrixtranspose, iters);
  run("bitonicsort", bench_bitonicsort, bitonic_iters);
  run("aes", bench_aes, iters);
  run("fir", bench_fir, iters);
  run("cache_latency", bench_cache_latency, iters);
  run("kmeans", bench_kmeans, iters);
  run("pagerank", bench_pagerank, iters);
  run("nw", bench_nw, iters);
  if (only == "storestride")
    bench_storestride(iters);
  if (only == "scratchspill")
    bench_scratchspill(iters);
  return 0;
}
