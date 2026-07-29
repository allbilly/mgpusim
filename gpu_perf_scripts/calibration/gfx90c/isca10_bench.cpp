// gfx90c ISCA-10 hardware timing harness.
// Compiles native HIP kernels from amd/benchmarks/*/native/ and reports
// average hipEvent kernel time (µs) for the same problem sizes as the sim.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static float elapsed_us(hipEvent_t a, hipEvent_t b) {
  float ms = 0;
  HIP_CHECK(hipEventElapsedTime(&ms, a, b));
  return ms * 1000.0f;
}

template <typename F>
static float time_iters(int iters, F &&launch) {
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  // Warmup
  launch();
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

// ---- kernels inlined / linked ----
// vectoradd + relu ship a main() in native/; keep kernels here to avoid
// hipcc treating .o as HIP source during the final link.
__global__ void vectoradd_float(float *a, const float *b, const float *c,
                                int width, int height) {
  int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
  int i = y * width + x;
  if (i < width * height)
    a[i] = b[i] + c[i];
}
__global__ void ReLUForward(const int count, float *in, float *out) {
  int index = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  if (index < count)
    out[index] = in[index] > 0 ? in[index] : 0;
}
extern "C" __global__ void mmmKernel_local(float4 *matrixA, float4 *matrixB,
                                           float4 *matrixC, int widthA);
extern "C" __global__ void
matrixTranspose(float4 *__restrict__ output, float4 *__restrict__ input,
                float4 *__restrict__ block, unsigned int wiWidth,
                unsigned int wiHeight, unsigned int num_of_blocks_x,
                unsigned int group_x_offset, unsigned int group_y_offset);
extern "C" __global__ void Encrypt(unsigned char *input,
                                   unsigned int *expanded_key,
                                   unsigned char *s);
extern "C" __global__ void FIR(float *output, float *coeff, float *input,
                               float *history, unsigned int num_tap);
extern "C" __global__ void kmeans_kernel_compute(float *feature,
                                                 float *clusters,
                                                 int *membership, int npoints,
                                                 int nclusters, int nfeatures,
                                                 int offset, int size);
extern "C" __global__ void kmeans_kernel_swap(float *feature,
                                              float *feature_swap, int npoints,
                                              int nfeatures);
extern "C" __global__ void PageRankUpdateGpu(unsigned int num_rows,
                                             unsigned int *rowOffset,
                                             unsigned int *col, float *val,
                                             float *x, float *y);
extern "C" __global__ void nw_kernel1(int *reference_d, int *input_itemsets_d,
                                      int *output_itemsets_d, int cols,
                                      int penalty, int blk, int block_size,
                                      int block_width, int worksize,
                                      int offset_r, int offset_c);
extern "C" __global__ void nw_kernel2(int *reference_d, int *input_itemsets_d,
                                      int *output_itemsets_d, int cols,
                                      int penalty, int blk, int block_size,
                                      int block_width, int worksize,
                                      int offset_r, int offset_c);

// Guarded bitonic: skip OOB pairs (sim tolerates; HW faults).
__global__ void BitonicSortGuarded(unsigned int *array, unsigned int n,
                                   unsigned int stage, unsigned int passOfStage,
                                   unsigned int direction) {
  unsigned int sortIncreasing = direction;
  unsigned int threadId = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  unsigned int pairDistance = 1u << (stage - passOfStage);
  unsigned int blockWidth = 2u * pairDistance;
  unsigned int leftId =
      (threadId % pairDistance) + (threadId / pairDistance) * blockWidth;
  unsigned int rightId = leftId + pairDistance;
  if (rightId >= n)
    return;
  unsigned int leftElement = array[leftId];
  unsigned int rightElement = array[rightId];
  unsigned int sameDirectionBlockWidth = 1u << stage;
  if ((threadId / sameDirectionBlockWidth) % 2 == 1)
    sortIncreasing = 1 - sortIncreasing;
  unsigned int greater, lesser;
  if (leftElement > rightElement) {
    greater = leftElement;
    lesser = rightElement;
  } else {
    greater = rightElement;
    lesser = leftElement;
  }
  if (sortIncreasing) {
    array[leftId] = lesser;
    array[rightId] = greater;
  } else {
    array[leftId] = greater;
    array[rightId] = lesser;
  }
}

static void bench_vectoradd(int iters) {
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
    hipLaunchKernelGGL(vectoradd_float, grid, block, 0, 0, a, b, c, width,
                       height);
  });
  printf("vectoradd %.3f\n", us);
  HIP_CHECK(hipFree(a));
  HIP_CHECK(hipFree(b));
  HIP_CHECK(hipFree(c));
}

static void bench_relu(int iters) {
  const int n = 65536;
  float *in, *out;
  HIP_CHECK(hipMalloc(&in, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&out, n * sizeof(float)));
  HIP_CHECK(hipMemset(in, 1, n * sizeof(float)));
  dim3 block(64);
  dim3 grid((n + 63) / 64);
  float us = time_iters(iters, [&] {
    hipLaunchKernelGGL(ReLUForward, grid, block, 0, 0, n, in, out);
  });
  printf("relu %.3f\n", us);
  HIP_CHECK(hipFree(in));
  HIP_CHECK(hipFree(out));
}

static void bench_matrixmult(int iters) {
  const int N = 128;
  float4 *A, *B, *C;
  size_t bytes = size_t(N) * N * sizeof(float);
  HIP_CHECK(hipMalloc(&A, bytes));
  HIP_CHECK(hipMalloc(&B, bytes));
  HIP_CHECK(hipMalloc(&C, bytes));
  HIP_CHECK(hipMemset(A, 1, bytes));
  HIP_CHECK(hipMemset(B, 1, bytes));
  // global = (N/4, N/4), local 8x8
  dim3 block(8, 8);
  dim3 grid(N / 4 / 8, N / 4 / 8);
  float us = time_iters(iters, [&] {
    hipLaunchKernelGGL(mmmKernel_local, grid, block, 0, 0, A, B, C, N);
  });
  printf("matrixmult %.3f\n", us);
  HIP_CHECK(hipFree(A));
  HIP_CHECK(hipFree(B));
  HIP_CHECK(hipFree(C));
}

static void bench_matrixtranspose(int iters) {
  const int width = 512;
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
    hipLaunchKernelGGL(matrixTranspose, grid, block, lds, 0, out, in,
                       (float4 *)nullptr, (unsigned)wiWidth, (unsigned)wiHeight,
                       (unsigned)numBlocks, 0u, 0u);
  });
  printf("matrixtranspose %.3f\n", us);
  HIP_CHECK(hipFree(in));
  HIP_CHECK(hipFree(out));
}

static void bench_bitonicsort(int iters) {
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
        hipLaunchKernelGGL(BitonicSortGuarded, grid, block, 0, 0, d,
                           (unsigned)length, (unsigned)stage, (unsigned)pass,
                           1u);
      }
    }
  });
  printf("bitonicsort %.3f\n", us);
  HIP_CHECK(hipFree(d));
}

static void bench_aes(int iters) {
  const int length = 4096; // bytes
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
    hipLaunchKernelGGL(Encrypt, grid, block, 0, 0, input, ek, sdev);
  });
  printf("aes %.3f\n", us);
  HIP_CHECK(hipFree(input));
  HIP_CHECK(hipFree(ek));
  HIP_CHECK(hipFree(sdev));
}

static void bench_fir(int iters) {
  const int length = 8192;
  const unsigned taps = 16;
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
    hipLaunchKernelGGL(FIR, grid, block, 0, 0, out, coeff, in, hist, taps);
  });
  printf("fir %.3f\n", us);
  HIP_CHECK(hipFree(out));
  HIP_CHECK(hipFree(coeff));
  HIP_CHECK(hipFree(in));
  HIP_CHECK(hipFree(hist));
}

static void bench_kmeans(int iters) {
  const int npoints = 4096, nfeatures = 16, nclusters = 5;
  float *feat, *feat_swap, *clusters;
  int *membership;
  HIP_CHECK(hipMalloc(&feat, npoints * nfeatures * sizeof(float)));
  HIP_CHECK(hipMalloc(&feat_swap, npoints * nfeatures * sizeof(float)));
  HIP_CHECK(hipMalloc(&clusters, nclusters * nfeatures * sizeof(float)));
  HIP_CHECK(hipMalloc(&membership, npoints * sizeof(int)));
  HIP_CHECK(hipMemset(feat, 1, npoints * nfeatures * sizeof(float)));
  HIP_CHECK(hipMemset(clusters, 1, nclusters * nfeatures * sizeof(float)));
  dim3 block(64);
  dim3 grid((npoints + 63) / 64);
  // max-iter=1: one swap + one compute (matches sim).
  float us = time_iters(iters, [&] {
    hipLaunchKernelGGL(kmeans_kernel_swap, grid, block, 0, 0, feat, feat_swap,
                       npoints, nfeatures);
    hipLaunchKernelGGL(kmeans_kernel_compute, grid, block, 0, 0, feat_swap,
                       clusters, membership, npoints, nclusters, nfeatures, 0,
                       0);
  });
  printf("kmeans %.3f\n", us);
  HIP_CHECK(hipFree(feat));
  HIP_CHECK(hipFree(feat_swap));
  HIP_CHECK(hipFree(clusters));
  HIP_CHECK(hipFree(membership));
}

static void bench_pagerank(int iters) {
  const unsigned num_nodes = 512;
  const float sparsity = 0.5f;
  unsigned num_conn =
      std::max(num_nodes, (unsigned)(num_nodes * num_nodes * sparsity));
  // Simple CSR: each row has num_conn/num_nodes edges (approx).
  unsigned edges_per = num_conn / num_nodes;
  num_conn = edges_per * num_nodes;
  std::vector<unsigned> row(num_nodes + 1), col(num_conn);
  std::vector<float> val(num_conn, 1.0f / edges_per), x(num_nodes, 1.0f),
      y(num_nodes, 0);
  for (unsigned r = 0; r <= num_nodes; r++)
    row[r] = r * edges_per;
  for (unsigned e = 0; e < num_conn; e++)
    col[e] = e % num_nodes;
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
      hipLaunchKernelGGL(PageRankUpdateGpu, grid, block, 0, 0, num_nodes, drow,
                         dcol, dval, dx, dy);
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
  // Match amd/benchmarks/rodinia/nw: blockSize=64, length=128.
  const int length = 128;
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
      dim3 grid(B * blk); // match sim: block_size * blk
      hipLaunchKernelGGL(nw_kernel1, grid, block, 0, 0, ref, in, out, cols,
                         penalty, blk, B, blockWidth, workSize, 0, 0);
    }
    for (int blk = workSize / B - 1; blk >= 1; blk--) {
      dim3 block(B);
      dim3 grid(B * blk);
      hipLaunchKernelGGL(nw_kernel2, grid, block, 0, 0, ref, in, out, cols,
                         penalty, blk, B, blockWidth, workSize, 0, 0);
    }
  });
  printf("nw %.3f\n", us);
  HIP_CHECK(hipFree(ref));
  HIP_CHECK(hipFree(in));
  HIP_CHECK(hipFree(out));
}

int main(int argc, char **argv) {
  int iters = 100;
  int bitonic_iters = 20;
  std::string only;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--iters") && i + 1 < argc)
      iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--only") && i + 1 < argc)
      only = argv[++i];
  }
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
  run("kmeans", bench_kmeans, iters);
  run("pagerank", bench_pagerank, iters);
  run("nw", bench_nw, iters);
  return 0;
}
