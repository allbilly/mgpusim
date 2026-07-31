#include "hip/hip_runtime.h"

#define TILEY_SHIFT 2

// This is the corrected production matrix-multiplication body. Compiling it
// at the production 64-VGPR ceiling creates the same four private dword spills
// as mmmKernel_local. Raising only that ceiling provides the paired no-spill
// control while preserving global, LDS, barrier, VALU, and output work.
#define PRIVATE_SPILL_PROBE_BODY                                               \
  do {                                                                         \
    __shared__ float4 blockA[8 * 8 * 4];                                      \
    const int lIdX = hipThreadIdx_x;                                          \
    const int lIdY = hipThreadIdx_y;                                          \
    const int lSizeX = hipBlockDim_x;                                         \
    const int gIdX = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;          \
    const int gIdY = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;          \
    const int gSizeX = hipGridDim_x * hipBlockDim_x;                          \
    const int blockPos = lIdX + lSizeX * (lIdY << TILEY_SHIFT);               \
    const int globalPos = gIdX + (gIdY << TILEY_SHIFT) * gSizeX;              \
    float4 sum0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);                        \
    float4 sum1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);                        \
    float4 sum2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);                        \
    float4 sum3 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);                        \
    const int temp = widthA / 4;                                              \
    const int numLoops = temp / lSizeX;                                       \
    for (int i = 0; i < numLoops; ++i) {                                     \
      const int globalPosA =                                                  \
          i * lSizeX + lIdX + (gIdY << TILEY_SHIFT) * temp;                  \
      blockA[blockPos] = matrixA[globalPosA];                                 \
      blockA[blockPos + lSizeX] = matrixA[globalPosA + temp];                 \
      blockA[blockPos + 2 * lSizeX] = matrixA[globalPosA + 2 * temp];         \
      blockA[blockPos + 3 * lSizeX] = matrixA[globalPosA + 3 * temp];         \
      __syncthreads();                                                        \
      const int globalPosB =                                                  \
          gIdX + ((i * lSizeX) << TILEY_SHIFT) * gSizeX;                     \
      for (int j = 0; j < lSizeX * 4; j += 4) {                              \
        const float4 tempA0 = blockA[(j >> 2) + lIdY * 4 * lSizeX];          \
        const float4 tempA1 = blockA[(j >> 2) + (lIdY * 4 + 1) * lSizeX];    \
        const float4 tempA2 = blockA[(j >> 2) + (lIdY * 4 + 2) * lSizeX];    \
        const float4 tempA3 = blockA[(j >> 2) + (lIdY * 4 + 3) * lSizeX];    \
        const float4 tempB0 = matrixB[globalPosB + j * gSizeX];              \
        const float4 tempB1 = matrixB[globalPosB + (j + 1) * gSizeX];        \
        const float4 tempB2 = matrixB[globalPosB + (j + 2) * gSizeX];        \
        const float4 tempB3 = matrixB[globalPosB + (j + 3) * gSizeX];        \
        sum0.x += tempA0.x * tempB0.x + tempA0.y * tempB1.x +                \
                  tempA0.z * tempB2.x + tempA0.w * tempB3.x;                 \
        sum0.y += tempA0.x * tempB0.y + tempA0.y * tempB1.y +                \
                  tempA0.z * tempB2.y + tempA0.w * tempB3.y;                 \
        sum0.z += tempA0.x * tempB0.z + tempA0.y * tempB1.z +                \
                  tempA0.z * tempB2.z + tempA0.w * tempB3.z;                 \
        sum0.w += tempA0.x * tempB0.w + tempA0.y * tempB1.w +                \
                  tempA0.z * tempB2.w + tempA0.w * tempB3.w;                 \
        sum1.x += tempA1.x * tempB0.x + tempA1.y * tempB1.x +                \
                  tempA1.z * tempB2.x + tempA1.w * tempB3.x;                 \
        sum1.y += tempA1.x * tempB0.y + tempA1.y * tempB1.y +                \
                  tempA1.z * tempB2.y + tempA1.w * tempB3.y;                 \
        sum1.z += tempA1.x * tempB0.z + tempA1.y * tempB1.z +                \
                  tempA1.z * tempB2.z + tempA1.w * tempB3.z;                 \
        sum1.w += tempA1.x * tempB0.w + tempA1.y * tempB1.w +                \
                  tempA1.z * tempB2.w + tempA1.w * tempB3.w;                 \
        sum2.x += tempA2.x * tempB0.x + tempA2.y * tempB1.x +                \
                  tempA2.z * tempB2.x + tempA2.w * tempB3.x;                 \
        sum2.y += tempA2.x * tempB0.y + tempA2.y * tempB1.y +                \
                  tempA2.z * tempB2.y + tempA2.w * tempB3.y;                 \
        sum2.z += tempA2.x * tempB0.z + tempA2.y * tempB1.z +                \
                  tempA2.z * tempB2.z + tempA2.w * tempB3.z;                 \
        sum2.w += tempA2.x * tempB0.w + tempA2.y * tempB1.w +                \
                  tempA2.z * tempB2.w + tempA2.w * tempB3.w;                 \
        sum3.x += tempA3.x * tempB0.x + tempA3.y * tempB1.x +                \
                  tempA3.z * tempB2.x + tempA3.w * tempB3.x;                 \
        sum3.y += tempA3.x * tempB0.y + tempA3.y * tempB1.y +                \
                  tempA3.z * tempB2.y + tempA3.w * tempB3.y;                 \
        sum3.z += tempA3.x * tempB0.z + tempA3.y * tempB1.z +                \
                  tempA3.z * tempB2.z + tempA3.w * tempB3.z;                 \
        sum3.w += tempA3.x * tempB0.w + tempA3.y * tempB1.w +                \
                  tempA3.z * tempB2.w + tempA3.w * tempB3.w;                 \
      }                                                                       \
      __syncthreads();                                                        \
    }                                                                         \
    matrixC[globalPos] = sum0;                                                \
    matrixC[globalPos + gSizeX] = sum1;                                       \
    matrixC[globalPos + 2 * gSizeX] = sum2;                                   \
    matrixC[globalPos + 3 * gSizeX] = sum3;                                   \
  } while (false)

extern "C" __global__ __launch_bounds__(64)
    __attribute__((amdgpu_num_vgpr(80))) void private_control4_kernel(
        float4 *__restrict__ matrixA, float4 *__restrict__ matrixB,
        float4 *__restrict__ matrixC, int widthA) {
  PRIVATE_SPILL_PROBE_BODY;
}

extern "C" __global__ void private_scratch4_kernel(
    float4 *__restrict__ matrixA, float4 *__restrict__ matrixB,
    float4 *__restrict__ matrixC, int widthA) {
  PRIVATE_SPILL_PROBE_BODY;
}

#undef PRIVATE_SPILL_PROBE_BODY
