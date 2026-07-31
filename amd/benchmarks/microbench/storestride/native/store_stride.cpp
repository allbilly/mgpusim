#include "hip/hip_runtime.h"
#include <cstdint>

// One wave emits 16 full 64-byte cache-line stores per loop iteration. Four
// adjacent lanes contribute one uint4 each to a line. stride_lines changes
// only the distance between those full lines; allocation_stride_lines keeps
// the allocation and initialization footprint constant across a sweep.
extern "C" __global__ __launch_bounds__(64) void full_line_store_stride_kernel(
    uint4 *__restrict__ output, uint32_t stride_lines,
    uint32_t allocation_stride_lines, uint32_t repeats) {
  const uint32_t lane = threadIdx.x;
  if (lane >= 64)
    return;

  const uint32_t line_in_epoch = lane >> 2;
  const uint32_t slot_in_line = lane & 3;
  const uint32_t epoch_span_lines = 15 * allocation_stride_lines + 1;

  for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
    const uint32_t epoch = blockIdx.x * repeats + repeat;
    const uint32_t line =
        epoch * epoch_span_lines + line_in_epoch * stride_lines;
    const uint32_t vector_index = line * 4 + slot_in_line;
    const uint32_t tag = epoch * 64 + lane + 1;
    output[vector_index] =
        make_uint4(tag, tag ^ 0x13579bdfu, tag ^ 0x2468ace0u,
                   tag ^ 0xa5a5a5a5u);
  }
}
