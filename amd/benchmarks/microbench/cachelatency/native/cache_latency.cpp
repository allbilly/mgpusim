/*
 * HIP kernel for the cache_latency microbenchmark (gfx942 / CDNA3)
 * Extracted from sarchlab/gpu_benchmarks tier1/cache_latency.
 *
 * Single-thread pointer-chasing latency measurement: one thread walks a
 * linked-list-style index chain where each array element holds the index of
 * the next element to visit. Because each load depends on the previous result
 * (a true data dependency), the loads cannot overlap, directly exposing the
 * per-access memory latency rather than bandwidth.
 *
 * Launched with a single work-item (grid 1, block 1), matching the HIP source.
 * The kernel reads blockIdx.x / threadIdx.x only to guard against extra
 * work-items; it uses a constant launch geometry, so no hidden ABI arguments
 * are emitted (kernarg_segment_size = 24).
 */
#include "hip/hip_runtime.h"
#include <cstdint>

extern "C" __global__ void pointer_chase_kernel(
    const uint32_t* __restrict__ arr,
    uint32_t start_idx,
    uint32_t num_accesses,
    uint32_t* result)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    uint32_t idx = start_idx;

    for (uint32_t i = 0; i < num_accesses; ++i) {
        idx = arr[idx];
    }

    *result = idx;
}

/*
 * Vector-memory variant. Each active lane follows a disjoint randomized cycle
 * of cache-line-spaced nodes. Loading the lane's start from global memory and
 * indexing the chain with that lane-dependent value forces the dependent loads
 * through the vector/global-memory path instead of the scalar-cache path.
 */
extern "C" __global__ void vector_pointer_chase_kernel(
    const uint32_t* __restrict__ arr,
    const uint32_t* __restrict__ start_indices,
    uint32_t num_accesses,
    uint32_t active_lanes,
    uint32_t* result)
{
    const uint32_t lane = threadIdx.x;
    if (blockIdx.x != 0 || lane >= active_lanes) return;

    uint32_t idx = start_indices[lane];

    for (uint32_t i = 0; i < num_accesses; ++i) {
        idx = arr[idx];
    }

    result[lane] = idx;
}
