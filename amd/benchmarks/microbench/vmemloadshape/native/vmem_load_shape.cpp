#include "hip/hip_runtime.h"

#include <cstdint>

// Keep the wait explicit: this probe distinguishes a serialized vmcnt(0)
// chain from four loads that may coexist before a single wait.
#define VMEM_WAIT() __builtin_amdgcn_s_waitcnt(0x0f70)

#define DEFINE_SERIAL_KERNEL(SYMBOL, TYPE, REDUCE)                            \
  extern "C" __global__ void SYMBOL(                                         \
      const TYPE *__restrict__ input, float *__restrict__ output,             \
      uint32_t repeats, uint32_t vector_elements, uint32_t address_groups,     \
      uint32_t threads_per_block) {                                            \
    const uint32_t tid =                                                       \
        uint32_t(blockIdx.x) * threads_per_block + uint32_t(threadIdx.x);       \
    const uint32_t lane = uint32_t(threadIdx.x) & 63u;                         \
    const uint32_t address_group = lane & (address_groups - 1u);               \
    const uint32_t mask = vector_elements - 1u;                                \
    float checksum = 0.0f;                                                     \
    for (uint32_t r = 0; r < repeats; ++r) {                                  \
      const uint32_t index =                                                   \
          ((uint32_t(blockIdx.x) + r) * address_groups + address_group) &      \
          mask;                                                                \
      const TYPE value = input[index];                                         \
      VMEM_WAIT();                                                             \
      checksum += (REDUCE);                                                    \
    }                                                                          \
    output[tid] = checksum;                                                     \
  }

#define DEFINE_INDEPENDENT4_KERNEL(SYMBOL, TYPE, KEEP_LIVE, REDUCE0, REDUCE1, \
                                   REDUCE2, REDUCE3)                           \
  extern "C" __global__ void SYMBOL(                                         \
      const TYPE *__restrict__ input, float *__restrict__ output,             \
      uint32_t repeats, uint32_t vector_elements, uint32_t address_groups,     \
      uint32_t threads_per_block) {                                            \
    const uint32_t tid =                                                       \
        uint32_t(blockIdx.x) * threads_per_block + uint32_t(threadIdx.x);       \
    const uint32_t lane = uint32_t(threadIdx.x) & 63u;                         \
    const uint32_t address_group = lane & (address_groups - 1u);               \
    const uint32_t mask = vector_elements - 1u;                                \
    float checksum = 0.0f;                                                     \
    for (uint32_t r = 0; r < repeats; r += 4) {                               \
      const uint32_t base =                                                    \
          (uint32_t(blockIdx.x) + r) * address_groups + address_group;          \
      const TYPE value0 = input[(base + 0u * address_groups) & mask];           \
      const TYPE value1 = input[(base + 1u * address_groups) & mask];           \
      const TYPE value2 = input[(base + 2u * address_groups) & mask];           \
      const TYPE value3 = input[(base + 3u * address_groups) & mask];           \
      /* The empty-asm operands keep all results live. LLVM inserts the one    \
         required vmcnt(0) before that use; verify_hsaco.sh audits it. */       \
      KEEP_LIVE;                                                               \
      checksum += (REDUCE0);                                                   \
      checksum += (REDUCE1);                                                   \
      checksum += (REDUCE2);                                                   \
      checksum += (REDUCE3);                                                   \
    }                                                                          \
    output[tid] = checksum;                                                     \
  }

DEFINE_SERIAL_KERNEL(vmem_load_dword_serial, float, value)
#define KEEP_LIVE_DWORD                                                       \
  asm volatile("" : : "v"(value0), "v"(value1), "v"(value2), "v"(value3))
#define KEEP_LIVE_DWORDX2                                                     \
  asm volatile("" : : "v"(value0.x), "v"(value0.y), "v"(value1.x),         \
               "v"(value1.y), "v"(value2.x), "v"(value2.y),                \
               "v"(value3.x), "v"(value3.y))
#define KEEP_LIVE_DWORDX4                                                     \
  asm volatile("" : : "v"(value0.x), "v"(value0.y), "v"(value0.z),         \
               "v"(value0.w), "v"(value1.x), "v"(value1.y),                \
               "v"(value1.z), "v"(value1.w), "v"(value2.x),                \
               "v"(value2.y), "v"(value2.z), "v"(value2.w),                \
               "v"(value3.x), "v"(value3.y), "v"(value3.z),                \
               "v"(value3.w))

DEFINE_INDEPENDENT4_KERNEL(vmem_load_dword_independent4, float,
                           KEEP_LIVE_DWORD, value0, value1, value2, value3)

DEFINE_SERIAL_KERNEL(vmem_load_dwordx2_serial, float2, value.x + value.y)
DEFINE_INDEPENDENT4_KERNEL(vmem_load_dwordx2_independent4, float2,
                           KEEP_LIVE_DWORDX2,
                           value0.x + value0.y, value1.x + value1.y,
                           value2.x + value2.y, value3.x + value3.y)

DEFINE_SERIAL_KERNEL(vmem_load_dwordx4_serial, float4,
                     (value.x + value.y) + (value.z + value.w))
DEFINE_INDEPENDENT4_KERNEL(
    vmem_load_dwordx4_independent4, float4, KEEP_LIVE_DWORDX4,
    (value0.x + value0.y) + (value0.z + value0.w),
    (value1.x + value1.y) + (value1.z + value1.w),
    (value2.x + value2.y) + (value2.z + value2.w),
    (value3.x + value3.y) + (value3.z + value3.w))

#undef DEFINE_INDEPENDENT4_KERNEL
#undef DEFINE_SERIAL_KERNEL
#undef KEEP_LIVE_DWORDX4
#undef KEEP_LIVE_DWORDX2
#undef KEEP_LIVE_DWORD
#undef VMEM_WAIT
