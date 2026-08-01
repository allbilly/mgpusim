#include "hip/hip_runtime.h"

#include <cstdint>

// Keep the wait explicit: this probe distinguishes a serialized vmcnt(0)
// chain from a fixed window of loads that may coexist before a single wait.
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

#define DEFINE_INDEPENDENT2_KERNEL(SYMBOL, TYPE, KEEP_LIVE, REDUCE0, REDUCE1) \
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
    for (uint32_t r = 0; r < repeats; r += 2) {                               \
      const uint32_t base =                                                    \
          (uint32_t(blockIdx.x) + r) * address_groups + address_group;          \
      const TYPE value0 = input[(base + 0u * address_groups) & mask];           \
      const TYPE value1 = input[(base + 1u * address_groups) & mask];           \
      /* Keep both results live at one use so LLVM emits one drain only after \
         the complete load window; verify_hsaco.sh audits the machine code. */ \
      KEEP_LIVE;                                                               \
      checksum += (REDUCE0);                                                   \
      checksum += (REDUCE1);                                                   \
    }                                                                          \
    output[tid] = checksum;                                                     \
  }

#define DEFINE_INDEPENDENT8_KERNEL(                                            \
    SYMBOL, TYPE, KEEP_LIVE, REDUCE0, REDUCE1, REDUCE2, REDUCE3, REDUCE4,     \
    REDUCE5, REDUCE6, REDUCE7)                                                 \
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
    for (uint32_t r = 0; r < repeats; r += 8) {                               \
      const uint32_t base =                                                    \
          (uint32_t(blockIdx.x) + r) * address_groups + address_group;          \
      const TYPE value0 = input[(base + 0u * address_groups) & mask];           \
      const TYPE value1 = input[(base + 1u * address_groups) & mask];           \
      const TYPE value2 = input[(base + 2u * address_groups) & mask];           \
      const TYPE value3 = input[(base + 3u * address_groups) & mask];           \
      const TYPE value4 = input[(base + 4u * address_groups) & mask];           \
      const TYPE value5 = input[(base + 5u * address_groups) & mask];           \
      const TYPE value6 = input[(base + 6u * address_groups) & mask];           \
      const TYPE value7 = input[(base + 7u * address_groups) & mask];           \
      /* Keep all results live at one use so LLVM emits one drain only after  \
         the complete load window; verify_hsaco.sh audits the machine code. */ \
      KEEP_LIVE;                                                               \
      checksum += (REDUCE0);                                                   \
      checksum += (REDUCE1);                                                   \
      checksum += (REDUCE2);                                                   \
      checksum += (REDUCE3);                                                   \
      checksum += (REDUCE4);                                                   \
      checksum += (REDUCE5);                                                   \
      checksum += (REDUCE6);                                                   \
      checksum += (REDUCE7);                                                   \
    }                                                                          \
    output[tid] = checksum;                                                     \
  }

DEFINE_SERIAL_KERNEL(vmem_load_dword_serial, float, value)
#define KEEP_LIVE_DWORD2 asm volatile("" : : "v"(value0), "v"(value1))
#define KEEP_LIVE_DWORD                                                       \
  asm volatile("" : : "v"(value0), "v"(value1), "v"(value2), "v"(value3))
#define KEEP_LIVE_DWORD8                                                      \
  asm volatile("" : : "v"(value0), "v"(value1), "v"(value2), "v"(value3), \
               "v"(value4), "v"(value5), "v"(value6), "v"(value7))
#define KEEP_LIVE_DWORDX2_2                                                   \
  asm volatile("" : : "v"(value0.x), "v"(value0.y), "v"(value1.x),        \
               "v"(value1.y))
#define KEEP_LIVE_DWORDX2                                                     \
  asm volatile("" : : "v"(value0.x), "v"(value0.y), "v"(value1.x),         \
               "v"(value1.y), "v"(value2.x), "v"(value2.y),                \
               "v"(value3.x), "v"(value3.y))
#define KEEP_LIVE_DWORDX2_8                                                   \
  asm volatile("" : : "v"(value0.x), "v"(value0.y), "v"(value1.x),        \
               "v"(value1.y), "v"(value2.x), "v"(value2.y),               \
               "v"(value3.x), "v"(value3.y), "v"(value4.x),               \
               "v"(value4.y), "v"(value5.x), "v"(value5.y),               \
               "v"(value6.x), "v"(value6.y), "v"(value7.x),               \
               "v"(value7.y))
#define KEEP_LIVE_DWORDX4_2                                                   \
  asm volatile("" : : "v"(value0.x), "v"(value0.y), "v"(value0.z),        \
               "v"(value0.w), "v"(value1.x), "v"(value1.y),               \
               "v"(value1.z), "v"(value1.w))
#define KEEP_LIVE_DWORDX4                                                     \
  asm volatile("" : : "v"(value0.x), "v"(value0.y), "v"(value0.z),         \
               "v"(value0.w), "v"(value1.x), "v"(value1.y),                \
               "v"(value1.z), "v"(value1.w), "v"(value2.x),                \
               "v"(value2.y), "v"(value2.z), "v"(value2.w),                \
               "v"(value3.x), "v"(value3.y), "v"(value3.z),                \
               "v"(value3.w))
#define KEEP_LIVE_DWORDX4_8                                                   \
  asm volatile("" : : "v"(value0.x), "v"(value0.y), "v"(value0.z),        \
               "v"(value0.w), "v"(value1.x), "v"(value1.y),               \
               "v"(value1.z), "v"(value1.w), "v"(value2.x),               \
               "v"(value2.y), "v"(value2.z), "v"(value2.w),               \
               "v"(value3.x), "v"(value3.y), "v"(value3.z),               \
               "v"(value3.w), "v"(value4.x), "v"(value4.y),               \
               "v"(value4.z), "v"(value4.w), "v"(value5.x),               \
               "v"(value5.y), "v"(value5.z), "v"(value5.w),               \
               "v"(value6.x), "v"(value6.y), "v"(value6.z),               \
               "v"(value6.w), "v"(value7.x), "v"(value7.y),               \
               "v"(value7.z), "v"(value7.w))

DEFINE_INDEPENDENT2_KERNEL(vmem_load_dword_independent2, float,
                           KEEP_LIVE_DWORD2, value0, value1)

DEFINE_INDEPENDENT4_KERNEL(vmem_load_dword_independent4, float,
                           KEEP_LIVE_DWORD, value0, value1, value2, value3)
DEFINE_INDEPENDENT8_KERNEL(vmem_load_dword_independent8, float,
                           KEEP_LIVE_DWORD8, value0, value1, value2, value3,
                           value4, value5, value6, value7)

DEFINE_SERIAL_KERNEL(vmem_load_dwordx2_serial, float2, value.x + value.y)
DEFINE_INDEPENDENT2_KERNEL(vmem_load_dwordx2_independent2, float2,
                           KEEP_LIVE_DWORDX2_2,
                           value0.x + value0.y, value1.x + value1.y)
DEFINE_INDEPENDENT4_KERNEL(vmem_load_dwordx2_independent4, float2,
                           KEEP_LIVE_DWORDX2,
                           value0.x + value0.y, value1.x + value1.y,
                           value2.x + value2.y, value3.x + value3.y)
DEFINE_INDEPENDENT8_KERNEL(
    vmem_load_dwordx2_independent8, float2, KEEP_LIVE_DWORDX2_8,
    value0.x + value0.y, value1.x + value1.y,
    value2.x + value2.y, value3.x + value3.y,
    value4.x + value4.y, value5.x + value5.y,
    value6.x + value6.y, value7.x + value7.y)

DEFINE_SERIAL_KERNEL(vmem_load_dwordx4_serial, float4,
                     (value.x + value.y) + (value.z + value.w))
DEFINE_INDEPENDENT2_KERNEL(
    vmem_load_dwordx4_independent2, float4, KEEP_LIVE_DWORDX4_2,
    (value0.x + value0.y) + (value0.z + value0.w),
    (value1.x + value1.y) + (value1.z + value1.w))
DEFINE_INDEPENDENT4_KERNEL(
    vmem_load_dwordx4_independent4, float4, KEEP_LIVE_DWORDX4,
    (value0.x + value0.y) + (value0.z + value0.w),
    (value1.x + value1.y) + (value1.z + value1.w),
    (value2.x + value2.y) + (value2.z + value2.w),
    (value3.x + value3.y) + (value3.z + value3.w))
DEFINE_INDEPENDENT8_KERNEL(
    vmem_load_dwordx4_independent8, float4, KEEP_LIVE_DWORDX4_8,
    (value0.x + value0.y) + (value0.z + value0.w),
    (value1.x + value1.y) + (value1.z + value1.w),
    (value2.x + value2.y) + (value2.z + value2.w),
    (value3.x + value3.y) + (value3.z + value3.w),
    (value4.x + value4.y) + (value4.z + value4.w),
    (value5.x + value5.y) + (value5.z + value5.w),
    (value6.x + value6.y) + (value6.z + value6.w),
    (value7.x + value7.y) + (value7.z + value7.w))

#undef DEFINE_INDEPENDENT8_KERNEL
#undef DEFINE_INDEPENDENT4_KERNEL
#undef DEFINE_INDEPENDENT2_KERNEL
#undef DEFINE_SERIAL_KERNEL
#undef KEEP_LIVE_DWORDX4_8
#undef KEEP_LIVE_DWORDX4
#undef KEEP_LIVE_DWORDX4_2
#undef KEEP_LIVE_DWORDX2_8
#undef KEEP_LIVE_DWORDX2
#undef KEEP_LIVE_DWORDX2_2
#undef KEEP_LIVE_DWORD8
#undef KEEP_LIVE_DWORD
#undef KEEP_LIVE_DWORD2
#undef VMEM_WAIT
