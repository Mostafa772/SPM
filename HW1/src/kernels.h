// SPM Module 1 - Partition Mapping Kernels
// Task 1: Plain C++ implementation (auto-vectorization friendly)
// Task 2: AVX2 intrinsics implementation
// Bonus: AVX-512 and multi-threaded versions

#ifndef SPM_MODULE1_KERNELS_H
#define SPM_MODULE1_KERNELS_H

#include "common.h"

namespace spm {

// ─────────────────────────────────────────────────────────────────
// Task 1: Plain C++ kernel (auto-vectorization friendly)
// ─────────────────────────────────────────────────────────────────
// Maps N keys to partition IDs using a hash function
// This version is designed to be auto-vectorized by the compiler
void map_scalar(const uint64_t* __restrict__ keys,
                uint32_t*       __restrict__ part_id,
                size_t N, uint64_t mask);

// ─────────────────────────────────────────────────────────────────
// Task 2: AVX2 intrinsics kernel (4 × uint64 per iteration)
// ─────────────────────────────────────────────────────────────────
// Explicit SIMD implementation using AVX2 intrinsics
// Processes 4 keys simultaneously with scalar tail for remainder
#if defined(__AVX2__)
void map_avx2(const uint64_t* __restrict__ keys,
              uint32_t*       __restrict__ part_id,
              size_t N, uint64_t mask);
#endif

// ─────────────────────────────────────────────────────────────────
// Bonus: AVX-512 kernel (8 × uint64 per iteration)
// ─────────────────────────────────────────────────────────────────
#if defined(__AVX512F__)
void map_avx512(const uint64_t* __restrict__ keys,
                uint32_t*       __restrict__ part_id,
                size_t N, uint64_t mask);
#endif

// ─────────────────────────────────────────────────────────────────
// Bonus: Multi-threaded versions
// ─────────────────────────────────────────────────────────────────
void map_scalar_mt(const uint64_t* __restrict__ keys,
                   uint32_t*       __restrict__ part_id,
                   size_t N, uint64_t mask, int num_threads);

#if defined(__AVX2__)
void map_avx2_mt(const uint64_t* __restrict__ keys,
                 uint32_t*       __restrict__ part_id,
                 size_t N, uint64_t mask, int num_threads);
#endif

#if defined(__AVX512F__)
void map_avx512_mt(const uint64_t* __restrict__ keys,
                   uint32_t*       __restrict__ part_id,
                   size_t N, uint64_t mask, int num_threads);
#endif

// ─────────────────────────────────────────────────────────────────
// Helper: 64-bit hash function used by all kernels
// ─────────────────────────────────────────────────────────────────
inline uint64_t mix64(uint64_t x)
{
    x ^= x >> 30;
    x *= MIX64_C1;
    x ^= x >> 27;
    x *= MIX64_C2;
    x ^= x >> 31;
    return x;
}

} // namespace spm

#endif // SPM_MODULE1_KERNELS_H
