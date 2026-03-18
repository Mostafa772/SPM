// SPM Module 1 - Kernel Implementations

#include "kernels.h"
#include <immintrin.h>
#include <thread>
#include <vector>
#include <algorithm>

namespace spm {

// ─────────────────────────────────────────────────────────────────
// Task 1: Plain C++ kernel (auto-vectorization friendly)
// ─────────────────────────────────────────────────────────────────
void map_scalar(const uint64_t* __restrict__ keys,
                uint32_t*       __restrict__ part_id,
                size_t N, uint64_t mask)
{
    for (size_t i = 0; i < N; ++i) {
        uint64_t x = keys[i];
        x ^= x >> 30;
        x *= MIX64_C1;
        x ^= x >> 27;
        x *= MIX64_C2;
        x ^= x >> 31;
        part_id[i] = static_cast<uint32_t>(x & mask);
    }
}

// ─────────────────────────────────────────────────────────────────
// Task 2: AVX2 kernel (4 × uint64 per iteration)
// ─────────────────────────────────────────────────────────────────
#if defined(__AVX2__)
void map_avx2(const uint64_t* __restrict__ keys,
              uint32_t*       __restrict__ part_id,
              size_t N, uint64_t mask)
{
    const __m256i c1   = _mm256_set1_epi64x(static_cast<int64_t>(MIX64_C1));
    const __m256i c2   = _mm256_set1_epi64x(static_cast<int64_t>(MIX64_C2));
    const __m256i vmask = _mm256_set1_epi64x(static_cast<int64_t>(mask));

    size_t i = 0;
    // Vectorized main loop: process 4 keys at a time
    for (; i + 4 <= N; i += 4) {
        __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(keys + i));

        // x ^= x >> 30
        x = _mm256_xor_si256(x, _mm256_srli_epi64(x, 30));

        // x *= c1 (64-bit multiply emulation)
        // Since AVX2 doesn't have 64-bit multiply, we emulate it:
        // For x = xH*2^32 + xL and c = cH*2^32 + cL
        // Product (mod 2^64) = xL*cL + (xL*cH + xH*cL)*2^32
        {
            __m256i xL = _mm256_and_si256(x, _mm256_set1_epi64x(0xFFFFFFFF));
            __m256i xH = _mm256_srli_epi64(x, 32);
            __m256i cL = _mm256_and_si256(c1, _mm256_set1_epi64x(0xFFFFFFFF));
            __m256i cH = _mm256_srli_epi64(c1, 32);
            __m256i lo = _mm256_mul_epu32(xL, cL);
            __m256i m1 = _mm256_mul_epu32(xL, cH);
            __m256i m2 = _mm256_mul_epu32(xH, cL);
            __m256i mid = _mm256_add_epi64(m1, m2);
            mid = _mm256_slli_epi64(mid, 32);
            x = _mm256_add_epi64(lo, mid);
        }

        // x ^= x >> 27
        x = _mm256_xor_si256(x, _mm256_srli_epi64(x, 27));

        // x *= c2
        {
            __m256i xL = _mm256_and_si256(x, _mm256_set1_epi64x(0xFFFFFFFF));
            __m256i xH = _mm256_srli_epi64(x, 32);
            __m256i cL = _mm256_and_si256(c2, _mm256_set1_epi64x(0xFFFFFFFF));
            __m256i cH = _mm256_srli_epi64(c2, 32);
            __m256i lo = _mm256_mul_epu32(xL, cL);
            __m256i m1 = _mm256_mul_epu32(xL, cH);
            __m256i m2 = _mm256_mul_epu32(xH, cL);
            __m256i mid = _mm256_add_epi64(m1, m2);
            mid = _mm256_slli_epi64(mid, 32);
            x = _mm256_add_epi64(lo, mid);
        }

        // x ^= x >> 31
        x = _mm256_xor_si256(x, _mm256_srli_epi64(x, 31));

        // Apply mask
        x = _mm256_and_si256(x, vmask);

        // Extract and store 4 partition IDs
        part_id[i+0] = static_cast<uint32_t>(_mm256_extract_epi64(x, 0));
        part_id[i+1] = static_cast<uint32_t>(_mm256_extract_epi64(x, 1));
        part_id[i+2] = static_cast<uint32_t>(_mm256_extract_epi64(x, 2));
        part_id[i+3] = static_cast<uint32_t>(_mm256_extract_epi64(x, 3));
    }

    // Scalar tail for remaining elements
    for (; i < N; ++i) {
        part_id[i] = static_cast<uint32_t>(mix64(keys[i]) & mask);
    }
}
#endif // __AVX2__

// ─────────────────────────────────────────────────────────────────
// Bonus: AVX-512 kernel (8 × uint64 per iteration)
// ─────────────────────────────────────────────────────────────────
#if defined(__AVX512F__)
void map_avx512(const uint64_t* __restrict__ keys,
                uint32_t*       __restrict__ part_id,
                size_t N, uint64_t mask)
{
    const __m512i c1   = _mm512_set1_epi64(static_cast<int64_t>(MIX64_C1));
    const __m512i c2   = _mm512_set1_epi64(static_cast<int64_t>(MIX64_C2));
    const __m512i vmask = _mm512_set1_epi64(static_cast<int64_t>(mask));

    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        __m512i x = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(keys + i));

        // x ^= x >> 30
        x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 30));

        // x *= c1
        {
            __m512i xL = _mm512_and_si512(x, _mm512_set1_epi64(0xFFFFFFFF));
            __m512i xH = _mm512_srli_epi64(x, 32);
            __m512i cL = _mm512_and_si512(c1, _mm512_set1_epi64(0xFFFFFFFF));
            __m512i cH = _mm512_srli_epi64(c1, 32);
            __m512i lo = _mm512_mul_epu32(xL, cL);
            __m512i m1 = _mm512_mul_epu32(xL, cH);
            __m512i m2 = _mm512_mul_epu32(xH, cL);
            __m512i mid = _mm512_add_epi64(m1, m2);
            mid = _mm512_slli_epi64(mid, 32);
            x = _mm512_add_epi64(lo, mid);
        }

        // x ^= x >> 27
        x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 27));

        // x *= c2
        {
            __m512i xL = _mm512_and_si512(x, _mm512_set1_epi64(0xFFFFFFFF));
            __m512i xH = _mm512_srli_epi64(x, 32);
            __m512i cL = _mm512_and_si512(c2, _mm512_set1_epi64(0xFFFFFFFF));
            __m512i cH = _mm512_srli_epi64(c2, 32);
            __m512i lo = _mm512_mul_epu32(xL, cL);
            __m512i m1 = _mm512_mul_epu32(xL, cH);
            __m512i m2 = _mm512_mul_epu32(xH, cL);
            __m512i mid = _mm512_add_epi64(m1, m2);
            mid = _mm512_slli_epi64(mid, 32);
            x = _mm512_add_epi64(lo, mid);
        }

        // x ^= x >> 31
        x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 31));

        // Apply mask
        x = _mm512_and_si512(x, vmask);

        // Extract and store
        alignas(64) uint64_t tmp[8];
        _mm512_store_si512(reinterpret_cast<__m512i*>(tmp), x);
        for (int j = 0; j < 8; ++j)
            part_id[i+j] = static_cast<uint32_t>(tmp[j]);
    }

    // Scalar tail
    for (; i < N; ++i) {
        part_id[i] = static_cast<uint32_t>(mix64(keys[i]) & mask);
    }
}
#endif // __AVX512F__

// ─────────────────────────────────────────────────────────────────
// Bonus: Multi-threaded scalar version
// ─────────────────────────────────────────────────────────────────
void map_scalar_mt(const uint64_t* __restrict__ keys,
                   uint32_t*       __restrict__ part_id,
                   size_t N, uint64_t mask, int num_threads)
{
    std::vector<std::thread> threads;
    size_t chunk_size = (N + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, N);
        if (start >= N) break;

        threads.emplace_back([=]() {
            for (size_t i = start; i < end; ++i) {
                uint64_t x = keys[i];
                x ^= x >> 30;
                x *= MIX64_C1;
                x ^= x >> 27;
                x *= MIX64_C2;
                x ^= x >> 31;
                part_id[i] = static_cast<uint32_t>(x & mask);
            }
        });
    }

    for (auto& t : threads) t.join();
}

// ─────────────────────────────────────────────────────────────────
// Bonus: Multi-threaded AVX2 version
// ─────────────────────────────────────────────────────────────────
#if defined(__AVX2__)
void map_avx2_mt(const uint64_t* __restrict__ keys,
                 uint32_t*       __restrict__ part_id,
                 size_t N, uint64_t mask, int num_threads)
{
    std::vector<std::thread> threads;
    size_t chunk_size = (N + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, N);
        if (start >= N) break;

        threads.emplace_back([=]() {
            map_avx2(keys + start, part_id + start, end - start, mask);
        });
    }

    for (auto& t : threads) t.join();
}
#endif // __AVX2__

// ─────────────────────────────────────────────────────────────────
// Bonus: Multi-threaded AVX-512 version
// ─────────────────────────────────────────────────────────────────
#if defined(__AVX512F__)
void map_avx512_mt(const uint64_t* __restrict__ keys,
                   uint32_t*       __restrict__ part_id,
                   size_t N, uint64_t mask, int num_threads)
{
    std::vector<std::thread> threads;
    size_t chunk_size = (N + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, N);
        if (start >= N) break;

        threads.emplace_back([=]() {
            map_avx512(keys + start, part_id + start, end - start, mask);
        });
    }

    for (auto& t : threads) t.join();
}
#endif // __AVX512F__

} // namespace spm
