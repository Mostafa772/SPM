// SPM Module 1 - Utility Implementations

#include "utils.h"
#include <random>
#include <algorithm>
#include <cstdio>

namespace spm {

// ─────────────────────────────────────────────────────────────────
// Key Generation
// ─────────────────────────────────────────────────────────────────
void generate_keys(uint64_t* keys, size_t N, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    for (size_t i = 0; i < N; ++i)
        keys[i] = rng();
}

// ─────────────────────────────────────────────────────────────────
// Checksum (FNV-1a hash)
// ─────────────────────────────────────────────────────────────────
uint64_t checksum(const uint32_t* part_id, size_t N)
{
    uint64_t h = FNV1A_OFFSET;
    for (size_t i = 0; i < N; ++i) {
        h ^= static_cast<uint64_t>(part_id[i]);
        h *= FNV1A_PRIME;
    }
    return h;
}

// ─────────────────────────────────────────────────────────────────
// Element-wise comparison
// ─────────────────────────────────────────────────────────────────
bool element_wise_equal(const uint32_t* a, const uint32_t* b, size_t N)
{
    for (size_t i = 0; i < N; ++i) {
        if (a[i] != b[i]) {
            std::fprintf(stderr, "  mismatch at i=%zu: %u vs %u\n",
                         i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────
// Range validation
// ─────────────────────────────────────────────────────────────────
bool range_valid(const uint32_t* part_id, size_t N, uint64_t P)
{
    for (size_t i = 0; i < N; ++i) {
        if (part_id[i] >= static_cast<uint32_t>(P))
            return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────
// Distribution statistics
// ─────────────────────────────────────────────────────────────────
void distribution_stats(const uint32_t* part_id, size_t N, uint64_t P)
{
    std::vector<size_t> cnt(P, 0);
    for (size_t i = 0; i < N; ++i)
        cnt[part_id[i]]++;

    size_t mn = *std::min_element(cnt.begin(), cnt.end());
    size_t mx = *std::max_element(cnt.begin(), cnt.end());
    double avg = static_cast<double>(N) / static_cast<double>(P);

    std::printf("  Distribution  min=%zu  max=%zu  avg=%.1f  imbalance=%.3f\n",
                mn, mx, avg, static_cast<double>(mx) / avg);
}

} // namespace spm
