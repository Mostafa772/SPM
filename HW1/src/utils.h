// SPM Module 1 - Utility Functions
// Key generation, validation, and correctness checking

#ifndef SPM_MODULE1_UTILS_H
#define SPM_MODULE1_UTILS_H

#include "common.h"
#include <vector>

namespace spm {

// ─────────────────────────────────────────────────────────────────
// Key Generation
// ─────────────────────────────────────────────────────────────────
// Generates N deterministic keys from a seed using Mersenne Twister
void generate_keys(uint64_t* keys, size_t N, uint64_t seed);

// ─────────────────────────────────────────────────────────────────
// Correctness Verification
// ─────────────────────────────────────────────────────────────────
// Computes FNV-1a checksum over partition IDs (for large N comparison)
uint64_t checksum(const uint32_t* part_id, size_t N);

// Element-wise comparison (for small N verification)
bool element_wise_equal(const uint32_t* a, const uint32_t* b, size_t N);

// Validates all partition IDs are in range [0, P)
bool range_valid(const uint32_t* part_id, size_t N, uint64_t P);

// ─────────────────────────────────────────────────────────────────
// Distribution Analysis
// ─────────────────────────────────────────────────────────────────
// Computes and prints distribution statistics (min, max, avg, imbalance)
void distribution_stats(const uint32_t* part_id, size_t N, uint64_t P);

} // namespace spm

#endif // SPM_MODULE1_UTILS_H
