#include <iostream>
#include <vector>
#include <random>
#include <cstdint>
#include <cmath>
#include "benchmark.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

using namespace spm;

// -----------------------------------------------------------------------------
// Core implementations
// -----------------------------------------------------------------------------
void simple_hash(const uint64_t* __restrict__ keys, const int P, const size_t N, uint32_t* __restrict__ part) {
    for (size_t i = 0; i < N; ++i) {
        part[i] = keys[i] & (P - 1);
    }
}

constexpr uint64_t MIX64_C1 = UINT64_C(0xbf58476d1ce4e5b9);
constexpr uint64_t MIX64_C2 = UINT64_C(0x94d049bb133111eb);

void map_scalar(const uint64_t* __restrict keys, uint32_t* __restrict part_id, size_t N, uint64_t mask) {
    for (size_t i = 0; i < N; ++i) {
        uint64_t x = keys[i];
        x ^= x >> 30;
        x *= MIX64_C1;
        x ^= x >> 27;
        x *= MIX64_C2;
        x ^= x >> 31;
        part_id[i] = x & mask;
    }
}

void map_fibo(const uint64_t* __restrict keys, uint32_t* __restrict part_id, size_t N, uint32_t shift) {
    for (size_t i = 0; i < N; ++i) {
        part_id[i] = (keys[i] * 0x9E3779B97F4A7C15ULL) >> shift;
    }
}

#ifdef __AVX2__
void simple_hash_avx2(const uint64_t* __restrict__ keys, const int P, const size_t N, uint32_t* __restrict__ part) {
    __m256i mask_vec = _mm256_set1_epi64x(P - 1);
    __m256i permute_mask = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
    size_t i = 0;
    size_t limit = (N >= 4) ? (N - (N % 4)) : 0;

    for (; i < limit; i += 4) {
        __m256i k = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&keys[i]));
        __m256i p = _mm256_and_si256(k, mask_vec);

        __m256i packed256 = _mm256_permutevar8x32_epi32(p, permute_mask);
        __m128i packed128 = _mm256_castsi256_si128(packed256);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(&part[i]), packed128);
    }
    for (; i < N; ++i) {
        part[i] = keys[i] & (P - 1);
    }
}
#endif

// -----------------------------------------------------------------------------
// Testing Support
// -----------------------------------------------------------------------------
void generate_rand(const int N, std::vector<uint64_t>& keys, int seed = 42) {
    std::mt19937_64 rng(seed);
    for (int i = 0; i < N; ++i) {
        keys[i] = rng();
    }
}

bool verify_exact(const uint32_t* base, const uint32_t* test, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        if (base[i] != test[i]) {
            std::cerr << "Mismatch at index " << i << ": expected " << base[i] << ", got " << test[i] << "\n";
            return false;
        }
    }
    return true;
}

bool verify_checksum(const uint32_t* base, const uint32_t* test, size_t N) {
    uint64_t base_xor = 0, test_xor = 0;
    for (size_t i = 0; i < N; ++i) {
        base_xor ^= (static_cast<uint64_t>(base[i]) * (i + 1));
        test_xor ^= (static_cast<uint64_t>(test[i]) * (i + 1));
    }
    if (base_xor != test_xor) {
        std::cerr << "Checksum mismatch!\n";
        return false;
    }
    return true;
}

template<typename T>
double check_dist(const std::vector<T>& part, uint32_t P, size_t N) {
    std::vector<double> counts(P, 0.0);
    for (size_t i = 0; i < N; ++i) {
        counts[part[i]] += 1.0;
    }

    double opt_mu = static_cast<double>(N) / P;
    double variance = 0.0;

    for (size_t i = 0; i < P; ++i) {
        double diff = counts[i] - opt_mu;
        variance += (diff * diff);
    }

    variance /= static_cast<double>(P);
    double standard_deviation = std::sqrt(variance);

    return (standard_deviation / opt_mu) * 100.0;
}

int main(int argc, char** argv) {
    const char* exec_name = (argc > 0) ? argv[0] : "unknown";
    std::vector<BenchResult> all_results;
    std::vector<size_t> N_values = {10000, 1000000, 10000000, 50000000};
    std::vector<int> P_values = {16, 64, 256, 1024, 4096};
    constexpr int REPS = 25;
    std::vector<std::vector<double>> dists;

    for (size_t N : N_values) {
        std::vector<uint64_t> keys(N);
        generate_rand(N, keys);

        for (int P : P_values) {
            std::vector<uint32_t> part_plain(N, 0);
            std::vector<uint32_t> part_fibo(N, 0);
            std::vector<uint32_t> part_mix(N, 0);
            std::vector<double> cvs;

            std::printf("\n--- Running N=%zu, P=%d ---\n", N, P);
            print_table_header();

            uint32_t shift = 64 - __builtin_ctz(P);

            // 1. Plain version (Simple Hash)
            auto plain_fn = [&]() { simple_hash(keys.data(), P, N, part_plain.data()); };
            Stats stat_plain = spm::benchmark(plain_fn, N, REPS, 0.0);
            print_table_row("Plain C++ (Simple)", stat_plain);
            all_results.push_back({"simple", N, P, stat_plain});
            cvs.push_back(check_dist(part_plain, P, N));

            // 1.b Fibonacci Hash
            auto fibo_fn = [&]() { map_fibo(keys.data(), part_fibo.data(), N, shift); };
            Stats stat_fibo = spm::benchmark(fibo_fn, N, REPS, stat_plain.median_s);
            print_table_row("Plain C++ (Fibo)", stat_fibo);
            all_results.push_back({"fibo", N, P, stat_fibo});
            cvs.push_back(check_dist(part_fibo, P, N));

            // 1.c Mix Hash
            auto mix_fn = [&]() { map_scalar(keys.data(), part_mix.data(), N, P - 1); };
            Stats stat_mix = spm::benchmark(mix_fn, N, REPS, stat_plain.median_s);
            print_table_row("Plain C++ (Mix)", stat_mix);
            all_results.push_back({"mix", N, P, stat_mix});
            cvs.push_back(check_dist(part_mix, P, N));

            // 2. AVX2 version (Simple Hash Vectorized)
#ifdef __AVX2__
            std::vector<uint32_t> part_avx2(N, 0);
            auto avx2_fn = [&]() { simple_hash_avx2(keys.data(), P, N, part_avx2.data()); };
            Stats stat_avx2 = spm::benchmark(avx2_fn, N, REPS, stat_plain.median_s);
            print_table_row("AVX2 Intrinsics", stat_avx2);
            all_results.push_back({"avx2", N, P, stat_avx2});
#else
            std::cout << "│ [INFO] Compiled without -mavx2. Skipping AVX2 benchmarks.          │\n";
#endif
            print_table_footer();
            
            dists.push_back(cvs);

#ifdef __AVX2__
            if (N <= 10000) {
                if (!verify_exact(part_plain.data(), part_avx2.data(), N)) {
                    std::cout << "  [FAIL] AVX2 exact mismatch with Plain C++ output!\n";
                }
            } else {
                if (!verify_checksum(part_plain.data(), part_avx2.data(), N)) {
                    std::cout << "  [FAIL] AVX2 checksum mismatch with Plain C++ output!\n";
                }
            }
#endif
        }
    }

    spm::export_csv(all_results, "sweep_results.csv", exec_name);

    // Print CV distributions at the end matching Old code logic
    int dist_idx = 0;
    for (size_t N : N_values) {
        for (int P : P_values) {
            std::cout << "\n##############################################################\n";
            std::cout << "N=" << N << ", P=" << P << " | CV%% for Simple, Fibo, Mix: ";
            for(double cv : dists[dist_idx]) {
                std::cout << cv << "%%  ";
            }
            std::cout << "\n";
            dist_idx++;
        }
    }

    std::cout << "\nResults exported to sweep_results.csv\n";
    return 0;
}
