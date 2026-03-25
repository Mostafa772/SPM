    #include <iostream>
    #include <vector>
    #include <random>
    #include <immintrin.h>
    #include <cstdlib>
    #include <stdlib.h>
    #include <iostream>
    #include <chrono>
    #include <algorithm>
    #include <numeric>
    #include <cmath>
    #include <cstdio>
    #include <fstream>
    #include "benchmark.h"

    const int seed = 42;

    namespace spm {

    using Clock = std::chrono::high_resolution_clock;
    using Dur   = std::chrono::duration<double>;


    // ─────────────────────────────────────────────────────────────────
    // Benchmark function
    // ─────────────────────────────────────────────────────────────────
    Stats benchmark(std::function<void()> fn, size_t N, int reps, double baseline_s)
    {
        std::vector<double> times(reps);

        // Run benchmark reps times
        for (int r = 0; r < reps; ++r) {
            auto t0 = Clock::now();
            fn();
            auto t1 = Clock::now();
            times[r] = Dur(t1 - t0).count();
        }

        // Sort to get median
        std::sort(times.begin(), times.end());
        double med = times[reps / 2];

        // Compute mean for standard deviation
        double mean = std::accumulate(times.begin(), times.end(), 0.0) / reps;
        double var  = 0.0;
        for (double t : times)
            var += (t - mean) * (t - mean);
        double sd = std::sqrt(var / reps);

        // Build stats structure
        Stats s;
        s.median_s   = med;
        s.stddev_s   = sd;
        s.throughput = static_cast<double>(N) / med;
        s.speedup    = (baseline_s > 0.0) ? (baseline_s / med) : 1.0;
        return s;
    }

    // ─────────────────────────────────────────────────────────────────
    // Simple stats output
    // ─────────────────────────────────────────────────────────────────
    void print_stats(const char* name, const Stats& s)
    {
        std::printf("  %-22s  median=%8.3f ms  sd=%7.3f ms  "
                    "tput=%9.3e elem/s  speedup=%.2fx\n",
                    name,
                    s.median_s * 1e3,
                    s.stddev_s * 1e3,
                    s.throughput,
                    s.speedup);
    }

    // ─────────────────────────────────────────────────────────────────
    // Formatted table output
    // ─────────────────────────────────────────────────────────────────
    void print_table_header()
    {
        std::printf("\n");
        std::printf("┌────────────────────────┬────────────┬────────────┬──────────────┬──────────┐\n");
        std::printf("│ %-22s │ %-10s │ %-10s │ %-12s │ %-8s │\n",
                    "Method", "Median(ms)", "StdDev(ms)", "Throughput", "Speedup");
        std::printf("├────────────────────────┼────────────┼────────────┼──────────────┼──────────┤\n");
    }

    void print_table_row(const char* name, const Stats& s)
    {
        char tput[16];
        std::snprintf(tput, sizeof(tput), "%.2e e/s", s.throughput);
        std::printf("│ %-22s │ %10.3f │ %10.3f │ %-12s │ %8.2fx │\n",
                    name,
                    s.median_s * 1e3,
                    s.stddev_s * 1e3,
                    tput,
                    s.speedup);
    }

    void print_table_footer()
    {
        std::printf("└────────────────────────┴────────────┴────────────┴──────────────┴──────────┘\n");
    }

    // ─────────────────────────────────────────────────────────────────
    // CSV export
    // ─────────────────────────────────────────────────────────────────
        void export_csv(const std::vector<BenchResult>& results, const char* filename)
    {
        std::ofstream f(filename);
        if (!f) return;

        // Added N and P to the header
        f << "N,P,Method,Median_ms,StdDev_ms,Throughput_elem_s,Speedup\n";
        for (const auto& r : results) {
            f << r.N << ","
              << r.P << ","
              << r.name << ","
              << (r.stats.median_s * 1e3) << ","
              << (r.stats.stddev_s * 1e3) << ","
              << r.stats.throughput << ","
              << r.stats.speedup << "\n";
        }
    }

    } // namespace spm



    // 1. Unified Checksum (Keep it clean, use pointers)
    bool verify_checksum(const uint32_t* base, const uint32_t* test, size_t N) {
        uint64_t base_xor = 0, test_xor = 0;
        for (size_t i = 0; i < N; ++i) {
            base_xor ^= (base[i] * (i + 1));
            test_xor ^= (test[i] * (i + 1));
        }
        if (base_xor != test_xor) {
            std::cerr << "Checksum mismatch!\n";
            return false;
        }
        return true;
    }

    // 2. Unified One-to-One
    bool verify_exact(const uint32_t* base, const uint32_t* test, size_t N) {
        for (size_t i = 0; i < N; ++i) {
            if (base[i] != test[i]) {
                std::cerr << "Mismatch at index " << i << ": expected " << base[i] << ", got " << test[i] << "\n";
                return false;
            }
        }
        return true;
    }

    void generate_rand(const int size_N, std::vector<uint64_t>& keys)
    {
        std::mt19937_64 rng(seed);
        for (int i = 0; i < size_N; ++i)
            keys[i] = rng();
    }

    void simple_hash(const uint64_t* __restrict__ keys, const int P, const int N, uint32_t* __restrict__ part)
    {
        for (size_t i = 0; i < static_cast<size_t>(N); ++i)
        {
            part[i] = keys[i] & (P-1);
        }
    }

//     // avx2 optimized simple hashexp_avx2.cpp
// void simple_hash_avx2(const uint64_t* __restrict__ keys, const int P, const int N, uint32_t* __restrict__ part)
//     {
//         __m256i mask_vec = _mm256_set1_epi64x(P - 1);
//         __m256i permute_mask = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
//         size_t i = 0;
//
//         for (; i + 3 < static_cast<size_t>(N); i += 4)
//         {
//             // Use LOADU for unaligned
//             __m256i k = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&keys[i]));
//             __m256i p = _mm256_and_si256(k, mask_vec);
//
//             __m256i packed256 = _mm256_permutevar8x32_epi32(p, permute_mask);
//             __m128i packed128 = _mm256_castsi256_si128(packed256);
//
//             // Use STOREU for unaligned
//             _mm_storeu_si128(reinterpret_cast<__m128i*>(&part[i]), packed128);
//         }
//         for (; i < static_cast<size_t>(N); ++i)
//         {
//             part[i] = keys[i] & (P - 1);
//         }
//     }

// avx2 optimized simple hashexp_avx2.cpp

void simple_hash_avx2(const uint64_t* __restrict__ keys, const int P, const int N, uint32_t* __restrict__ part)

    {
        __m256i mask_vec = _mm256_set1_epi64x(P - 1);
        size_t i = 0;
        for (; i + 3 < static_cast<size_t>(N); i += 4)
        {
            __m256i k = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&keys[i]));
            __m256i p = _mm256_and_si256(k, mask_vec);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&part[i]), p);
        }
        // Scalar tail
        for (; i < static_cast<size_t>(N); ++i)
        {
            part[i] = keys[i] & (P - 1);
        }
    }

    // 3. The Core Experiment Runner
    void run_experiment(size_t N, int P, std::vector<spm::BenchResult>& results) {
        std::printf("\n--- Running N=%zu, P=%d ---\n", N, P);

        // 1. Allocate Aligned Memory
        // auto* keys = static_cast<uint64_t*>(aligned_alloc(32, N * sizeof(uint64_t)));
        // auto* part_base = static_cast<uint32_t*>(aligned_alloc(32, N * sizeof(uint32_t)));
        // auto* part_avx_unaligned = static_cast<uint32_t*>(aligned_alloc(32, N * sizeof(uint32_t)));
        // auto* part_avx_aligned = static_cast<uint32_t*>(aligned_alloc(32, N * sizeof(uint32_t)));
        auto* keys = static_cast<uint64_t*>(_aligned_malloc(N * sizeof(uint64_t), 32));
        auto* part_base = static_cast<uint32_t*>(_aligned_malloc(N * sizeof(uint32_t), 32));
        auto* part_avx_unaligned = static_cast<uint32_t*>(_aligned_malloc(N * sizeof(uint32_t), 32));
        auto* part_avx_aligned = static_cast<uint32_t*>(_aligned_malloc(N * sizeof(uint32_t), 32));

        std::vector<uint64_t> k(N);
        // 2. Generate Data
        // (Wrap your generate_rand here to operate on the pointer directly)
        generate_rand(N, k); // Re-generate keys for the new N
        std::copy(k.begin(), k.end(), keys);

        // 3. Benchmark Baseline
        auto stats_base = spm::benchmark([&] {
            simple_hash(keys, P, N, part_base);
        }, N, 10, 0.0);
        spm::print_table_row("Baseline", stats_base);

        // 4. Benchmark AVX Unaligned
        auto stats_avx_u = spm::benchmark([&] {
            simple_hash_avx2(keys, P, N, part_avx_unaligned); // MUST output uint32_t now!
        }, N, 10, stats_base.median_s);
        spm::print_table_row("AVX2 (Unaligned)", stats_avx_u);

        // 6. Correctness Checks
        // Always run checksum for large N to guarantee stability
        verify_checksum(part_base, part_avx_unaligned, N);
        verify_checksum(part_base, part_avx_aligned, N);

        // Only run one-to-one for small N as specified by the requirements
        if (N <= 10000) {
            verify_exact(part_base, part_avx_unaligned, N);
            verify_exact(part_base, part_avx_aligned, N);
        }

        // 7. Store results for CSV
        // results.push_back({N, P, "Baseline", stats_base}); ...

        // 8. Cleanup
        // std::free(keys);
        // std::free(part_base);
        // std::free(part_avx_unaligned);
        // std::free(part_avx_aligned);
        _aligned_free(keys);
        _aligned_free(part_base);
        _aligned_free(part_avx_unaligned);
        _aligned_free(part_avx_aligned);
    }

    // 4. Clean Main
    int main() {
        std::vector<spm::BenchResult> all_results;
        // std::vector<size_t> N_values = {10000}; //, 1000000, 50000000};
        // std::vector<int> P_values = {16}; //, 64, 256, 1024, 4096};
        std::vector<size_t> N_values = {10000, 1000000, 50000000};
        std::vector<int> P_values = {16, 64, 256, 1024, 4096};

        spm::print_table_header();
        for (size_t N : N_values) {
            for (int P : P_values) {
                run_experiment(N, P, all_results);
            }
        }
        spm::print_table_footer();

        // spm::export_csv(all_results, "sweep_results.csv");
        return 0;
    }
