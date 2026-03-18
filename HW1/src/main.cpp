// SPM Module 1 - Main Program
// Partition Mapping Kernel: Plain C++ + AVX2 + AVX-512 + Multi-threading

#include "kernels.h"
#include "utils.h"
#include "benchmark.h"
#include <iostream>
#include <vector>
#include <thread>

using namespace spm;

int main(int argc, char* argv[])
{
    // ─────────────────────────────────────────────────────────────────
    // Configuration and Parameter Parsing
    // ─────────────────────────────────────────────────────────────────
    size_t   N_small = 1024;
    size_t   N_large = 32'000'000;  // "Tens of millions" as per requirements
    uint64_t P       = 256;          // Must be power of two
    uint64_t seed    = 42;
    int      reps    = 11;
    int      threads = std::thread::hardware_concurrency();
    if (threads == 0) threads = 4;

    // Parse command line: ./program [N_large] [P] [seed] [reps] [threads]
    if (argc > 1) N_large = static_cast<size_t>(std::atoll(argv[1]));
    if (argc > 2) P       = static_cast<uint64_t>(std::atoll(argv[2]));
    if (argc > 3) seed    = static_cast<uint64_t>(std::atoll(argv[3]));
    if (argc > 4) reps    = std::atoi(argv[4]);
    if (argc > 5) threads = std::atoi(argv[5]);

    // Validate P is power of two
    if ((P & (P - 1)) != 0) {
        std::fprintf(stderr, "Error: P must be a power of two.\n");
        return 1;
    }
    const uint64_t mask = P - 1;

    std::printf("=== SPM Module 1: Partition Mapping Kernel ===\n");
    std::printf("N_large=%zu  P=%llu  seed=%llu  reps=%d  threads=%d\n\n",
                N_large, (unsigned long long)P,
                (unsigned long long)seed, reps, threads);

    // ─────────────────────────────────────────────────────────────────
    // Correctness Tests (Small N)
    // ─────────────────────────────────────────────────────────────────
    std::printf("--- Correctness Tests (N=%zu) ---\n", N_small);
    {
        std::vector<uint64_t> keys(N_small);
        generate_keys(keys.data(), N_small, seed);

        std::vector<uint32_t> ref(N_small), avx(N_small);

        // Run scalar reference
        map_scalar(keys.data(), ref.data(), N_small, mask);

        // Test 1: Range validation
        std::printf("  Range valid (scalar):  %s\n",
                    range_valid(ref.data(), N_small, P) ? "PASS" : "FAIL");

        // Test 2: Distribution quality
        distribution_stats(ref.data(), N_small, P);

        // Test 3: Reproducibility
        {
            std::vector<uint64_t> keys2(N_small);
            std::vector<uint32_t> ref2(N_small);
            generate_keys(keys2.data(), N_small, seed);
            map_scalar(keys2.data(), ref2.data(), N_small, mask);
            bool repro = element_wise_equal(ref.data(), ref2.data(), N_small);
            std::printf("  Reproducibility:       %s\n", repro ? "PASS" : "FAIL");
        }

        // Test 4: AVX2 equivalence (element-by-element)
#if defined(__AVX2__)
        map_avx2(keys.data(), avx.data(), N_small, mask);
        bool avx_ok = element_wise_equal(ref.data(), avx.data(), N_small);
        std::printf("  AVX2 == scalar:        %s\n", avx_ok ? "PASS" : "FAIL");
#else
        std::printf("  AVX2 == scalar:        SKIP (not compiled with AVX2)\n");
#endif
    }

    // ─────────────────────────────────────────────────────────────────
    // Checksum Verification (Large N)
    // ─────────────────────────────────────────────────────────────────
    std::printf("\n--- Checksum Verification (N=%zu) ---\n", N_large);

    std::vector<uint64_t> keys_large(N_large);
    generate_keys(keys_large.data(), N_large, seed);

    std::vector<uint32_t> ref_large(N_large);
    map_scalar(keys_large.data(), ref_large.data(), N_large, mask);
    uint64_t cs_ref = checksum(ref_large.data(), N_large);

    std::printf("  Scalar checksum: 0x%016llx\n",
                (unsigned long long)cs_ref);
    std::printf("  Range valid (scalar): %s\n",
                range_valid(ref_large.data(), N_large, P) ? "PASS" : "FAIL");

#if defined(__AVX2__)
    {
        std::vector<uint32_t> avx_large(N_large);
        map_avx2(keys_large.data(), avx_large.data(), N_large, mask);
        uint64_t cs_avx = checksum(avx_large.data(), N_large);
        std::printf("  AVX2   checksum: 0x%016llx  %s\n",
                    (unsigned long long)cs_avx,
                    (cs_avx == cs_ref) ? "MATCH" : "MISMATCH");
    }
#endif

#if defined(__AVX512F__)
    {
        std::vector<uint32_t> avx512_large(N_large);
        map_avx512(keys_large.data(), avx512_large.data(), N_large, mask);
        uint64_t cs_avx512 = checksum(avx512_large.data(), N_large);
        std::printf("  AVX512 checksum: 0x%016llx  %s\n",
                    (unsigned long long)cs_avx512,
                    (cs_avx512 == cs_ref) ? "MATCH" : "MISMATCH");
    }
#endif

    // ─────────────────────────────────────────────────────────────────
    // Performance Benchmarks
    // ─────────────────────────────────────────────────────────────────
    std::printf("\n--- Benchmarks (N=%zu, reps=%d) ---\n", N_large, reps);

    std::vector<uint32_t> out(N_large);
    std::vector<BenchResult> results;

    // Baseline: Scalar version
    Stats base = benchmark([&]{ map_scalar(keys_large.data(), out.data(), N_large, mask); },
                           N_large, reps);
    print_stats("Scalar (baseline)", base);
    results.push_back({"Scalar", base});

#if defined(__AVX2__)
    // Task 2: AVX2 version
    Stats avx_s = benchmark([&]{ map_avx2(keys_large.data(), out.data(), N_large, mask); },
                            N_large, reps, base.median_s);
    print_stats("AVX2", avx_s);
    results.push_back({"AVX2", avx_s});
#endif

#if defined(__AVX512F__)
    // Bonus: AVX-512 version
    Stats avx512_s = benchmark([&]{ map_avx512(keys_large.data(), out.data(), N_large, mask); },
                               N_large, reps, base.median_s);
    print_stats("AVX-512", avx512_s);
    results.push_back({"AVX-512", avx512_s});
#endif

    // Bonus: Multi-threaded scalar
    Stats scalar_mt_s = benchmark([&]{ map_scalar_mt(keys_large.data(), out.data(), N_large, mask, threads); },
                                  N_large, reps, base.median_s);
    print_stats("Scalar MT", scalar_mt_s);
    results.push_back({"Scalar_MT", scalar_mt_s});

#if defined(__AVX2__)
    // Bonus: Multi-threaded AVX2
    Stats avx2_mt_s = benchmark([&]{ map_avx2_mt(keys_large.data(), out.data(), N_large, mask, threads); },
                                N_large, reps, base.median_s);
    print_stats("AVX2 MT", avx2_mt_s);
    results.push_back({"AVX2_MT", avx2_mt_s});
#endif

#if defined(__AVX512F__)
    // Bonus: Multi-threaded AVX-512
    Stats avx512_mt_s = benchmark([&]{ map_avx512_mt(keys_large.data(), out.data(), N_large, mask, threads); },
                                  N_large, reps, base.median_s);
    print_stats("AVX-512 MT", avx512_mt_s);
    results.push_back({"AVX512_MT", avx512_mt_s});
#endif

    // ─────────────────────────────────────────────────────────────────
    // Formatted Output
    // ─────────────────────────────────────────────────────────────────
    print_table_header();
    for (const auto& r : results) {
        print_table_row(r.name.c_str(), r.stats);
    }
    print_table_footer();

    // Export to CSV
    export_csv(results, "benchmark_results.csv");

    std::printf("\nDone.\n");
    return 0;
}
