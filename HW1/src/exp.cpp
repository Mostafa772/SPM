//
// Created by Mostafa on 3/17/2026.
//
#include <iostream>
#include <vector>
#include <random>
#include "benchmark.h"


// SPM Module 1 - Benchmarking Implementations

#include "benchmark.h"
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdio>
#include <fstream>

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
    if (!f) {
        std::fprintf(stderr, "Failed to open %s for writing\n", filename);
        return;
    }

    f << "Method,Median_ms,StdDev_ms,Throughput_elem_per_s,Speedup\n";
    for (const auto& r : results) {
        f << r.name << ","
          << (r.stats.median_s * 1e3) << ","
          << (r.stats.stddev_s * 1e3) << ","
          << r.stats.throughput << ","
          << r.stats.speedup << "\n";
    }

    f.close();
    std::printf("  Results exported to %s\n", filename);
}

} // namespace spm


constexpr uint64_t MIX64_C1 = UINT64_C(0xbf58476d1ce4e5b9);
constexpr uint64_t MIX64_C2 = UINT64_C(0x94d049bb133111eb);
const int seed = 42;
auto N_small = 10000000;
auto P = 256;

void print_vec(const std::vector<uint64_t>& __restrict vec)
{
    for (auto& v : vec) std::cout << v << " ";
    std::cout << std::endl;
}


auto generate_rand(const int size_N, std::vector<uint64_t>& keys) ->auto
{
    std::mt19937_64 rng(seed);
    for (int i = 0; i < size_N; ++i)
        keys[i] = rng();
    return keys;
}

void simple_hash(const std::vector<uint64_t>& __restrict__ keys, const int P, const int N, std::vector<uint64_t>& __restrict__ part)
{
    for (size_t i = 0; i < N; ++i)
    {
        part[i] = keys[i] & (P-1);
    }
}

// Pass raw pointers with __restrict to guarantee no aliasing
void map_scalar(const uint64_t* __restrict keys,
                uint64_t* __restrict part_id,
                size_t N, uint64_t mask)
{
    // The compiler can now safely vectorize this hot loop
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

void map_fibo(const uint64_t* __restrict keys,
              uint64_t* __restrict part_id,
              size_t N, uint32_t shift)
{
    for (size_t i = 0; i < N; ++i)
    {
        part_id[i] = (keys[i] * 0x9E3779B97F4A7C15ULL) >> shift;
    }
}


int main()
{
    std::vector<uint64_t> keys(N_small);
    generate_rand(N_small, keys);
    // print_vec(keys);
    std::vector<uint64_t> part(N_small);
    spm::Stats stats_base = spm::benchmark([&] { simple_hash(keys, P, N_small, part); }
        , N_small, 10);
    spm::Stats stats_mix = spm::benchmark([&] { map_scalar(keys.data(), part.data(), N_small, P-1); }
    , N_small, 10);
    const auto m = static_cast<uint64_t>(std::log2(P));
    const auto shift = P - m;
    spm::Stats stats_fibo = spm::benchmark([&] { map_fibo(keys.data(), part.data(), N_small, shift); }
    , N_small, 10);
    spm::print_stats("basic", stats_base);
    spm::print_stats("mix", stats_mix);
    spm::print_stats("fibo", stats_fibo);
    return 0;
}
