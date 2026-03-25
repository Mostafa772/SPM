// SPM Module 1 - Benchmarking and Output
// Performance measurement, statistics, and result formatting

#ifndef SPM_MODULE1_BENCHMARK_H
#define SPM_MODULE1_BENCHMARK_H

// #include "common.h"
#include <vector>
#include <functional>
#include <string>

namespace spm {

// ─────────────────────────────────────────────────────────────────
// Statistics Structure
// ─────────────────────────────────────────────────────────────────
struct Stats {
    double median_s;    // Median time in seconds
    double stddev_s;    // Standard deviation in seconds
    double throughput;  // Elements per second
    double speedup;     // Speedup vs baseline

};

// ─────────────────────────────────────────────────────────────────
// Benchmark Result Structure
// ─────────────────────────────────────────────────────────────────
struct BenchResult {
    std::string name;
    size_t N;
    int P;
    Stats stats;
};

// ─────────────────────────────────────────────────────────────────
// Benchmarking
// ─────────────────────────────────────────────────────────────────
// Runs a function multiple times and computes statistics
Stats benchmark(std::function<void()> fn, size_t N, int reps,
                double baseline_s = 0.0);

// ─────────────────────────────────────────────────────────────────
// Output Formatting
// ─────────────────────────────────────────────────────────────────
// Print stats in simple format
void print_stats(const char* name, const Stats& s);

// Print formatted table header
void print_table_header();

// Print formatted table row
void print_table_row(const char* name, const Stats& s);

// Print formatted table footer
void print_table_footer();

// Export results to CSV file (Appends if exists, logs Executable name)
void export_csv(const std::vector<BenchResult>& results, const char* filename, const char* exec_name = "unknown");

} // namespace spm

#endif // SPM_MODULE1_BENCHMARK_H
