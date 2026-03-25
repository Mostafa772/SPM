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

Stats benchmark(std::function<void()> fn, size_t N, int reps, double baseline_s) {
    std::vector<double> times(reps);

    // Warm-up run (optional, but good practice)
    fn();

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

void print_stats(const char* name, const Stats& s) {
    std::printf("  %-22s  median=%8.3f ms  sd=%7.3f ms  tput=%9.3e elem/s  speedup=%.2fx\n",
                name,
                s.median_s * 1e3,
                s.stddev_s * 1e3,
                s.throughput,
                s.speedup);
}

void print_table_header() {
    std::printf("\n");
    std::printf("┌────────────────────────┬────────────┬────────────┬──────────────┬──────────┐\n");
    std::printf("│ %-22s │ %-10s │ %-10s │ %-12s │ %-8s │\n",
                "Method", "Median(ms)", "StdDev(ms)", "Throughput", "Speedup");
    std::printf("├────────────────────────┼────────────┼────────────┼──────────────┼──────────┤\n");
}

void print_table_row(const char* name, const Stats& s) {
    char tput[16];
    std::snprintf(tput, sizeof(tput), "%.2e e/s", s.throughput);
    std::printf("│ %-22s │ %10.3f │ %10.3f │ %-12s │ %8.2fx │\n",
                name,
                s.median_s * 1e3,
                s.stddev_s * 1e3,
                tput,
                s.speedup);
}

void print_table_footer() {
    std::printf("└────────────────────────┴────────────┴────────────┴──────────────┴──────────┘\n");
}

void export_csv(const std::vector<BenchResult>& results, const char* filename, const char* exec_name) {
    bool file_exists = false;
    if (std::ifstream(filename)) {
        file_exists = true;
    }

    std::ofstream f(filename, std::ios::app);
    if (!f) return;

    if (!file_exists) {
        f << "Executable,N,P,Method,Median_ms,StdDev_ms,Throughput_elem_s,Speedup\n";
    }

    std::string ename = exec_name ? exec_name : "unknown";
    size_t last_slash = ename.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        ename = ename.substr(last_slash + 1);
    }

    for (const auto& r : results) {
        f << ename << ","
          << r.N << ","
          << r.P << ","
          << r.name << ","
          << (r.stats.median_s * 1e3) << ","
          << (r.stats.stddev_s * 1e3) << ","
          << r.stats.throughput << ","
          << r.stats.speedup << "\n";
    }
}

} // namespace spm
