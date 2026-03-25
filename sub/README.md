# SPM Module 1: Vectorization of the Partition Mapping Kernel

This directory contains the implementation and benchmarking tools for Module 1 of the SPM course project. The objective is to evaluate and optimize the performance of mapping randomly generated 64-bit keys to a specific partition ID (0 to P-1), which is a core component of partitioned hash joins.

## File Structure

- **`main.cpp`**: Contains the core logic for the different hash mapping strategies, including the plain scalar versions and the manually vectorized version using AVX2 intrinsics. It also contains the parameter sweep logic over array size (N) and partitions (P), validation checks, and distribution quality metrics (CV%).
- **`benchmark.h` / `benchmark.cpp`**: A modular performance tracking unit that measures execution median times, standard deviations, throughput, and speedup ratios. Contains logic to output formatted results and export summary tables to CSV.
- **`Makefile`**: Streamlined build system designed to cleanly compile all three requested binaries, extracting proper optimization/vectorization logs from GCC.
- **`module_1.md`**: The task specification and prompt document detailing implementation constraints and final deliverables.

## Build Instructions

Using the provided Makefile, you can build all targets simultaneously. The build process explicitly outputs auto-vectorization reports for documentation and evidence.

```bash
make clean
make
```

### Generated Executables
- **`baseline`**: Built with `-O3` but specifically configured with `-fno-tree-vectorize` to serve as a strict, non-vectorized reference.
- **`autovec`**: Built with `-O3 -ftree-vectorize`. Will generate the `autovec_report.txt` file containing the GCC auto-vectorization feedback.
- **`avx2`**: Built natively with `-O3 -mavx2 -ftree-vectorize`. **Important Note:** In this build, the explicit `simple_hash_avx2` routine comprises pure hardware intrinsics. However, compiling with `-mavx2 -ftree-vectorize` means that plain C++ fallback functions (like `simple_hash`) within this binary are also highly **auto-vectorized** by GCC utilizing 256-bit AVX2 registers. This target outputs `avx2_report.txt` confirming both manual loop implementations and compiler-assisted SIMD utilization.

## Execution and Testing

Once compiled, simply execute the generated binaries. Ensure you run the `avx2` build to test all intrinsic operations.

```bash
./avx2
```

### What the test runner does:
1. **Parameter Sweep:** Iterates over different Input Sizes $N$ ($10^4$ to $5 \times 10^7$) and Partition Values $P$ ($16, 64, \ldots, 4096$). Each configuration executes a preliminary **warmup iteration** followed by **25 measured repetitions** to guarantee stable computational timing and standard deviations.
2. **Correctness Verification:** Automatically validates the manually vectorized AVX2 kernel directly against the deterministic baseline array:
   - For small arrays ($N \leq 10000$), it guarantees mathematical soundness via an **exact element-by-element equivalence** test.
   - For massively large arrays, it switches to a highly optimized **XOR checksum algorithm** ensuring no data overlaps or data races occur without immense verification latency.
3. **Distribution Quality:** Automatically calculates the Coefficient of Variation (CV%) comparing the uniformity of the partitions bucket distributions.
4. **Performance Output:** Displays a terminal-friendly layout with medians, standard deviations, speedups and generates a final `sweep_results.csv` logging all parameters dynamically.
