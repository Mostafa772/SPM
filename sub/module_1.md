# [cite_start]SPM Modular Project: Modulo1 [cite: 1]

**Course:** SPM course a.y. [cite_start]25/26 [cite: 2]

## Project Overview
[cite_start]In this project, you will implement a partitioned hash join with duplicates, a common building block in data analytics[cite: 3, 5]. [cite_start]In an equi-join, you need to find all pairs of records r in R, s in S such that r.key=s.key even when the same key appears multiple times in either relation[cite: 6]. 

[cite_start]A standard way to parallelize this problem is to decompose it into smaller independent subproblems[cite: 7]. [cite_start]This can be done by partitioning the input dataset[cite: 8]. [cite_start]Each input record is assigned to a partition based on its key, typically using a hash function[cite: 8]. [cite_start]All records whose keys map to the same partition are grouped together[cite: 9]. [cite_start]After partitioning, you can process each partition independently, building a compact lookup structure for the R-partition and probing it with the S-partition[cite: 10].

## Module 1: Vectorization of the Partition Mapping Kernel
[cite_start]The first operation you perform in this parallelization pipeline, and one you may repeat in later phases, is mapping each 64-bit key to a partition identifier in [0, P)[cite: 11, 14]. [cite_start]This "key to partition id" mapping is a streaming kernel applied to a large number of keys[cite: 12]. [cite_start]Its performance often has an important impact on end-to-end time because it is executed over the full input size and it sits on the critical path before any per-partition work can start[cite: 13].

**Input/Output Specifications:**
* [cite_start]**Input:** An array of N keys (uint64_t, 64-bit unsigned integer) and a number of partitions P (usually a power of two)[cite: 18]. [cite_start]Input keys are generated deterministically from a seed, so that the output is reproducible[cite: 20].
* [cite_start]**Output:** An array `part_id` of length N[cite: 21]. [cite_start]For each input key `keys[i]`, `part_id[i]` in [0, P) is the partition index assigned to that key[cite: 21].
* [cite_start]**Constraints:** The mapping must be deterministic and must distribute keys reasonably across partitions for typical inputs[cite: 19].

## Tasks
* [cite_start]**Task 1 (Plain C++):** Provide one C++ implementation without intrinsics (plain version)[cite: 23]. [cite_start]Produce two binaries from the same source: one compiled with auto-vectorization disabled (baseline) and one with auto-vectorization enabled[cite: 23]. [cite_start]You must provide evidence that GCC vectorized the hot loop(s) using compiler reports[cite: 24].
* [cite_start]**Task 2 (AVX2 Intrinsics):** Provide a version using intrinsics AVX2 available on node09 (gpu-shared and gpu-exclusive partitions) and on the front-end node of the spmcluster[cite: 25]. [cite_start]It must produce identical output to your plain C++ version[cite: 26].
* [cite_start]**Task 3 (Optional - CUDA):** Provide a CUDA implementation that computes the same output array on GPU[cite: 27]. [cite_start]If you do this extra task, you must separately report kernel-only time and host-device transfer time, and you must verify correctness against the CPU reference[cite: 28].

## Correctness and Testing Requirements
* [cite_start]You must implement a verification strategy that can detect mismatches without printing the full output array (for example a checksum, or hash over the output)[cite: 43].
* [cite_start]For at least one small input size, it must be possible to perform an element-by-element comparison between implementations[cite: 44].
* [cite_start]You must benchmark some value for N and at least one N that is "large enough" to obtain stable timings (order of magnitude: tens of millions)[cite: 45].
* [cite_start]Use a repeatable methodology, for example multiple repetitions and reporting a median time and the standard deviation[cite: 47].

## Deliverables
[cite_start]Send the teacher an email with subject "SPM Modulo1" which has attached a zip file with the name `Modulo1 NameSurname.zip` by 31 March (soft deadline)[cite: 30].

**Zip File Contents:**
* [cite_start]Source code and build instructions (Makefile or explicit compile commands)[cite: 32].
* [cite_start]README containing instructions on how to run the tests and perform correctness checks[cite: 33].
* [cite_start]A short PDF report (max 4 pages)[cite: 34].

**PDF Report Requirements:**
* [cite_start]Your design choices for the mapping function and for P (and why)[cite: 35].
* [cite_start]Evidence of GCC auto-vectorization (what flags you used and what the compiler reported - a short excerpt of the GCC report)[cite: 36, 37].
* [cite_start]A brief explanation of your intrinsics strategy[cite: 38].
* [cite_start]Performance results obtained on node09 and a discussion of what limits performance (compute vs memory vs overhead)[cite: 39, 40].
* [cite_start]Execution time, throughput (elements/s), and speedup of each optimized version over the plain C++ version[cite: 46].
* [cite_start]*(Optional)* If you implement the CUDA optional part, include correctness verification and a breakdown of time components[cite: 41].

### Suggested Experiments (Not Mandatory)
* [cite_start]Sweep P to study its effect on performance and distribution[cite: 50].
* [cite_start]Sweep `key_space` (or duplicate rate) to study sensitivity to input distribution[cite: 51].

TESTS

Gemini said
Here is a structured set of tests you should implement to guarantee your project meets all the requirements outlined in the prompt document.

I've broken these down into Correctness, Distribution, and Performance tests to make them easy to track.

1. Correctness Verification Tests
These tests ensure that your logic is sound and consistent across all versions of your code.


Small Input Element-by-Element Test: Run your mapping functions on a small input size and perform a strict element-by-element comparison between the output arrays.


Large Input Checksum/Hash Test: For larger arrays where printing or looping element-by-element is too slow, compute a checksum or hash over the entire output array to detect mismatches.


AVX2 Equivalence Test: Verify that the output array produced by your AVX2 intrinsics implementation is exactly identical to the output of your plain C++ version.


CUDA Equivalence Test (Optional): If you complete the optional GPU task, verify its correctness against the CPU reference implementation.


Reproducibility Test: Generate your input keys deterministically using a seed and assert that running the program multiple times with the same seed yields the exact same output array.

2. Boundary and Distribution Tests
These tests validate that your mapping function adheres to the mathematical constraints of the assignment.


Range Validation Test: Iterate through the generated part_id array and assert that every single partition index assigned to a key falls strictly within the range [0, P).


Distribution Quality Test: Count the number of keys assigned to each partition and verify that the mapping distributes the keys reasonably across all partitions for typical inputs.

3. Performance and Benchmarking Tests
These tests ensure your performance metrics are stable, accurate, and properly formatted for your final report.


Large Scale Stability Test: Benchmark your implementations using at least one input size N that is "large enough" (in the tens of millions) to guarantee stable timings.


Statistical Variance Test: Run your benchmarks using multiple repetitions to capture the median time and standard deviation, ensuring your methodology is repeatable.


Metrics Calculation Test: For each optimized version, calculate and log the execution time, the throughput in elements/s, and the speedup factor compared to the plain C++ version.


CUDA Timing Breakdown (Optional): If testing the CUDA implementation, ensure your testing framework separately captures and reports the kernel-only execution time and the host-device transfer time.