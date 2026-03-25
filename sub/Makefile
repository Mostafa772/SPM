# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I.

# Source files
SRCS = main.cpp benchmark.cpp
HDRS = benchmark.h

# Build targets
TARGET_BASELINE = baseline
TARGET_AUTOVEC  = autovec
TARGET_AVX2     = avx2

# The targets for module 1
all: $(TARGET_BASELINE) $(TARGET_AUTOVEC) $(TARGET_AVX2)

# Task 1: Plain C++ (auto-vectorization disabled)
# Using -O3 but explicitly disabling vectorizer
$(TARGET_BASELINE): $(SRCS) $(HDRS)
	$(CXX) $(CXXFLAGS) -O3 -fno-tree-vectorize $(SRCS) -o $@

# Task 1: Plain C++ (auto-vectorization enabled)
# Using -O3 which enables vectorization, and we output the compiler report
$(TARGET_AUTOVEC): $(SRCS) $(HDRS)
	$(CXX) $(CXXFLAGS) -O3 -ftree-vectorize -fopt-info-vec-optimized=autovec_report.txt $(SRCS) -o $@

# Task 2: AVX2 Intrinsics
# Using -mavx2 to enable architecture-specific instructions and macros
$(TARGET_AVX2): $(SRCS) $(HDRS)
	$(CXX) $(CXXFLAGS) -O3 -mavx2 -ftree-vectorize -fopt-info-vec-optimized=avx2_report.txt $(SRCS) -o $@

clean:
	rm -f $(TARGET_BASELINE) $(TARGET_AUTOVEC) $(TARGET_AVX2) *.txt

.PHONY: all clean
