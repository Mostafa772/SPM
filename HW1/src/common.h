// SPM Module 1 - Common definitions and includes
// This file contains shared types and includes used across the project

#ifndef SPM_MODULE1_COMMON_H
#define SPM_MODULE1_COMMON_H

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

// Project constants
namespace spm {

// Mix64 hash constants (from MurmurHash3 finalizer)
constexpr uint64_t MIX64_C1 = UINT64_C(0xbf58476d1ce4e5b9);
constexpr uint64_t MIX64_C2 = UINT64_C(0x94d049bb133111eb);

// FNV-1a hash constants for checksum
constexpr uint64_t FNV1A_OFFSET = UINT64_C(0xcbf29ce484222325);
constexpr uint64_t FNV1A_PRIME  = UINT64_C(0x00000100000001B3);

} // namespace spm

#endif // SPM_MODULE1_COMMON_H
