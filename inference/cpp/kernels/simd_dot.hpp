#pragma once

#include <cstdint>

namespace axion {

// Single-precision dot product of two contiguous arrays of length n.
//
// Uses an AVX2 path when the translation unit is built with AVX2
// (-mavx2 / /arch:AVX2) and falls back to a scalar loop otherwise. The
// result is numerically equivalent to the scalar accumulation to within
// float rounding; tests assert agreement to a tight tolerance.
float simd_dot_f32(
    const float* a,
    const float* b,
    int64_t n
);

// True if this build compiled the AVX2 implementation. Exposed so
// tests/benchmarks can report which path is active.
bool simd_dot_has_avx2();

}
