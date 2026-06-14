#include "simd_dot.hpp"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace axion {

bool simd_dot_has_avx2() {
#if defined(__AVX2__)
    return true;
#else
    return false;
#endif
}

float simd_dot_f32(
    const float* a,
    const float* b,
    int64_t n
) {
#if defined(__AVX2__)
    // 8-wide FMA accumulation with a scalar tail. Two independent
    // accumulators shorten the dependency chain on the critical path.
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 b0 = _mm256_loadu_ps(b + i);
        __m256 a1 = _mm256_loadu_ps(a + i + 8);
        __m256 b1 = _mm256_loadu_ps(b + i + 8);
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
    }
    for (; i + 8 <= n; i += 8) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 b0 = _mm256_loadu_ps(b + i);
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
    }

    __m256 acc = _mm256_add_ps(acc0, acc1);

    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float result = _mm_cvtss_f32(s);

    for (; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
#else
    float result = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
#endif
}

}
