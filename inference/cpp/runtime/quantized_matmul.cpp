#include "quantized_matmul.hpp"

#include "../core/tensor_factory.hpp"
#include "../kernels/simd_dot.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace axion {

int64_t QuantizedWeight::payload_bytes() const {
    switch (type) {
        case QuantType::FP32: return (int64_t)f32.size() * 4;
        case QuantType::Q8_0:
        case QuantType::Q8_1: return (int64_t)q8.size();
        case QuantType::Q4_0: return (int64_t)q4.size();
    }
    return 0;
}

static float max_abs_2d(const Tensor& w) {
    float m = 0.0f;
    for (int64_t i = 0; i < w.numel(); i++) {
        float v = std::abs(w.value(i));
        if (v > m) m = v;
    }
    return m < 1e-8f ? 1e-8f : m;
}

QuantizedWeight quantize_weight(
    const Tensor& weight,
    QuantType type
) {
    if (weight.shape.size() != 2) {
        throw std::runtime_error(
            "quantize_weight expects a 2D [K, N] weight");
    }

    QuantizedWeight W;
    W.type = type;
    W.K = weight.shape[0];
    W.N = weight.shape[1];

    int64_t n = weight.numel();

    if (type == QuantType::FP32) {
        W.f32.resize(n);
        for (int64_t i = 0; i < n; i++) W.f32[i] = weight.value(i);
        return W;
    }

    if (type == QuantType::Q8_0) {
        W.scale = max_abs_2d(weight) / 127.0f;
        W.q8.resize(n);
        for (int64_t i = 0; i < n; i++) {
            int q = (int)std::lround(weight.value(i) / W.scale);
            q = std::max(-127, std::min(127, q));
            W.q8[i] = (int8_t)q;
        }
        return W;
    }

    if (type == QuantType::Q8_1) {
        // Symmetric int8 around a per-tensor bias (mean), which lets
        // asymmetric weight distributions reconstruct more accurately.
        double mean = 0.0;
        for (int64_t i = 0; i < n; i++) mean += weight.value(i);
        mean /= (double)n;
        W.bias = (float)mean;

        float m = 0.0f;
        for (int64_t i = 0; i < n; i++) {
            float v = std::abs(weight.value(i) - W.bias);
            if (v > m) m = v;
        }
        if (m < 1e-8f) m = 1e-8f;
        W.scale = m / 127.0f;

        W.q8.resize(n);
        for (int64_t i = 0; i < n; i++) {
            int q = (int)std::lround((weight.value(i) - W.bias) / W.scale);
            q = std::max(-127, std::min(127, q));
            W.q8[i] = (int8_t)q;
        }
        return W;
    }

    // Q4_0: symmetric nibble, value range [-8, 7], single scale.
    W.scale = max_abs_2d(weight) / 7.0f;
    W.q4.resize((n + 1) / 2, 0);
    for (int64_t i = 0; i < n; i += 2) {
        int q1 = (int)std::lround(weight.value(i) / W.scale);
        q1 = std::max(-8, std::min(7, q1));
        uint8_t packed = (uint8_t)((q1 + 8) & 0x0F);
        if (i + 1 < n) {
            int q2 = (int)std::lround(weight.value(i + 1) / W.scale);
            q2 = std::max(-8, std::min(7, q2));
            packed |= (uint8_t)(((q2 + 8) & 0x0F) << 4);
        }
        W.q4[i / 2] = packed;
    }
    return W;
}

Tensor quantized_matmul(
    const Tensor& input,
    const QuantizedWeight& W
) {
    if (input.shape.size() != 2) {
        throw std::runtime_error(
            "quantized_matmul expects a 2D [M, K] input");
    }

    int64_t M = input.shape[0];
    int64_t K = input.shape[1];
    int64_t N = W.N;

    if (K != W.K) {
        throw std::runtime_error(
            "quantized_matmul dimension mismatch (K)");
    }

    Tensor out = create_owned_tensor({M, N}, DType::FLOAT32);
    out.name = "quantized_matmul_output";

    const float scale = W.scale;
    const float bias  = W.bias;

    // When the input is contiguous and the weight has a single column,
    // both operands are contiguous length-K and we can use the SIMD dot
    // kernel. For general N the weight column is strided, so we keep the
    // scalar gather (unchanged behavior, no regression).
    const bool input_contig = !input.is_strided;

    #pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {

            float acc = 0.0f;

            switch (W.type) {

                case QuantType::FP32: {
                    if (input_contig && N == 1) {
                        acc = simd_dot_f32(
                            input.data() + i * K,
                            W.f32.data(),
                            K);
                    } else {
                        for (int64_t k = 0; k < K; k++) {
                            acc += input.value(i * K + k) *
                                   W.f32[k * N + j];
                        }
                    }
                    break;
                }

                case QuantType::Q8_0: {
                    for (int64_t k = 0; k < K; k++) {
                        acc += input.value(i * K + k) *
                               (float)W.q8[k * N + j];
                    }
                    acc *= scale;
                    break;
                }

                case QuantType::Q8_1: {
                    // w = q*scale + bias
                    float qacc = 0.0f;
                    float iacc = 0.0f;
                    for (int64_t k = 0; k < K; k++) {
                        float a = input.value(i * K + k);
                        qacc += a * (float)W.q8[k * N + j];
                        iacc += a;
                    }
                    acc = qacc * scale + iacc * bias;
                    break;
                }

                case QuantType::Q4_0: {
                    for (int64_t k = 0; k < K; k++) {
                        int64_t idx = k * N + j;
                        uint8_t packed = W.q4[idx >> 1];
                        int nib = (idx & 1)
                            ? ((packed >> 4) & 0x0F)
                            :  (packed       & 0x0F);
                        float w = (float)(nib - 8) * scale;
                        acc += input.value(i * K + k) * w;
                    }
                    break;
                }
            }

            out.data()[i * N + j] = acc;
        }
    }

    return out;
}

}
