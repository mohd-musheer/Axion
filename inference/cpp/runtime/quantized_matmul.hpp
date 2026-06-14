#pragma once

#include "../core/tensor.hpp"

#include <cstdint>
#include <vector>

namespace axion {

// Runtime weight quantization variants. These are Axion's per-tensor
// runtime formats (single scale), distinct from the GGUF on-disk block
// formats decoded by the loader. They exist so the executor can hold a
// weight in a compact form and multiply against it directly.
enum class QuantType {
    FP32,
    Q4_0,   // 4-bit, symmetric, single scale (nibble (v-8)*scale)
    Q8_0,   // 8-bit, symmetric, single scale
    Q8_1    // 8-bit, symmetric, single scale + per-tensor bias
};

// A weight matrix [K, N] stored in a chosen quantized form.
struct QuantizedWeight {
    QuantType            type = QuantType::FP32;
    int64_t              K = 0;
    int64_t              N = 0;
    float                scale = 1.0f;
    float                bias  = 0.0f;   // Q8_1 only
    std::vector<float>   f32;            // FP32 path
    std::vector<int8_t>  q8;             // Q8_0 / Q8_1 path
    std::vector<uint8_t> q4;             // Q4_0 path (two values per byte)

    // Resident payload size in bytes for the quantized representation.
    int64_t payload_bytes() const;
};

// Quantize a 2D [K, N] weight tensor into the requested form, reusing
// the existing q4/q8 packers.
QuantizedWeight quantize_weight(
    const Tensor& weight,
    QuantType type
);

// out[M, N] = input[M, K] . W[K, N], dispatching on W.type. The weight
// is never expanded to a full FP32 matrix for the quantized paths.
Tensor quantized_matmul(
    const Tensor& input,
    const QuantizedWeight& W
);

}
