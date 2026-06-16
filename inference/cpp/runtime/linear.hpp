#pragma once

#include "../core/tensor.hpp"
#include "../core/scheduler.hpp"
namespace axion {

Tensor linear(
    const Tensor& input,
    const Tensor& weight,
    RuntimeMemoryScheduler* scheduler = nullptr
);

// Multiply input [M, K] by a weight stored in native GGUF layout
// [N, K] (row-major: each output row is a contiguous length-K vector),
// producing [M, N]. This is mathematically identical to
//     matmul(input, transpose(weight))
// but skips materializing the transpose and accesses both operands
// contiguously, which removes the dominant per-token transpose cost.
//
// Both operands must be contiguous, owned/buffer-backed FLOAT32
// tensors (the dequantized GGUF weights and activations always are).
Tensor linear_from_gguf(
    const Tensor& input,
    const Tensor& weight
);

}