#pragma once

#include "../core/tensor.hpp"

#include <cstdint>
#include <random>
#include <vector>

namespace axion {

// Token sampling configuration. Greedy decoding is the default
// (temperature 0); positive temperature enables stochastic sampling
// with optional top-k and top-p (nucleus) truncation.
struct SamplingParams {
    float        temperature = 0.0f;  // <= 0 => greedy argmax
    int          top_k       = 0;     // 0 => disabled
    float        top_p       = 0.0f;  // <= 0 or >= 1 => disabled
    uint64_t     seed        = 0;
};

// Sample a token id from a single [vocab] logits row.
// `rng` is advanced so repeated calls in a loop stay reproducible for
// a fixed seed.
int sample_token(
    const float* logits,
    int vocab,
    const SamplingParams& params,
    std::mt19937_64& rng
);

// Convenience overload taking the last row of a [seq, vocab] tensor.
int sample_last_row(
    const Tensor& logits,
    const SamplingParams& params,
    std::mt19937_64& rng
);

}
