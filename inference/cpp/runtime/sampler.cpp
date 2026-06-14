#include "sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace axion {

static int greedy_argmax(
    const float* logits,
    int vocab
) {
    int best = 0;
    float best_v = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < vocab; i++) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best   = i;
        }
    }
    return best;
}

int sample_token(
    const float* logits,
    int vocab,
    const SamplingParams& params,
    std::mt19937_64& rng
) {
    if (vocab <= 0) return -1;

    // Greedy decode when temperature is disabled.
    if (params.temperature <= 0.0f) {
        return greedy_argmax(logits, vocab);
    }

    // 1. Temperature-scaled softmax (numerically stable).
    std::vector<float> probs(vocab);
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < vocab; i++) {
        max_logit = std::max(max_logit, logits[i]);
    }

    float inv_t = 1.0f / params.temperature;
    float sum = 0.0f;
    for (int i = 0; i < vocab; i++) {
        float p = std::exp((logits[i] - max_logit) * inv_t);
        probs[i] = p;
        sum += p;
    }
    for (int i = 0; i < vocab; i++) probs[i] /= sum;

    // Indices sorted by descending probability (needed for top-k/top-p).
    std::vector<int> order(vocab);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return probs[a] > probs[b]; });

    // 2. top-k truncation: keep only the k most probable tokens.
    int keep = vocab;
    if (params.top_k > 0 && params.top_k < keep) {
        keep = params.top_k;
    }

    // 3. top-p (nucleus): keep the smallest prefix whose cumulative
    //    probability reaches p. Disabled when p <= 0 or p >= 1.
    if (params.top_p > 0.0f && params.top_p < 1.0f) {
        float cum = 0.0f;
        int nucleus = 0;
        for (int i = 0; i < keep; i++) {
            cum += probs[order[i]];
            nucleus = i + 1;
            if (cum >= params.top_p) break;
        }
        keep = std::min(keep, nucleus);
    }

    if (keep < 1) keep = 1;

    // 4. Renormalize the surviving set and sample categorically.
    float kept_sum = 0.0f;
    for (int i = 0; i < keep; i++) kept_sum += probs[order[i]];

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng) * kept_sum;

    float acc = 0.0f;
    for (int i = 0; i < keep; i++) {
        acc += probs[order[i]];
        if (r <= acc) return order[i];
    }
    return order[keep - 1];
}

int sample_last_row(
    const Tensor& logits,
    const SamplingParams& params,
    std::mt19937_64& rng
) {
    if (logits.shape.size() != 2) {
        throw std::runtime_error(
            "sample_last_row expects 2D [seq, vocab] logits");
    }

    int64_t seq   = logits.shape[0];
    int64_t vocab = logits.shape[1];
    int64_t off   = (seq - 1) * vocab;

    // Materialize the last row contiguously (logits may be strided).
    std::vector<float> row(vocab);
    for (int64_t i = 0; i < vocab; i++) {
        row[i] = logits.value(off + i);
    }

    return sample_token(row.data(),
                        static_cast<int>(vocab),
                        params, rng);
}

}
