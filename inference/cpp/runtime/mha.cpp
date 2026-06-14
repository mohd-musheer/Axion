#include "mha.hpp"

#include "../core/tensor_factory.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace axion {

// Rotate one head's vector in place. `base` is the element offset of the
// head within the row. Both conventions use the same per-pair rotation
// math; they differ in which two lanes form a pair and in the frequency
// exponent.
static void rope_rotate_head(
    Tensor& x,
    int64_t base,
    int head_dim,
    float theta,
    float position,
    RopeType type
) {
    if (type == RopeType::NORM) {
        for (int d = 0; d < head_dim; d += 2) {
            float freq  =
                1.0f / std::pow(theta, (float)d / (float)head_dim);
            float angle = position * freq;
            float c = std::cos(angle);
            float s = std::sin(angle);
            float a = x.value(base + d);
            float b = x.value(base + d + 1);
            x.data()[base + d]     = a * c - b * s;
            x.data()[base + d + 1] = a * s + b * c;
        }
    } else {
        // NEOX: pair lane i with lane i+half; freq exponent 2i/head_dim.
        int half = head_dim / 2;
        for (int i = 0; i < half; i++) {
            float freq  =
                1.0f / std::pow(theta, (float)(2 * i) / (float)head_dim);
            float angle = position * freq;
            float c = std::cos(angle);
            float s = std::sin(angle);
            float a = x.value(base + i);
            float b = x.value(base + i + half);
            x.data()[base + i]        = a * c - b * s;
            x.data()[base + i + half] = a * s + b * c;
        }
    }
}

void rope_apply_rows(
    Tensor& x,
    int n_heads,
    int head_dim,
    float theta,
    RopeType type
) {
    if (x.shape.size() != 2) {
        throw std::runtime_error("rope_apply_rows expects 2D [seq, H*hd]");
    }
    if (head_dim % 2 != 0) {
        throw std::runtime_error("head_dim must be even for RoPE");
    }

    int64_t seq   = x.shape[0];
    int64_t width = x.shape[1];
    if (width != (int64_t)n_heads * head_dim) {
        throw std::runtime_error("rope width != n_heads*head_dim");
    }

    for (int64_t pos = 0; pos < seq; pos++) {
        for (int h = 0; h < n_heads; h++) {
            int64_t base = pos * width + (int64_t)h * head_dim;
            rope_rotate_head(x, base, head_dim, theta, (float)pos, type);
        }
    }
}

Tensor split_head(
    const Tensor& x,
    int n_heads,
    int head_dim,
    int h
) {
    int64_t seq   = x.shape[0];
    int64_t width = x.shape[1];
    if (width != (int64_t)n_heads * head_dim) {
        throw std::runtime_error("split_head width mismatch");
    }

    Tensor out = create_owned_tensor({seq, head_dim}, DType::FLOAT32);
    out.name = "head";
    for (int64_t s = 0; s < seq; s++) {
        int64_t src = s * width + (int64_t)h * head_dim;
        for (int d = 0; d < head_dim; d++) {
            out.data()[s * head_dim + d] = x.value(src + d);
        }
    }
    return out;
}

Tensor merge_heads(
    const std::vector<Tensor>& heads
) {
    if (heads.empty()) {
        throw std::runtime_error("merge_heads: no heads");
    }
    int64_t seq      = heads[0].shape[0];
    int64_t head_dim = heads[0].shape[1];
    int64_t H        = (int64_t)heads.size();

    Tensor out = create_owned_tensor({seq, H * head_dim}, DType::FLOAT32);
    out.name = "merged_heads";
    for (int64_t h = 0; h < H; h++) {
        for (int64_t s = 0; s < seq; s++) {
            for (int64_t d = 0; d < head_dim; d++) {
                out.data()[s * (H * head_dim) + h * head_dim + d] =
                    heads[h].value(s * head_dim + d);
            }
        }
    }
    return out;
}

// Single-head causal scaled dot-product attention over [seq, hd].
static Tensor sdpa_causal(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    int head_dim
) {
    int64_t seq = q.shape[0];
    float scale = 1.0f / std::sqrt((float)head_dim);

    Tensor out = create_owned_tensor({seq, head_dim}, DType::FLOAT32);

    for (int64_t i = 0; i < seq; i++) {
        std::vector<float> scores(i + 1);
        float maxv = -std::numeric_limits<float>::infinity();
        for (int64_t j = 0; j <= i; j++) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q.value(i * head_dim + d) *
                       k.value(j * head_dim + d);
            }
            dot *= scale;
            scores[j] = dot;
            if (dot > maxv) maxv = dot;
        }
        float sum = 0.0f;
        for (int64_t j = 0; j <= i; j++) {
            scores[j] = std::exp(scores[j] - maxv);
            sum += scores[j];
        }
        for (int d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (int64_t j = 0; j <= i; j++) {
                acc += scores[j] * v.value(j * head_dim + d);
            }
            out.data()[i * head_dim + d] = acc / sum;
        }
    }
    return out;
}

Tensor mha_attention(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const MHAConfig& cfg
) {
    if (cfg.n_head <= 0 || cfg.n_kv_head <= 0 || cfg.head_dim <= 0) {
        throw std::runtime_error("mha_attention: invalid config");
    }
    if (cfg.n_head % cfg.n_kv_head != 0) {
        throw std::runtime_error(
            "mha_attention: n_head must be a multiple of n_kv_head");
    }

    int group = cfg.n_head / cfg.n_kv_head;   // query heads per kv head

    std::vector<Tensor> contexts;
    contexts.reserve(cfg.n_head);

    for (int h = 0; h < cfg.n_head; h++) {
        int kv = h / group;   // GQA: which kv head this query head uses

        Tensor qh = split_head(q, cfg.n_head,    cfg.head_dim, h);
        Tensor kh = split_head(k, cfg.n_kv_head, cfg.head_dim, kv);
        Tensor vh = split_head(v, cfg.n_kv_head, cfg.head_dim, kv);

        contexts.push_back(
            sdpa_causal(qh, kh, vh, cfg.head_dim));
    }

    return merge_heads(contexts);
}

void rope_apply_row_at(
    Tensor& x,
    int n_heads,
    int head_dim,
    float theta,
    int position,
    RopeType type
) {
    if (x.shape.size() != 2 || x.shape[0] != 1) {
        throw std::runtime_error(
            "rope_apply_row_at expects a single row [1, H*hd]");
    }
    if (head_dim % 2 != 0) {
        throw std::runtime_error("head_dim must be even for RoPE");
    }

    int64_t width = x.shape[1];
    if (width != (int64_t)n_heads * head_dim) {
        throw std::runtime_error("rope row width != n_heads*head_dim");
    }

    for (int h = 0; h < n_heads; h++) {
        int64_t base = (int64_t)h * head_dim;
        rope_rotate_head(x, base, head_dim, theta, (float)position, type);
    }
}

Tensor mha_attention_incremental(
    const Tensor& q_row,
    const Tensor& k_cache,
    const Tensor& v_cache,
    const MHAConfig& cfg
) {
    if (q_row.shape.size() != 2 || q_row.shape[0] != 1) {
        throw std::runtime_error(
            "mha_attention_incremental: q_row must be [1, n_head*hd]");
    }
    if (cfg.n_head % cfg.n_kv_head != 0) {
        throw std::runtime_error(
            "mha_attention_incremental: n_head % n_kv_head != 0");
    }

    int64_t ctx = k_cache.shape[0];
    int group   = cfg.n_head / cfg.n_kv_head;
    int hd      = cfg.head_dim;
    float scale = 1.0f / std::sqrt((float)hd);

    Tensor out =
        create_owned_tensor({1, (int64_t)cfg.n_head * hd}, DType::FLOAT32);
    out.name = "incremental_context";

    int64_t kv_width = (int64_t)cfg.n_kv_head * hd;

    for (int h = 0; h < cfg.n_head; h++) {
        int kv = h / group;
        int64_t q_base = (int64_t)h * hd;

        std::vector<float> scores(ctx);
        float maxv = -std::numeric_limits<float>::infinity();
        for (int64_t j = 0; j < ctx; j++) {
            int64_t k_base = j * kv_width + (int64_t)kv * hd;
            float dot = 0.0f;
            for (int d = 0; d < hd; d++) {
                dot += q_row.value(q_base + d) * k_cache.value(k_base + d);
            }
            dot *= scale;
            scores[j] = dot;
            if (dot > maxv) maxv = dot;
        }
        float sum = 0.0f;
        for (int64_t j = 0; j < ctx; j++) {
            scores[j] = std::exp(scores[j] - maxv);
            sum += scores[j];
        }
        for (int d = 0; d < hd; d++) {
            float acc = 0.0f;
            for (int64_t j = 0; j < ctx; j++) {
                int64_t v_base = j * kv_width + (int64_t)kv * hd;
                acc += scores[j] * v_cache.value(v_base + d);
            }
            out.data()[q_base + d] = acc / sum;
        }
    }

    return out;
}

}
