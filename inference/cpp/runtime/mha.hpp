#pragma once

#include "../core/tensor.hpp"

#include <vector>

namespace axion {

// RoPE rotation convention.
//   NORM: rotate interleaved adjacent pairs (2i, 2i+1), freq exp
//         d/head_dim. Kept for backward compatibility and the original
//         Phase-14 tests.
//   NEOX: rotate split-half pairs (i, i + head_dim/2), freq exp
//         2i/head_dim. This is what llama.cpp uses for LLaMA / TinyLlama
//         / Mistral / Qwen2 GGUF conversions; default for real models.
enum class RopeType {
    NORM,
    NEOX
};

struct MHAConfig {
    int      n_head     = 1;   // number of query heads
    int      n_kv_head  = 1;   // number of key/value heads (GQA; <= n_head)
    int      head_dim   = 0;   // per-head dimension
    float    rope_theta = 10000.0f;
    RopeType rope_type  = RopeType::NEOX;
};

// Apply rotary position embeddings in place to a [seq, n_heads*head_dim]
// tensor. Each row's position is its row index. Rotation convention is
// selected by `type`. Used for Q and K.
void rope_apply_rows(
    Tensor& x,
    int n_heads,
    int head_dim,
    float theta,
    RopeType type = RopeType::NEOX
);

// Extract head h (0-based) from a [seq, n_heads*head_dim] tensor into a
// contiguous [seq, head_dim] tensor.
Tensor split_head(
    const Tensor& x,
    int n_heads,
    int head_dim,
    int h
);

// Concatenate per-head [seq, head_dim] tensors into [seq, H*head_dim].
Tensor merge_heads(
    const std::vector<Tensor>& heads
);

// Multi-head causal self-attention with GQA.
//   q: [seq, n_head    * head_dim]
//   k: [seq, n_kv_head * head_dim]
//   v: [seq, n_kv_head * head_dim]
// returns context [seq, n_head * head_dim].
Tensor mha_attention(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const MHAConfig& cfg
);

// RoPE for a single new row [1, n_heads*head_dim] at an explicit
// position (used by the incremental decode path).
void rope_apply_row_at(
    Tensor& x,
    int n_heads,
    int head_dim,
    float theta,
    int position,
    RopeType type = RopeType::NEOX
);

// Incremental attention: one query row attends over a cached context.
//   q_row:   [1,   n_head    * head_dim]
//   k_cache: [ctx, n_kv_head * head_dim]
//   v_cache: [ctx, n_kv_head * head_dim]
// returns context [1, n_head * head_dim]. Equivalent to the last row of
// mha_attention when the cache holds the full sequence's K/V.
Tensor mha_attention_incremental(
    const Tensor& q_row,
    const Tensor& k_cache,
    const Tensor& v_cache,
    const MHAConfig& cfg
);

}
