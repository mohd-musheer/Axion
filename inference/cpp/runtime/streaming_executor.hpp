#pragma once

#include "../gguf/gguf.hpp"
#include "../core/tensor.hpp"
#include "tensor_evictor.hpp"
#include "mha.hpp"
#include <string>
#include <vector>
#include "../core/scheduler.hpp"

namespace axion {

// Minimal per-run configuration. Defaults are safe for a single-head
// full-dimension attention pass; later MRs supply real model
// hyperparameters parsed from GGUF metadata.
struct LayerConfig {
    int      num_heads  = 1;     // query heads
    int      n_kv_heads = 1;     // key/value heads (GQA)
    int      head_dim   = 0;     // per-head dim (0 => derive hidden/num_heads)
    float    rope_theta = 10000.0f;
    float    eps        = 1e-6f;
    RopeType rope_type  = RopeType::NEOX;   // real GGUF models use NEOX
};

class StreamingExecutor {

public:

    StreamingExecutor(
        GGUFLoader* loader
    );

    // Run num_layers transformer blocks over the input hidden state.
    // Weights are streamed in per layer and evicted before the next.
    Tensor forward(
        const Tensor& input,
        int num_layers,
        const LayerConfig& config = LayerConfig{}
    );

private:

    GGUFLoader* loader;
    TensorEvictor evictor;
    LayerConfig config;

    Tensor execute_layer(
        const Tensor& hidden,
        int layer_idx
    );

    // ----------------------------------------------------------------
    // Incremental decode (MR13.1): per-layer K/V caches reused across
    // tokens so each new position is O(context), not O(context^2).
    // ----------------------------------------------------------------
public:
    // Reset the decode session for a model with n_layers layers.
    void begin_decode(int n_layers, const LayerConfig& cfg);

    // Process one new position [1, hidden] through all layers, updating
    // the K/V caches; returns the layer-stack output [1, hidden].
    Tensor decode_step(const Tensor& hidden_row);

    // Print the env-gated (AXION_PROFILE=1) generation summary and
    // whole-run phase totals. No-op when profiling is disabled. Call
    // once after the generation loop completes.
    void print_profile_summary(int tokens_generated, double total_seconds);

private:
    struct DecodeSession {
        bool active = false;
        int  position = 0;
        std::vector<Tensor> k_cache;   // per layer: [ctx, n_kv*hd]
        std::vector<Tensor> v_cache;   // per layer: [ctx, n_kv*hd]
    };
    DecodeSession session;

    // One layer of the incremental path for the current position.
    Tensor decode_layer(
        const Tensor& hidden_row,
        int layer_idx,
        int position
    );

    // Scaled dot-product self-attention over the full sequence with a
    // causal mask. Q/K/V are [seq, dim]; returns [seq, dim].
    Tensor self_attention(
        const Tensor& q,
        const Tensor& k,
        const Tensor& v
    );

    void unload_tensor(
        Tensor& t
    );
};

}
