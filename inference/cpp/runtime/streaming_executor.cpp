#include "streaming_executor.hpp"
#include "../kernels/blas.hpp"
#include "../runtime/linear.hpp"
#include "../runtime/residual.hpp"
// transpose.hpp retained for the legacy full-sequence forward() path.
#include "../kernels/rmsnorm.hpp"
#include "../kernels/transpose.hpp"
#include "../kernels/attention.hpp"
#include "../kernels/softmax.hpp"
#include "../kernels/silu.hpp"
#include "../kernels/elementwise.hpp"
#include "../core/tensor_factory.hpp"
#include "../core/profile.hpp"
#include "mha.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <cstdio>

namespace axion {

StreamingExecutor::StreamingExecutor(
    GGUFLoader* loader
) : loader(loader) {}

Tensor StreamingExecutor::forward(
    const Tensor& input,
    int num_layers,
    const LayerConfig& cfg
) {

    config = cfg;

    Tensor hidden = input;

    for (int i = 0; i < num_layers; i++) {

        std::cout
            << "Streaming Layer "
            << i
            << std::endl;

        hidden =
            execute_layer(
                hidden,
                i
            );
    }

    return hidden;
}

// --------------------------------------------------------------------
// Scaled dot-product self-attention (causal) over the full sequence.
//
//   scores = (Q . K^T) / sqrt(dim)        [seq, seq]
//   apply causal mask (upper triangle -> -inf)
//   weights = softmax(scores) per row
//   out = weights . V                     [seq, dim]
//
// Q/K/V are [seq, dim]. This is a correct full-attention reference;
// multi-head splitting is layered on in a later MR via config.num_heads.
// --------------------------------------------------------------------
Tensor StreamingExecutor::self_attention(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v
) {

    if (q.shape.size() != 2 ||
        k.shape.size() != 2 ||
        v.shape.size() != 2) {

        throw std::runtime_error(
            "self_attention expects 2D Q/K/V"
        );
    }

    int64_t seq = q.shape[0];
    int64_t dim = q.shape[1];

    // scores = Q . K^T  -> [seq, seq], already scaled by 1/sqrt(dim)
    Tensor scores =
        attention_scores(q, k);

    // Causal mask: position i may not attend to j > i.
    for (int64_t i = 0; i < seq; i++) {
        for (int64_t j = i + 1; j < seq; j++) {
            scores.data()[i * seq + j] =
                -std::numeric_limits<float>::infinity();
        }
    }

    // Row-wise softmax over the key dimension.
    Tensor weights =
        softmax(scores);

    // out = weights . V  -> [seq, dim]
    Tensor out =
        matmul(weights, v);

    out.name = "attention_context";

    (void)dim;
    return out;
}

Tensor StreamingExecutor::execute_layer(
    const Tensor& hidden,
    int layer_idx
) {

    // --------------------------------
    // LOAD ONLY CURRENT LAYER
    // --------------------------------

    std::string prefix =
        "blk." +
        std::to_string(layer_idx);

    Tensor attn_norm =
        loader->load_tensor(prefix + ".attn_norm.weight");

    Tensor ffn_norm =
        loader->load_tensor(prefix + ".ffn_norm.weight");

    Tensor q_weight =
        loader->load_tensor(prefix + ".attn_q.weight");

    Tensor k_weight =
        loader->load_tensor(prefix + ".attn_k.weight");

    Tensor v_weight =
        loader->load_tensor(prefix + ".attn_v.weight");

    Tensor out_weight =
        loader->load_tensor(prefix + ".attn_output.weight");

    Tensor gate_weight =
        loader->load_tensor(prefix + ".ffn_gate.weight");

    Tensor up_weight =
        loader->load_tensor(prefix + ".ffn_up.weight");

    Tensor down_weight =
        loader->load_tensor(prefix + ".ffn_down.weight");

    // Register every weight for end-of-layer eviction so peak memory
    // never exceeds a single layer's resident weights.
    evictor.register_tensor(&attn_norm);
    evictor.register_tensor(&ffn_norm);
    evictor.register_tensor(&q_weight);
    evictor.register_tensor(&k_weight);
    evictor.register_tensor(&v_weight);
    evictor.register_tensor(&out_weight);
    evictor.register_tensor(&gate_weight);
    evictor.register_tensor(&up_weight);
    evictor.register_tensor(&down_weight);

    // --------------------------------
    // ATTENTION SUB-BLOCK
    // --------------------------------

    // GGUF linear weights are [out, in]; matmul wants [in, out].
    Tensor q_w_t = transpose(q_weight);
    Tensor k_w_t = transpose(k_weight);
    Tensor v_w_t = transpose(v_weight);
    Tensor o_w_t = transpose(out_weight);

    Tensor normed =
        rmsnorm(hidden, attn_norm, config.eps);

    Tensor q = linear(normed, q_w_t);
    Tensor k = linear(normed, k_w_t);
    Tensor v = linear(normed, v_w_t);

    // Multi-head + GQA + RoPE.
    int n_head    = config.num_heads > 0 ? config.num_heads : 1;
    int n_kv_head = config.n_kv_heads > 0 ? config.n_kv_heads : n_head;
    int head_dim  = config.head_dim > 0
        ? config.head_dim
        : (int)(q.shape[1] / n_head);

    MHAConfig mcfg;
    mcfg.n_head     = n_head;
    mcfg.n_kv_head  = n_kv_head;
    mcfg.head_dim   = head_dim;
    mcfg.rope_theta = config.rope_theta;
    mcfg.rope_type  = config.rope_type;

    // RoPE on Q and K only (V is not rotated).
    rope_apply_rows(q, n_head,    head_dim, config.rope_theta, config.rope_type);
    rope_apply_rows(k, n_kv_head, head_dim, config.rope_theta, config.rope_type);

    Tensor context =
        mha_attention(q, k, v, mcfg);

    Tensor attn_out =
        linear(context, o_w_t);

    Tensor after_attn =
        residual_add(hidden, attn_out);

    // --------------------------------
    // FEED-FORWARD SUB-BLOCK (SwiGLU)
    // --------------------------------

    Tensor gate_w_t = transpose(gate_weight);
    Tensor up_w_t   = transpose(up_weight);
    Tensor down_w_t = transpose(down_weight);

    Tensor ffn_normed =
        rmsnorm(after_attn, ffn_norm, config.eps);

    Tensor gate = linear(ffn_normed, gate_w_t);
    Tensor up   = linear(ffn_normed, up_w_t);

    Tensor activated = silu(gate);
    Tensor gated     = elementwise_mul(activated, up);

    Tensor ffn_out =
        linear(gated, down_w_t);

    Tensor output =
        residual_add(after_attn, ffn_out);

    output.name = "layer_output";

    // --------------------------------
    // UNLOAD CURRENT LAYER WEIGHTS
    // --------------------------------

    evictor.evict_all();

    return output;
}

// --------------------------------------------------------------------
// Incremental decode path (MR13.1)
// --------------------------------------------------------------------

void StreamingExecutor::begin_decode(
    int n_layers,
    const LayerConfig& cfg
) {
    config = cfg;
    session.active   = true;
    session.position = 0;
    session.k_cache.assign(n_layers, Tensor{});
    session.v_cache.assign(n_layers, Tensor{});
}

static Tensor append_row(const Tensor& cache, const Tensor& row) {
    // cache: [ctx, w] (possibly empty), row: [1, w] -> [ctx+1, w]
    int64_t w   = row.shape[1];
    int64_t ctx = (cache.shape.size() == 2) ? cache.shape[0] : 0;
    Tensor out = create_owned_tensor({ctx + 1, w}, DType::FLOAT32);
    for (int64_t i = 0; i < ctx; i++)
        for (int64_t j = 0; j < w; j++)
            out.data()[i * w + j] = cache.value(i * w + j);
    for (int64_t j = 0; j < w; j++)
        out.data()[ctx * w + j] = row.value(j);
    return out;
}

Tensor StreamingExecutor::decode_layer(
    const Tensor& hidden_row,
    int layer_idx,
    int position
) {
    // Per-layer phase accumulators are captured by diffing the global
    // counters around each section, so a single layer's breakdown can be
    // printed even though load/dequant are timed inside the loader.
    prof::ScopedTimer _layer(prof::Phase::DECODE_LAYER);

    const double load0      = prof::seconds(prof::Phase::LOAD);
    const double dequant0   = prof::seconds(prof::Phase::DEQUANT);
    const double transpose0 = prof::seconds(prof::Phase::TRANSPOSE);
    const double attn0      = prof::seconds(prof::Phase::ATTENTION);
    const double ffn0       = prof::seconds(prof::Phase::FFN);

    std::string prefix = "blk." + std::to_string(layer_idx);

    Tensor attn_norm;
    Tensor ffn_norm;
    Tensor q_weight;
    Tensor k_weight;
    Tensor v_weight;
    Tensor out_weight;
    Tensor gate_weight;
    Tensor up_weight;
    Tensor down_weight;

    {
        // load_tensor internally times LOAD (file read) and DEQUANT
        // (dequantization) separately.
        attn_norm  = loader->load_tensor(prefix + ".attn_norm.weight");
        ffn_norm   = loader->load_tensor(prefix + ".ffn_norm.weight");
        q_weight   = loader->load_tensor(prefix + ".attn_q.weight");
        k_weight   = loader->load_tensor(prefix + ".attn_k.weight");
        v_weight   = loader->load_tensor(prefix + ".attn_v.weight");
        out_weight = loader->load_tensor(prefix + ".attn_output.weight");
        gate_weight= loader->load_tensor(prefix + ".ffn_gate.weight");
        up_weight  = loader->load_tensor(prefix + ".ffn_up.weight");
        down_weight= loader->load_tensor(prefix + ".ffn_down.weight");
    }

    evictor.register_tensor(&attn_norm);
    evictor.register_tensor(&ffn_norm);
    evictor.register_tensor(&q_weight);
    evictor.register_tensor(&k_weight);
    evictor.register_tensor(&v_weight);
    evictor.register_tensor(&out_weight);
    evictor.register_tensor(&gate_weight);
    evictor.register_tensor(&up_weight);
    evictor.register_tensor(&down_weight);

    Tensor after_attn;
    {
        prof::ScopedTimer _a(prof::Phase::ATTENTION);

        Tensor normed = rmsnorm(hidden_row, attn_norm, config.eps);

        // GGUF linear weights are [out, in]; multiply directly without
        // materializing a transpose (removes the dominant per-token cost).
        Tensor q = linear_from_gguf(normed, q_weight);   // [1, n_head*hd]
        Tensor k = linear_from_gguf(normed, k_weight);   // [1, n_kv*hd]
        Tensor v = linear_from_gguf(normed, v_weight);   // [1, n_kv*hd]

        int n_head    = config.num_heads > 0 ? config.num_heads : 1;
        int n_kv_head = config.n_kv_heads > 0 ? config.n_kv_heads : n_head;
        int head_dim  = config.head_dim > 0
            ? config.head_dim
            : (int)(q.shape[1] / n_head);

        MHAConfig mcfg;
        mcfg.n_head     = n_head;
        mcfg.n_kv_head  = n_kv_head;
        mcfg.head_dim   = head_dim;
        mcfg.rope_theta = config.rope_theta;
        mcfg.rope_type  = config.rope_type;

        rope_apply_row_at(q, n_head,    head_dim, config.rope_theta, position, config.rope_type);
        rope_apply_row_at(k, n_kv_head, head_dim, config.rope_theta, position, config.rope_type);

        // Append this position's K/V to the layer cache.
        session.k_cache[layer_idx] = append_row(session.k_cache[layer_idx], k);
        session.v_cache[layer_idx] = append_row(session.v_cache[layer_idx], v);

        Tensor context =
            mha_attention_incremental(
                q,
                session.k_cache[layer_idx],
                session.v_cache[layer_idx],
                mcfg);

        Tensor attn_out = linear_from_gguf(context, out_weight);
        after_attn      = residual_add(hidden_row, attn_out);
    }

    Tensor output;
    {
        prof::ScopedTimer _f(prof::Phase::FFN);

        Tensor ffn_normed = rmsnorm(after_attn, ffn_norm, config.eps);
        Tensor gate = linear_from_gguf(ffn_normed, gate_weight);
        Tensor up   = linear_from_gguf(ffn_normed, up_weight);
        Tensor activated = silu(gate);
        Tensor gated     = elementwise_mul(activated, up);
        Tensor ffn_out   = linear_from_gguf(gated, down_weight);
        output           = residual_add(after_attn, ffn_out);
        output.name = "decode_layer_output";
    }

    evictor.evict_all();

    if (prof::enabled()) {
        // The FFN block also includes its transposes; report transpose
        // as the combined attention+ffn transpose time for the layer.
        double load_d      = prof::seconds(prof::Phase::LOAD)      - load0;
        double dequant_d   = prof::seconds(prof::Phase::DEQUANT)   - dequant0;
        double transpose_d = prof::seconds(prof::Phase::TRANSPOSE) - transpose0;
        double attn_d      = prof::seconds(prof::Phase::ATTENTION) - attn0;
        double ffn_d       = prof::seconds(prof::Phase::FFN)       - ffn0;
        double total_d     = load_d + dequant_d + transpose_d + attn_d + ffn_d;

        std::fprintf(stderr,
            "Layer %d:\n"
            "  load=%.4fs\n"
            "  dequant=%.4fs\n"
            "  transpose=%.4fs\n"
            "  attention=%.4fs\n"
            "  ffn=%.4fs\n"
            "  total=%.4fs\n",
            layer_idx, load_d, dequant_d, transpose_d, attn_d, ffn_d, total_d);
    }

    return output;
}

Tensor StreamingExecutor::decode_step(
    const Tensor& hidden_row
) {
    if (!session.active) {
        throw std::runtime_error(
            "decode_step called before begin_decode");
    }

    prof::ScopedTimer _step(prof::Phase::DECODE_STEP);

    int n_layers = (int)session.k_cache.size();
    Tensor h = hidden_row;
    for (int l = 0; l < n_layers; l++) {
        h = decode_layer(h, l, session.position);
    }
    session.position++;
    return h;
}

void StreamingExecutor::print_profile_summary(
    int tokens_generated,
    double total_seconds
) {
    if (!prof::enabled()) {
        return;
    }

    double tps = (total_seconds > 0.0)
        ? (double)tokens_generated / total_seconds
        : 0.0;

    std::fprintf(stderr,
        "\nGeneration summary:\n"
        "  tokens_generated=%d\n"
        "  total_time=%.4fs\n"
        "  tokens_per_second=%.4f\n"
        "\nPhase totals (whole run):\n"
        "  load=%.4fs (%llu calls)\n"
        "  dequant=%.4fs (%llu calls)\n"
        "  transpose=%.4fs (%llu calls)\n"
        "  attention=%.4fs (%llu calls)\n"
        "  ffn=%.4fs (%llu calls)\n"
        "  decode_layer=%.4fs (%llu calls)\n"
        "  decode_step=%.4fs (%llu calls)\n",
        tokens_generated, total_seconds, tps,
        prof::seconds(prof::Phase::LOAD),         (unsigned long long)prof::calls(prof::Phase::LOAD),
        prof::seconds(prof::Phase::DEQUANT),      (unsigned long long)prof::calls(prof::Phase::DEQUANT),
        prof::seconds(prof::Phase::TRANSPOSE),    (unsigned long long)prof::calls(prof::Phase::TRANSPOSE),
        prof::seconds(prof::Phase::ATTENTION),    (unsigned long long)prof::calls(prof::Phase::ATTENTION),
        prof::seconds(prof::Phase::FFN),          (unsigned long long)prof::calls(prof::Phase::FFN),
        prof::seconds(prof::Phase::DECODE_LAYER), (unsigned long long)prof::calls(prof::Phase::DECODE_LAYER),
        prof::seconds(prof::Phase::DECODE_STEP),  (unsigned long long)prof::calls(prof::Phase::DECODE_STEP));
}

void StreamingExecutor::unload_tensor(
    Tensor& t
) {

    t.owned_data.clear();

    t.owned_data.shrink_to_fit();
}

}
