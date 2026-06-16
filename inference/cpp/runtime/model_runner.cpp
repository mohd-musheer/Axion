#include "model_runner.hpp"

#include "token_embedding.hpp"
#include "final_norm.hpp"
#include "logits.hpp"
#include "../core/tensor_factory.hpp"
#include "../core/profile.hpp"

#include <chrono>
#include <stdexcept>
#include <iostream>

namespace axion {

ModelRunner::ModelRunner(
    GGUFLoader* loader
) : loader(loader), executor(loader) {

    resolve_config();
}

void ModelRunner::resolve_config() {

    cfg.arch = loader->architecture();

    // GGUF hyperparameter keys are prefixed by the architecture name.
    // Fall back to "llama" naming when the architecture key is absent.
    std::string a = cfg.arch.empty() ? "llama" : cfg.arch;

    cfg.n_layers =
        static_cast<int>(loader->get_u32(a + ".block_count", 0));

    cfg.n_heads =
        static_cast<int>(
            loader->get_u32(a + ".attention.head_count", 1));

    cfg.n_kv_heads =
        static_cast<int>(
            loader->get_u32(a + ".attention.head_count_kv", cfg.n_heads));

    cfg.hidden =
        static_cast<int>(loader->get_u32(a + ".embedding_length", 0));

    // head_dim: explicit rope.dimension_count, else hidden / n_heads.
    cfg.head_dim =
        static_cast<int>(loader->get_u32(a + ".rope.dimension_count", 0));
    if (cfg.head_dim <= 0 && cfg.n_heads > 0 && cfg.hidden > 0) {
        cfg.head_dim = cfg.hidden / cfg.n_heads;
    }

    // RoPE base frequency (theta). GGUF key: <arch>.rope.freq_base.
    cfg.rope_theta =
        loader->get_f32(a + ".rope.freq_base", 10000.0f);

    // Vocab size resolution, in priority order:
    //   1. <arch>.vocab_size scalar (often absent in LLaMA-family GGUFs)
    //   2. length of the tokenizer.ggml.tokens array (canonical)
    //   3. outer dimension of token_embd.weight ([vocab, hidden])
    cfg.vocab =
        static_cast<int>(loader->get_u32(a + ".vocab_size", 0));

    if (cfg.vocab <= 0) {
        cfg.vocab = static_cast<int>(
            loader->get_arr_len("tokenizer.ggml.tokens", 0));
    }

    if (cfg.vocab <= 0) {
        std::vector<int64_t> embd_shape =
            loader->tensor_shape("token_embd.weight");
        if (embd_shape.size() == 2) {
            cfg.vocab = static_cast<int>(embd_shape[0]);
        }
    }

    cfg.rms_eps =
        loader->get_f32(
            a + ".attention.layer_norm_rms_epsilon", 1e-6f);

    if (cfg.n_heads <= 0) cfg.n_heads = 1;

    // RoPE convention. llama / tinyllama / mistral / qwen2 GGUFs are all
    // NEOX-style. Unknown architectures default to NEOX as well, since
    // that is the dominant convention for modern decoder GGUFs.
    cfg.rope_type = RopeType::NEOX;

    std::cout
        << "ModelConfig: arch=" << (cfg.arch.empty() ? "<none>" : cfg.arch)
        << " layers=" << cfg.n_layers
        << " heads="  << cfg.n_heads
        << " kv_heads=" << cfg.n_kv_heads
        << " head_dim=" << cfg.head_dim
        << " hidden=" << cfg.hidden
        << " vocab="  << cfg.vocab
        << " rope_theta=" << cfg.rope_theta
        << " eps="    << cfg.rms_eps
        << std::endl;
}

Tensor ModelRunner::embed_tokens(
    const std::vector<int>& tokens
) {

    if (tokens.empty()) {

        throw std::runtime_error(
            "embed_tokens: empty token sequence"
        );
    }

    // Load the embedding matrix [vocab, hidden], do all lookups, then
    // let it fall out of scope so it does not stay resident during the
    // transformer stack.
    Tensor embd =
        loader->load_tensor("token_embd.weight");

    if (embd.shape.size() != 2) {

        throw std::runtime_error(
            "token_embd.weight is not 2D"
        );
    }

    int64_t hidden = embd.shape[1];

    Tensor out =
        create_owned_tensor(
            { static_cast<int64_t>(tokens.size()), hidden },
            DType::FLOAT32
        );
    out.name = "token_embeddings";

    for (size_t s = 0; s < tokens.size(); s++) {

        Tensor row =
            token_embedding(embd, tokens[s]);

        for (int64_t h = 0; h < hidden; h++) {
            out.data()[s * hidden + h] = row.data()[h];
        }
    }

    return out;
}

Tensor ModelRunner::forward_hidden(
    const std::vector<int>& tokens
) {

    // 1. Embedding: tokens -> [seq, hidden]
    Tensor hidden = embed_tokens(tokens);

    // 2. Transformer stack (streamed layer-by-layer)
    LayerConfig lc;
    lc.num_heads  = cfg.n_heads;
    lc.n_kv_heads = cfg.n_kv_heads;
    lc.head_dim   = cfg.head_dim;
    lc.rope_theta = cfg.rope_theta;
    lc.eps        = cfg.rms_eps;
    lc.rope_type  = cfg.rope_type;

    int n_layers = cfg.n_layers;
    if (n_layers <= 0) {
        // No metadata: nothing to stream. Return the embeddings so the
        // caller still gets a well-formed [seq, hidden] tensor.
        std::cout
            << "WARNING: block_count is 0; skipping transformer stack"
            << std::endl;
    } else {
        hidden = executor.forward(hidden, n_layers, lc);
    }

    // 3. Final norm
    Tensor norm_w =
        loader->load_tensor("output_norm.weight");

    Tensor normed =
        final_norm(hidden, norm_w, cfg.rms_eps);

    return normed;
}

std::string select_output_weight_name(
    const std::vector<std::string>& tensor_names
) {
    for (const auto& name : tensor_names) {
        if (name == "output.weight") {
            return "output.weight";          // untied LM head
        }
    }
    return "token_embd.weight";              // tied weights
}

Tensor ModelRunner::load_output_weight() {

    std::string head =
        select_output_weight_name(loader->tensor_names());

    return loader->load_tensor(head);
}

Tensor ModelRunner::forward_logits(
    const std::vector<int>& tokens
) {

    Tensor hidden = forward_hidden(tokens);

    Tensor w_out = load_output_weight();

    // compute_logits: [seq, hidden] x transpose([vocab, hidden])
    //              -> [seq, vocab]
    Tensor logits =
        compute_logits(hidden, w_out);

    logits.name = "logits";
    return logits;
}

int ModelRunner::predict_next_token(
    const std::vector<int>& tokens
) {

    Tensor logits = forward_logits(tokens);

    // Greedy decode over the final position.
    return argmax(logits);
}

Tensor ModelRunner::embed_one(int token) {
    Tensor embd = loader->load_tensor("token_embd.weight");
    if (embd.shape.size() != 2) {
        throw std::runtime_error("token_embd.weight is not 2D");
    }
    Tensor row = token_embedding(embd, token);   // [1, hidden]
    return row;
}

Tensor ModelRunner::logits_from_hidden_row(const Tensor& hidden_row) {
    Tensor norm_w = loader->load_tensor("output_norm.weight");
    Tensor normed = final_norm(hidden_row, norm_w, cfg.rms_eps);
    Tensor w_out  = load_output_weight();
    Tensor logits = compute_logits(normed, w_out);   // [1, vocab]
    return logits;
}

std::vector<int> ModelRunner::generate(
    const std::vector<int>& prompt,
    const GenerationParams& params
) {
    if (prompt.empty()) {
        throw std::runtime_error("generate: empty prompt");
    }

    std::vector<int> seq = prompt;
    std::mt19937_64 rng(params.sampling.seed);

    int n_layers = cfg.n_layers;
    if (n_layers <= 0) {
        // No transformer to run; fall back to the reference path.
        for (int step = 0; step < params.max_new_tokens; step++) {
            Tensor logits = forward_logits(seq);
            int next = sample_last_row(logits, params.sampling, rng);
            if (next < 0) break;
            seq.push_back(next);
            if (params.eos_token_id >= 0 && next == params.eos_token_id) {
                break;
            }
        }
        return seq;
    }

    LayerConfig lc;
    lc.num_heads  = cfg.n_heads;
    lc.n_kv_heads = cfg.n_kv_heads;
    lc.head_dim   = cfg.head_dim;
    lc.rope_theta = cfg.rope_theta;
    lc.eps        = cfg.rms_eps;
    lc.rope_type  = cfg.rope_type;

    // Incremental decode: prefill the prompt one position at a time,
    // building the per-layer K/V caches, then generate token-by-token.
    executor.begin_decode(n_layers, lc);

    // Profiling: reset phase accumulators so the summary reflects only
    // this generation call, and time the whole loop (env-gated).
    if (prof::enabled()) {
        prof::reset();
    }
    auto gen_start = std::chrono::steady_clock::now();
    int tokens_generated = 0;

    Tensor last_hidden;
    for (size_t i = 0; i < seq.size(); i++) {
        Tensor row = embed_one(seq[i]);   // [1, hidden]
        last_hidden = executor.decode_step(row);
    }

    for (int step = 0; step < params.max_new_tokens; step++) {
        Tensor logits = logits_from_hidden_row(last_hidden);  // [1, vocab]
        int next = sample_last_row(logits, params.sampling, rng);
        if (next < 0) break;
        seq.push_back(next);
        tokens_generated++;
        if (params.eos_token_id >= 0 && next == params.eos_token_id) {
            break;
        }

        Tensor row = embed_one(next);
        last_hidden = executor.decode_step(row);
    }

    auto gen_end = std::chrono::steady_clock::now();
    double total_seconds =
        std::chrono::duration<double>(gen_end - gen_start).count();
    executor.print_profile_summary(tokens_generated, total_seconds);

    return seq;
}

}
