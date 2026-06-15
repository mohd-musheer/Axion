#pragma once

#include "../gguf/gguf.hpp"
#include "../core/tensor.hpp"
#include "streaming_executor.hpp"
#include "sampler.hpp"
#include "mha.hpp"

#include <string>
#include <vector>

namespace axion {

// Pick the LM-head tensor name from the model's tensor list, applying
// GGUF tie rules: prefer a dedicated "output.weight"; otherwise fall
// back to the input embedding "token_embd.weight" (tied weights).
// Pure and free of I/O so it can be unit-tested without a model file.
std::string select_output_weight_name(
    const std::vector<std::string>& tensor_names
);

// Hyperparameters resolved from GGUF metadata (architecture-prefixed).
struct ModelConfig {
    std::string arch;
    int   n_layers   = 0;
    int   n_heads    = 1;
    int   n_kv_heads = 1;
    int   head_dim   = 0;
    int   hidden     = 0;
    int   vocab      = 0;
    float rope_theta = 10000.0f;
    float rms_eps    = 1e-6f;
    RopeType rope_type = RopeType::NEOX;
};

// End-to-end GGUF transformer pass: tokens -> embedding -> streaming
// transformer stack -> final norm. Produces the normalized final
// hidden state [seq, hidden]; logits are layered on in MR7.
class ModelRunner {

public:

    explicit ModelRunner(
        GGUFLoader* loader
    );

    // Resolve hyperparameters from metadata. Called by the constructor;
    // exposed so callers can inspect/override the config.
    const ModelConfig& config() const { return cfg; }

    // Run the full stack over the token sequence and return the
    // normalized final hidden state [seq, hidden].
    Tensor forward_hidden(
        const std::vector<int>& tokens
    );

    // Run the full stack and project to vocabulary logits [seq, vocab].
    // The LM head is "output.weight" when present, otherwise the input
    // embedding "token_embd.weight" (tied weights).
    Tensor forward_logits(
        const std::vector<int>& tokens
    );

    // Full prompt -> next token: forward_logits then greedy argmax over
    // the last position. Returns the predicted token id (-1 if empty).
    int predict_next_token(
        const std::vector<int>& tokens
    );

    // Autoregressive generation. Returns prompt + generated tokens.
    struct GenerationParams {
        int            max_new_tokens = 16;
        SamplingParams sampling;
        // Stop generation when this token id is sampled. -1 disables
        // the check (runs the full max_new_tokens budget).
        int            eos_token_id  = -1;
    };

    std::vector<int> generate(
        const std::vector<int>& prompt,
        const GenerationParams& params
    );

private:

    // Load the LM head weight [vocab, hidden] following GGUF tie rules.
    Tensor load_output_weight();

    // Embed a single token id into a [1, hidden] row.
    Tensor embed_one(int token);

    // Project a [1, hidden] normalized row to vocab logits [1, vocab].
    Tensor logits_from_hidden_row(const Tensor& hidden_row);


    GGUFLoader*       loader;
    ModelConfig       cfg;
    StreamingExecutor executor;

    void resolve_config();

    // Build [seq, hidden] by looking up each token row in token_embd.
    Tensor embed_tokens(
        const std::vector<int>& tokens
    );
};

}
