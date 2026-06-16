// Per-layer hidden-state dumper for Axion <-> llama.cpp parity.
//
// Phase 14.5 correctness tooling. Runs the SAME forward pass that
// ModelRunner uses, but captures the hidden state after every
// transformer layer (plus embeddings, final norm, and last-position
// logits) and writes a compact JSON. Pairing this with a llama.cpp
// per-layer dump pinpoints the FIRST layer where the two diverge,
// which is the exact mathematical source of any generation corruption.
//
// To keep files small and comparisons robust we emit per-tensor
// summary statistics (mean, abs-mean, min, max, l2) for each layer's
// last-position hidden row, plus the full last-position logits.
//
// Usage:
//   axion_dump_layers <model.gguf> <out.json> <tok0> [tok1 ...]
//
// Feed the identical ids to the llama.cpp reference dumper.

#include "../gguf/gguf.hpp"
#include "../runtime/streaming_executor.hpp"
#include "../runtime/model_runner.hpp"
#include "../runtime/token_embedding.hpp"
#include "../runtime/final_norm.hpp"
#include "../runtime/logits.hpp"
#include "../core/tensor.hpp"
#include "../core/tensor_factory.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace axion;
using json = nlohmann::json;

// Summary stats over the last-position row of a [seq, hidden] tensor.
static json row_stats(const Tensor& t) {
    int64_t seq    = t.shape.size() == 2 ? t.shape[0] : 1;
    int64_t hidden = t.shape.size() == 2 ? t.shape[1] : t.numel();
    int64_t off    = (seq - 1) * hidden;

    double sum = 0.0, abssum = 0.0, l2 = 0.0;
    double mn = 1e30, mx = -1e30;
    for (int64_t j = 0; j < hidden; j++) {
        double v = t.value(off + j);
        sum    += v;
        abssum += std::fabs(v);
        l2     += v * v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    json s;
    s["mean"]     = sum / (double)hidden;
    s["abs_mean"] = abssum / (double)hidden;
    s["min"]      = mn;
    s["max"]      = mx;
    s["l2"]       = std::sqrt(l2);
    s["hidden"]   = hidden;
    return s;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <model.gguf> <out.json> <tok0> [tok1 ...]\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string out_path   = argv[2];

    std::vector<int> tokens;
    for (int i = 3; i < argc; i++) tokens.push_back(std::atoi(argv[i]));
    if (tokens.empty()) {
        std::fprintf(stderr, "error: need at least one token id\n");
        return 2;
    }

    GGUFLoader loader;
    if (!loader.load_file(model_path)) {
        std::fprintf(stderr, "failed to load %s\n", model_path.c_str());
        return 2;
    }

    ModelRunner runner(&loader);   // resolves config + logs it
    const ModelConfig& cfg = runner.config();

    // Re-run the stack here (rather than ModelRunner::forward_hidden) so
    // we can capture each layer's output. This mirrors forward_hidden
    // exactly: embeddings -> per-layer execute -> final norm -> logits.
    Tensor embd = loader.load_tensor("token_embd.weight");
    int64_t hidden_sz = embd.shape[1];
    Tensor hidden = create_owned_tensor(
        {(int64_t)tokens.size(), hidden_sz}, DType::FLOAT32);
    for (size_t s = 0; s < tokens.size(); s++) {
        Tensor row = token_embedding(embd, tokens[s]);
        for (int64_t h = 0; h < hidden_sz; h++)
            hidden.data()[s * hidden_sz + h] = row.data()[h];
    }

    json out;
    out["model"]   = model_path;
    out["arch"]    = cfg.arch;
    out["tokens"]  = tokens;
    out["n_layer"] = cfg.n_layers;
    out["n_embd"]  = cfg.hidden;
    out["layers"]  = json::array();
    out["embeddings"] = row_stats(hidden);

    LayerConfig lc;
    lc.num_heads  = cfg.n_heads;
    lc.n_kv_heads = cfg.n_kv_heads;
    lc.head_dim   = cfg.head_dim;
    lc.rope_theta = cfg.rope_theta;
    lc.eps        = cfg.rms_eps;
    lc.rope_type  = cfg.rope_type;

    // Stream one layer at a time, recording the output stats per layer.
    StreamingExecutor exec(&loader);
    for (int l = 0; l < cfg.n_layers; l++) {
        hidden = exec.forward(hidden, 1, lc);   // exactly one layer
        json le;
        le["layer"] = l;
        le["stats"] = row_stats(hidden);
        out["layers"].push_back(le);
    }

    Tensor norm_w = loader.load_tensor("output_norm.weight");
    Tensor normed = final_norm(hidden, norm_w, cfg.rms_eps);
    out["final_hidden"] = row_stats(normed);

    std::string head =
        select_output_weight_name(loader.tensor_names());
    Tensor w_out  = loader.load_tensor(head);
    Tensor logits = compute_logits(normed, w_out);   // [seq, vocab]

    int64_t seq   = logits.shape[0];
    int64_t vocab = logits.shape[1];
    std::vector<float> last(vocab);
    int64_t best = 0;
    for (int64_t j = 0; j < vocab; j++) {
        last[j] = logits.value((seq - 1) * vocab + j);
        if (last[j] > last[best]) best = j;
    }
    out["n_vocab"]   = vocab;
    out["argmax"]    = best;
    out["logits"]    = last;   // last-position only

    std::ofstream of(out_path);
    if (!of) {
        std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
        return 2;
    }
    of << out.dump();
    std::fprintf(stderr,
        "wrote %s: layers=%d vocab=%lld argmax=%lld\n",
        out_path.c_str(), cfg.n_layers, (long long)vocab, (long long)best);
    return 0;
}
