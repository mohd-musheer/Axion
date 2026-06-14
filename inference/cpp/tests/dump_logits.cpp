// Standalone Axion logits dumper (Phase 14.5).
//
// Loads a GGUF, runs Axion's full forward pass over explicit token ids,
// and writes the last-position logits (and config/shape metadata) to a
// JSON file with the SAME schema as the llama.cpp reference generator.
// This lets the comparison be done with any external diff tool in
// addition to the in-process comparison in model_validation.cpp.
//
// Usage:
//   axion_dump_logits <model.gguf> <out.json> <tok0> [tok1 ...]
//
// Token ids are passed explicitly so the tokenizer is never a variable;
// feed the exact same ids to gen_reference_logits.py.

#include "../gguf/gguf.hpp"
#include "../runtime/model_runner.hpp"
#include "../core/tensor.hpp"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace axion;
using json = nlohmann::json;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <model.gguf> <out.json> <tok0> [tok1 ...]\n",
            argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string out_path   = argv[2];

    std::vector<int> tokens;
    for (int i = 3; i < argc; i++) {
        tokens.push_back(std::atoi(argv[i]));
    }

    GGUFLoader loader;
    if (!loader.load_file(model_path)) {
        std::fprintf(stderr, "failed to load %s\n", model_path.c_str());
        return 2;
    }

    ModelRunner runner(&loader);
    const ModelConfig& cfg = runner.config();

    Tensor logits = runner.forward_logits(tokens);   // [seq, vocab]
    int64_t seq   = logits.shape[0];
    int64_t vocab = logits.shape[1];

    // Emit only the last position's logits to keep the file small; the
    // harness compares the prediction (last) position.
    std::vector<float> last(vocab);
    for (int64_t j = 0; j < vocab; j++) {
        last[j] = logits.value((seq - 1) * vocab + j);
    }

    json out;
    out["model"]      = model_path;
    out["arch"]       = cfg.arch;
    out["tokens"]     = tokens;
    out["n_embd"]     = cfg.hidden;
    out["n_vocab"]    = vocab;
    out["n_layer"]    = cfg.n_layers;
    out["n_head"]     = cfg.n_heads;
    out["n_head_kv"]  = cfg.n_kv_heads;
    out["rope_theta"] = cfg.rope_theta;
    out["seq"]        = seq;
    // Pad to [seq][vocab]; only the last row is populated.
    json rows = json::array();
    for (int64_t i = 0; i < seq - 1; i++) rows.push_back(json::array());
    rows.push_back(last);
    out["logits"] = rows;

    std::ofstream of(out_path);
    if (!of) {
        std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
        return 2;
    }
    of << out.dump();

    std::fprintf(stderr,
        "wrote %s: seq=%lld vocab=%lld arch=%s\n",
        out_path.c_str(), (long long)seq, (long long)vocab,
        cfg.arch.empty() ? "<none>" : cfg.arch.c_str());
    return 0;
}
