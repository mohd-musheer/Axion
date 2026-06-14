// Native llama.cpp reference-logits dumper (Phase 14.5).
//
// Produces reference.json for the Axion real-model verification path
// WITHOUT llama-cpp-python. It links directly against the llama.cpp
// libraries you already built (libllama + libggml) using the stable
// public C API declared in <llama.h>. There is no CLI-flag dependency
// (no --logits-all / --dump-logits) and no Python wheel to build.
//
//     llama.cpp (GGUF) -> llama_logits_dump -> reference.json
//                      -> axion_model_validation -> PASS
//
// Determinism: we work in TOKEN-ID space. The exact same integer ids
// are fed here and to Axion, so the tokenizer is never a variable.
// Single batch, full sequence, logits enabled for every position.
//
// Output schema (identical to scripts/gen_reference_logits.py and to
// what tests/model_validation.cpp reads):
//     {
//       "model":      "<basename>",
//       "arch":       "llama",
//       "tokens":     [int, ...],
//       "n_embd":     int,
//       "n_vocab":    int,
//       "n_layer":    int,
//       "n_head":     int,
//       "n_head_kv":  int,
//       "rope_theta": float,
//       "seq":        int,
//       "logits":     [[float, ...], ...]   // [seq][vocab];
//                                            // with --last-only earlier
//                                            // rows are empty.
//     }
//
// Usage:
//   llama_logits_dump --model <m.gguf> --out reference.json \
//                     --tokens 1 15043 29892 590 1024 338 [--last-only]
//
// The metadata fields (n_embd/n_layer/...) are read straight from the
// GGUF via the llama.cpp model API, so the harness's config-equality
// checks compare apples to apples.

#include "llama.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;

static void usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s --model <m.gguf> --out <reference.json> "
        "--tokens <id> [<id> ...] [--last-only]\n", argv0);
}

// Read a string GGUF metadata key via the llama.cpp meta API; returns
// fallback when absent. Works across recent llama.cpp versions.
static std::string meta_str(const llama_model* model, const char* key,
                            const std::string& fallback) {
    char buf[256];
    int n = llama_model_meta_val_str(model, key, buf, sizeof(buf));
    if (n < 0) return fallback;
    return std::string(buf, (size_t)n);
}

static int meta_int(const llama_model* model, const char* key, int fallback) {
    char buf[64];
    int n = llama_model_meta_val_str(model, key, buf, sizeof(buf));
    if (n < 0) return fallback;
    return std::atoi(buf);
}

static float meta_float(const llama_model* model, const char* key,
                        float fallback) {
    char buf[64];
    int n = llama_model_meta_val_str(model, key, buf, sizeof(buf));
    if (n < 0) return fallback;
    return (float)std::atof(buf);
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string out_path;
    std::vector<llama_token> tokens;
    bool last_only = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (a == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (a == "--last-only") {
            last_only = true;
        } else if (a == "--tokens") {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                tokens.push_back((llama_token)std::atoi(argv[++i]));
            }
        } else {
            std::fprintf(stderr, "unknown/!malformed arg: %s\n", a.c_str());
            usage(argv[0]);
            return 2;
        }
    }

    if (model_path.empty() || out_path.empty() || tokens.empty()) {
        usage(argv[0]);
        return 2;
    }

    const int seq = (int)tokens.size();

    // --- Load the model (CPU, deterministic) ---
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;   // CPU only -> reproducible reference.

    llama_model* model =
        llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "failed to load model: %s\n",
                     model_path.c_str());
        llama_backend_free();
        return 2;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = (uint32_t)(seq + 8);
    cparams.n_batch   = (uint32_t)(seq + 8);
    cparams.n_threads = 1;       // single thread -> bit-stable.
    cparams.n_threads_batch = 1;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // --- Build a single batch with logits requested per position ---
    llama_batch batch = llama_batch_init(seq, 0, 1);
    batch.n_tokens = seq;
    for (int i = 0; i < seq; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        // Request logits for every position (or only the last one).
        batch.logits[i]    = last_only ? (i == seq - 1) : 1;
    }

    if (llama_decode(ctx, batch) != 0) {
        std::fprintf(stderr, "llama_decode failed\n");
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }

    // --- Collect logits into [seq][vocab] ---
    json rows = json::array();
    for (int i = 0; i < seq; i++) {
        bool want = last_only ? (i == seq - 1) : true;
        if (!want) {
            rows.push_back(json::array());   // empty padding row.
            continue;
        }
        const float* lp = llama_get_logits_ith(ctx, i);
        if (!lp) {
            std::fprintf(stderr, "no logits for position %d\n", i);
            llama_batch_free(batch);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 2;
        }
        std::vector<float> row(lp, lp + n_vocab);
        rows.push_back(row);
    }

    // --- Metadata (read from GGUF so harness equality checks line up) ---
    std::string arch = meta_str(model, "general.architecture", "llama");
    auto key = [&](const char* suffix) {
        return arch + "." + suffix;
    };

    const char* base = model_path.c_str();
    for (const char* p = model_path.c_str(); *p; ++p) {
        if (*p == '/' || *p == '\\') base = p + 1;
    }

    json out;
    out["model"]      = std::string(base);
    out["arch"]       = arch;
    out["tokens"]     = tokens;
    out["n_embd"]     = meta_int(model, key("embedding_length").c_str(),
                                 llama_model_n_embd(model));
    out["n_vocab"]    = n_vocab;
    out["n_layer"]    = meta_int(model, key("block_count").c_str(),
                                 llama_model_n_layer(model));
    out["n_head"]     = meta_int(model, key("attention.head_count").c_str(),
                                 llama_model_n_head(model));
    out["n_head_kv"]  = meta_int(model,
                                 key("attention.head_count_kv").c_str(),
                                 0);
    out["rope_theta"] = meta_float(model, key("rope.freq_base").c_str(),
                                   10000.0f);
    out["seq"]        = seq;
    out["logits"]     = rows;

    std::ofstream of(out_path);
    if (!of) {
        std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }
    of << out.dump();

    std::fprintf(stderr,
        "wrote %s: seq=%d n_vocab=%d arch=%s (last-only=%d)\n",
        out_path.c_str(), seq, n_vocab, arch.c_str(), (int)last_only);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
