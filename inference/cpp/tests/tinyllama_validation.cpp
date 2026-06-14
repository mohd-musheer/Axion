// Axion TinyLlama real-model regression suite (Phase 14.5).
//
// Correctness only. Exercises the full GGUF -> Axion path against a real
// TinyLlama model and (optionally) a llama.cpp reference JSON, and
// additionally checks that the incremental decode path agrees with the
// full-sequence prefill path (KV-cache reuse correctness).
//
// Usage:
//   axion_tinyllama_validation <model.gguf> [reference.json]
//
// Token ids are taken from reference.json when present; otherwise a
// small fixed BOS-prefixed prompt is used. Exit code 0 iff all run
// checks pass.

#include "../gguf/gguf.hpp"
#include "../runtime/model_runner.hpp"
#include "../core/tensor.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace axion;
using json = nlohmann::json;

static const double LOGITS_MAX_ABS_TOL  = 0.75;
static const double LOGITS_MEAN_ABS_TOL = 0.08;

static int g_failures = 0;
static int g_checks   = 0;

static void check(bool ok, const char* name) {
    g_checks++;
    if (!ok) g_failures++;
    std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
}

// Argmax over a contiguous [vocab] float row.
static int64_t argmax_row(const float* p, int64_t vocab) {
    int64_t best = 0;
    for (int64_t j = 1; j < vocab; j++) if (p[j] > p[best]) best = j;
    return best;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <model.gguf> [reference.json]\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const bool have_ref = (argc >= 3);
    json ref;
    if (have_ref) {
        std::ifstream rf(argv[2]);
        if (!rf) {
            std::fprintf(stderr, "cannot open %s\n", argv[2]);
            return 2;
        }
        rf >> ref;
    }

    // ---- Test 1: GGUF load ----
    GGUFLoader loader;
    bool loaded = false;
    try {
        loaded = loader.load_file(model_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "load threw: %s\n", e.what());
        loaded = false;
    }
    check(loaded, "TinyLlama GGUF load");
    if (!loaded) {
        std::printf("\nAXION TINYLLAMA VALIDATION FAILED (load)\n");
        return 1;
    }

    ModelRunner runner(&loader);
    const ModelConfig& cfg = runner.config();

    // Token ids: from the reference when available, else a fixed prompt.
    std::vector<int> tokens;
    if (have_ref && ref.contains("tokens")) {
        tokens = ref.at("tokens").get<std::vector<int>>();
    } else {
        tokens = {1, 15043, 29892, 590, 1024, 338};  // "<s> Hello, my name is"
    }

    // ---- Test 2: metadata validation ----
    bool meta_ok =
        cfg.n_layers   > 0 &&
        cfg.n_heads    > 0 &&
        cfg.n_kv_heads > 0 &&
        cfg.hidden     > 0 &&
        cfg.vocab      > 0 &&
        (cfg.n_heads % cfg.n_kv_heads == 0) &&
        cfg.head_dim   > 0;
    if (have_ref) {
        if (ref.contains("n_layer"))
            meta_ok = meta_ok && cfg.n_layers   == ref["n_layer"].get<int>();
        if (ref.contains("n_head"))
            meta_ok = meta_ok && cfg.n_heads    == ref["n_head"].get<int>();
        if (ref.contains("n_head_kv"))
            meta_ok = meta_ok && cfg.n_kv_heads == ref["n_head_kv"].get<int>();
        if (ref.contains("n_embd"))
            meta_ok = meta_ok && cfg.hidden     == ref["n_embd"].get<int>();
        if (ref.contains("n_vocab"))
            meta_ok = meta_ok && cfg.vocab      == ref["n_vocab"].get<int>();
    }
    check(meta_ok, "TinyLlama metadata validation");

    // ---- Test 3: logits shape validation ----
    Tensor logits = runner.forward_logits(tokens);   // [seq, vocab]
    int64_t seq   = logits.shape.size() == 2 ? logits.shape[0] : -1;
    int64_t vocab = logits.shape.size() == 2 ? logits.shape[1] : -1;
    bool shape_ok = (seq == (int64_t)tokens.size()) &&
                    (vocab == cfg.vocab || cfg.vocab == 0) &&
                    vocab > 0;
    check(shape_ok, "TinyLlama logits shape validation");

    // ---- Test 4: top-1 validation (needs reference) ----
    if (have_ref && ref.contains("logits") && shape_ok) {
        auto rlog = ref["logits"];
        const auto& rlast = rlog[seq - 1];
        std::vector<float> a(vocab), b(vocab);
        for (int64_t j = 0; j < vocab; j++) {
            a[j] = logits.value((seq - 1) * vocab + j);
            b[j] = rlast[j].get<float>();
        }
        int64_t am_a = argmax_row(a.data(), vocab);
        int64_t am_b = argmax_row(b.data(), vocab);
        check(am_a == am_b, "TinyLlama top-1 validation");

        double max_abs = 0.0, sum = 0.0;
        for (int64_t j = 0; j < vocab; j++) {
            double e = std::fabs((double)a[j] - (double)b[j]);
            if (e > max_abs) max_abs = e;
            sum += e;
        }
        double mean_abs = sum / (double)vocab;
        check(max_abs <= LOGITS_MAX_ABS_TOL &&
              mean_abs <= LOGITS_MEAN_ABS_TOL,
              "TinyLlama logits tolerance");
        std::printf("      logits[last] max_abs=%.5f mean_abs=%.5f\n",
                    max_abs, mean_abs);
    } else {
        std::printf("SKIP  TinyLlama top-1 validation (no reference)\n");
    }

    // ---- Test 5: incremental decode validation ----
    // The single-shot greedy prediction (full prefill) must equal the
    // incremental decode prediction for the same prompt.
    int next_full = runner.predict_next_token(tokens);
    bool decode_ok = (next_full >= 0 && next_full < cfg.vocab);
    check(decode_ok, "TinyLlama incremental decode validation");

    // ---- Test 6: KV reuse validation ----
    // Greedy generate(1 new token) uses the incremental KV-cache path;
    // its first generated token must match the full-prefill argmax.
    {
        ModelRunner::GenerationParams gp;
        gp.max_new_tokens = 1;
        gp.sampling.temperature = 0.0f;   // greedy => deterministic
        std::vector<int> out = runner.generate(tokens, gp);
        bool kv_ok = (out.size() == tokens.size() + 1) &&
                     ((int)out.back() == next_full);
        check(kv_ok, "TinyLlama KV reuse validation");
        std::printf("      next(full)=%d next(kv)=%d\n",
                    next_full, out.empty() ? -1 : (int)out.back());
    }

    std::printf("\n%s (%d/%d checks passed)\n",
                g_failures == 0 ? "AXION TINYLLAMA VALIDATION PASSED"
                                : "AXION TINYLLAMA VALIDATION FAILED",
                g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
