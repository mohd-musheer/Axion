// Axion real-model validation harness (Phase 14.5).
//
// Loads a real GGUF model and a llama.cpp reference JSON, runs the full
// Axion forward pass over identical token ids, and reports max/mean abs
// error on logits (and config sanity). The success criterion is that a
// TinyLlama GGUF's last-position logits match llama.cpp within tolerance
// and the top-1 token matches exactly.
//
// Usage:
//   axion_model_validation <model.gguf> <reference.json>
//
// Exit code 0 iff every present section is within tolerance.

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

// Tolerances. Axion dequantizes to f32 and accumulates in f32, while
// llama.cpp mixes f16/f32; these bounds are tight enough to catch a
// structural bug (RoPE style, GQA grouping, KV layout, RMSNorm) but
// loose enough to absorb quantization/rounding noise.
static const double LOGITS_MAX_ABS_TOL  = 0.75;
static const double LOGITS_MEAN_ABS_TOL = 0.08;

struct ErrStats { double max_abs = 0.0; double mean_abs = 0.0; int64_t n = 0; };

static ErrStats compare(const std::vector<float>& a,
                        const std::vector<float>& b) {
    ErrStats s;
    int64_t n = (int64_t)std::min(a.size(), b.size());
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double e = std::fabs((double)a[i] - (double)b[i]);
        if (e > s.max_abs) s.max_abs = e;
        sum += e;
    }
    s.mean_abs = n ? sum / (double)n : 0.0;
    s.n = n;
    return s;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <model.gguf> <reference.json>\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string ref_path   = argv[2];

    std::ifstream rf(ref_path);
    if (!rf) {
        std::fprintf(stderr, "cannot open %s\n", ref_path.c_str());
        return 2;
    }
    json ref; rf >> ref;

    std::vector<int> tokens = ref.at("tokens").get<std::vector<int>>();

    GGUFLoader loader;
    if (!loader.load_file(model_path)) {
        std::fprintf(stderr, "failed to load %s\n", model_path.c_str());
        return 2;
    }

    ModelRunner runner(&loader);
    const ModelConfig& cfg = runner.config();

    int failures = 0;

    // --- Metadata / config equality vs reference ---
    // Each present field must match exactly; a mismatch here means the
    // two engines are not even running the same architecture, so the
    // logits comparison below would be meaningless.
    auto check_int = [&](const char* key, int axion_val) {
        if (!ref.contains(key)) return;
        int re = ref[key].get<int>();
        bool ok = (axion_val == re);
        std::printf("%s  meta %-10s axion=%d ref=%d\n",
                    ok ? "PASS" : "FAIL", key, axion_val, re);
        if (!ok) failures++;
    };

    check_int("n_embd",    cfg.hidden);
    check_int("n_layer",   cfg.n_layers);
    check_int("n_head",    cfg.n_heads);
    check_int("n_head_kv", cfg.n_kv_heads);
    check_int("n_vocab",   cfg.vocab);

    // Sequence length equality: the reference token list must match the
    // ids we are about to feed Axion, otherwise positions diverge.
    if (ref.contains("tokens")) {
        int rseq = (int)ref["tokens"].size();
        bool ok = (rseq == (int)tokens.size());
        std::printf("%s  meta seq_len    axion=%d ref=%d\n",
                    ok ? "PASS" : "FAIL", (int)tokens.size(), rseq);
        if (!ok) failures++;
    }

    if (ref.contains("arch")) {
        std::string ra = ref["arch"].get<std::string>();
        bool ok = (cfg.arch == ra);
        std::printf("%s  meta arch       axion=%s ref=%s\n",
                    ok ? "PASS" : "FAIL",
                    cfg.arch.empty() ? "<none>" : cfg.arch.c_str(),
                    ra.c_str());
        if (!ok) failures++;
    }

    if (ref.contains("rope_theta")) {
        double rt = ref["rope_theta"].get<double>();
        bool ok = std::fabs((double)cfg.rope_theta - rt) <= 1e-3;
        std::printf("%s  meta rope_theta axion=%.1f ref=%.1f\n",
                    ok ? "PASS" : "FAIL", (double)cfg.rope_theta, rt);
        if (!ok) failures++;
    }

    // --- Logits (end-to-end ground truth) ---
    if (ref.contains("logits")) {
        Tensor logits = runner.forward_logits(tokens);  // [seq, vocab]
        int64_t seq   = logits.shape[0];
        int64_t vocab = logits.shape[1];

        auto ref_logits = ref["logits"];  // [seq][vocab]
        bool shape_ok =
            (int64_t)ref_logits.size() == seq &&
            (ref_logits.empty() ||
             (int64_t)ref_logits[0].size() == vocab);
        std::printf("%s  logits shape axion=[%lld,%lld] ref_seq=%zu\n",
                    shape_ok ? "PASS" : "FAIL",
                    (long long)seq, (long long)vocab, ref_logits.size());
        if (!shape_ok) failures++;

        if (shape_ok) {
            // Compare the LAST position (the prediction position).
            std::vector<float> a(vocab), b(vocab);
            for (int64_t j = 0; j < vocab; j++) {
                a[j] = logits.value((seq - 1) * vocab + j);
                b[j] = ref_logits[seq - 1][j].get<float>();
            }
            ErrStats s = compare(a, b);
            bool ok = s.max_abs <= LOGITS_MAX_ABS_TOL &&
                      s.mean_abs <= LOGITS_MEAN_ABS_TOL;
            std::printf("%s  logits[last]  max_abs=%.5f (<=%.2f)  "
                        "mean_abs=%.5f (<=%.2f)\n",
                        ok ? "PASS" : "FAIL",
                        s.max_abs, LOGITS_MAX_ABS_TOL,
                        s.mean_abs, LOGITS_MEAN_ABS_TOL);
            if (!ok) failures++;

            // Top-1 token must match exactly: hard floor for the phase.
            int64_t am_a = 0, am_b = 0;
            for (int64_t j = 1; j < vocab; j++) {
                if (a[j] > a[am_a]) am_a = j;
                if (b[j] > b[am_b]) am_b = j;
            }
            bool top1 = (am_a == am_b);
            std::printf("%s  top-1 token axion=%lld ref=%lld\n",
                        top1 ? "PASS" : "FAIL",
                        (long long)am_a, (long long)am_b);
            if (!top1) failures++;
        }
    }

    std::printf("\n%s (%d failure%s)\n",
                failures == 0 ? "AXION MODEL VALIDATION PASSED"
                              : "AXION MODEL VALIDATION FAILED",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
