// Axion command-line inference runtime (Phase 14 / MR11).
//
// Usage:
//   axion --model <model.gguf> --prompt "<tok0 tok1 ...>"
//         [--max-tokens N] [--temperature F]
//         [--top-k N] [--top-p F] [--seed N]
//
// Tokenizer note: Axion does not yet ship an in-process BPE tokenizer,
// so --prompt takes SPACE-SEPARATED INTEGER TOKEN IDS rather than raw
// text. This keeps the runtime correct end-to-end; a real tokenizer
// (so --prompt can take "Hello") is a later MR. The flag surface is
// already the final one, so nothing here changes when it lands.

#include "gguf/gguf.hpp"
#include "runtime/model_runner.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string model;
    std::string prompt;
    int    max_tokens  = 16;
    float  temperature = 0.0f;
    int    top_k       = 0;
    float  top_p       = 0.0f;
    unsigned long long seed = 0;
};

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; i++) {
        std::string f = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (f == "--model") {
            const char* v = next("--model"); if (!v) return false; a.model = v;
        } else if (f == "--prompt") {
            const char* v = next("--prompt"); if (!v) return false; a.prompt = v;
        } else if (f == "--max-tokens") {
            const char* v = next("--max-tokens"); if (!v) return false; a.max_tokens = std::atoi(v);
        } else if (f == "--temperature") {
            const char* v = next("--temperature"); if (!v) return false; a.temperature = (float)std::atof(v);
        } else if (f == "--top-k") {
            const char* v = next("--top-k"); if (!v) return false; a.top_k = std::atoi(v);
        } else if (f == "--top-p") {
            const char* v = next("--top-p"); if (!v) return false; a.top_p = (float)std::atof(v);
        } else if (f == "--seed") {
            const char* v = next("--seed"); if (!v) return false; a.seed = std::strtoull(v, nullptr, 10);
        } else if (f == "-h" || f == "--help") {
            return false;
        } else {
            std::fprintf(stderr, "unknown flag: %s\n", f.c_str());
            return false;
        }
    }
    return !a.model.empty();
}

std::vector<int> parse_token_ids(const std::string& s) {
    std::vector<int> ids;
    std::istringstream iss(s);
    int t;
    while (iss >> t) ids.push_back(t);
    return ids;
}

void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s --model <model.gguf> --prompt \"<tok0 tok1 ...>\"\n"
        "           [--max-tokens N] [--temperature F]\n"
        "           [--top-k N] [--top-p F] [--seed N]\n"
        "note: --prompt takes integer token ids (no tokenizer yet)\n",
        prog);
}

} // namespace

int main(int argc, char** argv) {

    Args args;
    if (!parse_args(argc, argv, args)) {
        usage(argv[0]);
        return 2;
    }

    std::vector<int> prompt = parse_token_ids(args.prompt);
    if (prompt.empty()) {
        // Fall back to a single BOS-like token 1 so the pipeline still runs.
        prompt.push_back(1);
    }

    axion::GGUFLoader loader;
    if (!loader.load_file(args.model)) {
        std::fprintf(stderr, "failed to load model: %s\n", args.model.c_str());
        return 1;
    }

    axion::ModelRunner runner(&loader);

    const axion::ModelConfig& cfg = runner.config();
    std::printf("model: %s\n", args.model.c_str());
    std::printf("config: arch=%s layers=%d heads=%d hidden=%d vocab=%d\n",
                cfg.arch.empty() ? "<none>" : cfg.arch.c_str(),
                cfg.n_layers, cfg.n_heads, cfg.hidden, cfg.vocab);

    std::printf("prompt ids:");
    for (int t : prompt) std::printf(" %d", t);
    std::printf("\n");

    axion::ModelRunner::GenerationParams gp;
    gp.max_new_tokens        = args.max_tokens;
    gp.sampling.temperature  = args.temperature;
    gp.sampling.top_k        = args.top_k;
    gp.sampling.top_p        = args.top_p;
    gp.sampling.seed         = args.seed;

    std::vector<int> out = runner.generate(prompt, gp);

    std::printf("generated ids:");
    for (int t : out) std::printf(" %d", t);
    std::printf("\n");

    // "text" view: space-joined ids (placeholder until a detokenizer
    // exists). Generated-only suffix after the prompt.
    std::printf("text:");
    for (size_t i = prompt.size(); i < out.size(); i++) {
        std::printf(" %d", out[i]);
    }
    std::printf("\n");

    return 0;
}
