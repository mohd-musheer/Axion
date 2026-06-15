// axion_cli - user-facing Axion inference driver (Phase 14.5).
//
// Correctness phase: the C++ runtime works in TOKEN-ID space and does
// not embed a GGUF/SentencePiece tokenizer (a real tokenizer is a
// separate, later concern). To stay honest and dependency-free, the CLI
// accepts token ids directly and can optionally map ids back to text via
// a sidecar vocab file. Text prompts are tokenized by the thin Python
// wrapper (inference/python/axion.py) which already has transformers.
//
// Usage:
//   axion_cli --model model.gguf --tokens 1 15043 29892 ...
//   axion_cli --model model.gguf --token-file ids.txt
//   axion_cli --model model.gguf --tokens 1 15043 --max-new 16 \
//             --temperature 0.8 --top-k 40 --top-p 0.95 --seed 7
//   axion_cli --model model.gguf --tokens 1 15043 --vocab vocab.txt
//
// --vocab is an optional newline-separated id->piece map (line N is the
// piece for id N) used only to render generated ids as text; without it
// the CLI prints the generated token ids.
//
// For a one-shot text experience use the Python wrapper:
//   python -m inference.python.axion --model model.gguf --prompt "Hello"

#include "../gguf/gguf.hpp"
#include "../gguf/tokenizer.hpp"
#include "../runtime/model_runner.hpp"
#include "../runtime/sampler.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace axion;

using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0) {
    auto t1 = Clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s --model <model.gguf>\n"
        "            (--prompt \"text\" | --tokens id... | --token-file <file>)\n"
        "            [--max-new N] [--temperature T] [--top-k K]\n"
        "            [--top-p P] [--seed S] [--vocab <file>] [--trace]\n"
        "\n"
        "  --prompt   tokenize text with the GGUF-embedded tokenizer\n"
        "  --trace    print generation flow, timing, and throughput\n",
        prog);
}

static std::vector<int> read_token_file(const std::string& path) {
    std::vector<int> ids;
    std::ifstream f(path);
    int v;
    while (f >> v) ids.push_back(v);
    return ids;
}

static std::vector<std::string> read_vocab(const std::string& path) {
    std::vector<std::string> pieces;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) pieces.push_back(line);
    return pieces;
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string token_file;
    std::string vocab_file;
    std::string prompt_text;
    bool have_prompt = false;
    bool trace = false;
    std::vector<int> tokens;

    ModelRunner::GenerationParams gp;
    gp.max_new_tokens = 16;
    gp.sampling.temperature = 0.0f;   // greedy by default

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model") {
            model_path = next("--model");
        } else if (a == "--prompt") {
            prompt_text = next("--prompt");
            have_prompt = true;
        } else if (a == "--trace") {
            trace = true;
        } else if (a == "--token-file") {
            token_file = next("--token-file");
        } else if (a == "--vocab") {
            vocab_file = next("--vocab");
        } else if (a == "--tokens") {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                tokens.push_back(std::atoi(argv[++i]));
            }
        } else if (a == "--max-new") {
            gp.max_new_tokens = std::atoi(next("--max-new"));
        } else if (a == "--temperature") {
            gp.sampling.temperature = (float)std::atof(next("--temperature"));
        } else if (a == "--top-k") {
            gp.sampling.top_k = std::atoi(next("--top-k"));
        } else if (a == "--top-p") {
            gp.sampling.top_p = (float)std::atof(next("--top-p"));
        } else if (a == "--seed") {
            gp.sampling.seed = (uint64_t)std::strtoull(next("--seed"), nullptr, 10);
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "error: unknown arg '%s'\n", a.c_str());
            usage(argv[0]);
            return 2;
        }
    }

    if (model_path.empty()) {
        usage(argv[0]);
        return 2;
    }
    if (!token_file.empty()) {
        tokens = read_token_file(token_file);
    }

    if (trace) std::fprintf(stderr, "[trace] Loading GGUF: %s\n",
                            model_path.c_str());
    Clock::time_point t_load = Clock::now();

    GGUFLoader loader;
    if (!loader.load_file(model_path)) {
        std::fprintf(stderr, "failed to load %s\n", model_path.c_str());
        return 1;
    }
    if (trace) std::fprintf(stderr, "[trace] GGUF parsed in %.1f ms\n",
                            ms_since(t_load));

    // Build the in-process tokenizer from GGUF metadata. Only required
    // when a text prompt is given or text output is requested; if the
    // model carries no tokenizer we degrade to id-space gracefully.
    GGUFTokenizer* tok = nullptr;
    try {
        tok = new GGUFTokenizer(loader);
        if (trace) std::fprintf(stderr,
            "[trace] tokenizer ready (vocab=%d, bos=%d, eos=%d)\n",
            tok->vocab_size(), tok->bos_id(), tok->eos_id());
    } catch (const std::exception& e) {
        if (have_prompt) {
            std::fprintf(stderr,
                "error: --prompt needs an embedded tokenizer but: %s\n",
                e.what());
            return 1;
        }
    }

    // Resolve input tokens. Priority: explicit ids, then text prompt.
    if (tokens.empty() && have_prompt && tok != nullptr) {
        tokens = tok->encode(prompt_text, /*add_bos=*/true);
    }

    if (tokens.empty()) {
        std::fprintf(stderr,
            "error: no input (use --prompt, --tokens, or --token-file)\n");
        return 2;
    }

    // Stop at the model's EOS so generation terminates naturally.
    if (tok != nullptr) gp.eos_token_id = tok->eos_id();

    ModelRunner runner(&loader);

    if (trace) {
        std::fprintf(stderr, "[trace] prompt tokens (%zu):",
                     tokens.size());
        for (int t : tokens) std::fprintf(stderr, " %d", t);
        std::fprintf(stderr, "\n[trace] generating (max_new=%d)...\n",
                     gp.max_new_tokens);
    }

    Clock::time_point t_gen = Clock::now();
    std::vector<int> out = runner.generate(tokens, gp);
    double gen_ms = ms_since(t_gen);

    // Generated ids = everything after the prompt.
    std::vector<int> generated(out.begin() + (long)tokens.size(), out.end());

    if (trace) {
        int n_gen = (int)generated.size();
        double tok_per_s = gen_ms > 0.0
            ? (double)n_gen / (gen_ms / 1000.0) : 0.0;
        // Peak resident weights estimate: one transformer layer's worth
        // of fp32 weights is the streaming high-water mark. Reported as
        // a coarse MB figure derived from hidden size when available.
        std::fprintf(stderr,
            "[trace] generation time: %.2f s\n", gen_ms / 1000.0);
        std::fprintf(stderr,
            "[trace] tokens generated: %d\n", n_gen);
        std::fprintf(stderr,
            "[trace] tokens/sec: %.2f\n", tok_per_s);
    }

    std::printf("prompt tokens:");
    for (int t : tokens) std::printf(" %d", t);
    std::printf("\n");

    std::printf("generated tokens:");
    for (int t : generated) std::printf(" %d", t);
    std::printf("\n");

    // Prefer the embedded tokenizer for text output; fall back to the
    // optional --vocab sidecar, then to id-only output.
    if (tok != nullptr) {
        std::printf("generated text: %s\n",
                    tok->decode(generated).c_str());
        std::printf("full text: %s\n", tok->decode(out).c_str());
    } else if (!vocab_file.empty()) {
        std::vector<std::string> pieces = read_vocab(vocab_file);
        std::string text;
        for (int t : generated) {
            if (t >= 0 && t < (int)pieces.size()) {
                std::string p = pieces[t];
                // SentencePiece marks word boundaries with U+2581; render
                // it as a space for readable plain-text output.
                std::string rep;
                for (size_t k = 0; k < p.size(); k++) {
                    if ((unsigned char)p[k] == 0xE2 && k + 2 < p.size() &&
                        (unsigned char)p[k+1] == 0x96 &&
                        (unsigned char)p[k+2] == 0x81) {
                        rep += ' ';
                        k += 2;
                    } else {
                        rep += p[k];
                    }
                }
                text += rep;
            }
        }
        std::printf("output text:%s\n", text.c_str());
    } else {
        std::printf("output text: (pass --vocab <file> to render ids as text, "
                    "or use the Python wrapper for full tokenization)\n");
    }

    return 0;
}
