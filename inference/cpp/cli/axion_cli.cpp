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
#include "../runtime/model_runner.hpp"
#include "../runtime/sampler.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace axion;

static void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s --model <model.gguf> "
        "(--tokens id... | --token-file <file>)\n"
        "            [--max-new N] [--temperature T] [--top-k K]\n"
        "            [--top-p P] [--seed S] [--vocab <file>]\n",
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
    if (tokens.empty()) {
        std::fprintf(stderr,
            "error: no input tokens (use --tokens or --token-file)\n");
        return 2;
    }

    GGUFLoader loader;
    if (!loader.load_file(model_path)) {
        std::fprintf(stderr, "failed to load %s\n", model_path.c_str());
        return 1;
    }

    ModelRunner runner(&loader);

    std::vector<int> out = runner.generate(tokens, gp);

    // Generated ids = everything after the prompt.
    std::vector<int> generated(out.begin() + (long)tokens.size(), out.end());

    std::printf("prompt tokens:");
    for (int t : tokens) std::printf(" %d", t);
    std::printf("\n");

    std::printf("generated tokens:");
    for (int t : generated) std::printf(" %d", t);
    std::printf("\n");

    if (!vocab_file.empty()) {
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
