// Axion single-token inference example (Phase 13 / MR8).
//
// Usage:
//   axion_single_token <model.gguf> <tok0> [tok1 ...]
//
// Loads a GGUF model, builds a ModelRunner, and prints the greedily
// predicted next token id for the given prompt token sequence. This is
// the smallest possible end-to-end inference driver and exists to
// exercise the full pipeline against a real model from the command
// line; the production CLI (with text prompts and sampling) arrives in
// a later MR.

#include "../gguf/gguf.hpp"
#include "../runtime/model_runner.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {

    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <model.gguf> <tok0> [tok1 ...]\n", argv[0]);
        return 2;
    }

    const std::string model_path = argv[1];

    std::vector<int> tokens;
    for (int i = 2; i < argc; i++) {
        tokens.push_back(std::atoi(argv[i]));
    }

    axion::GGUFLoader loader;
    if (!loader.load_file(model_path)) {
        std::fprintf(stderr, "failed to load GGUF: %s\n",
                     model_path.c_str());
        return 1;
    }

    axion::ModelRunner runner(&loader);

    const axion::ModelConfig& cfg = runner.config();
    std::printf("arch=%s layers=%d heads=%d hidden=%d vocab=%d\n",
                cfg.arch.empty() ? "<none>" : cfg.arch.c_str(),
                cfg.n_layers, cfg.n_heads, cfg.hidden, cfg.vocab);

    std::printf("prompt tokens:");
    for (int t : tokens) std::printf(" %d", t);
    std::printf("\n");

    int next = runner.predict_next_token(tokens);

    std::printf("next token id: %d\n", next);
    return 0;
}
