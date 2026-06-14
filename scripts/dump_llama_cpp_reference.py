#!/usr/bin/env python3
"""Dump llama.cpp reference logits for Axion Phase 14.5 validation.

Produces a JSON consumed by the C++ harness (axion_model_validation):

    {
      "model":   "<path>",
      "tokens":  [int, ...],
      "n_vocab": int,
      "n_embd":  int,
      "logits":  [[float, ...], ...]   # [seq][vocab]
    }

Determinism: greedy, no sampling, single thread, fixed token ids,
seed 0. The Axion harness loads the SAME gguf and the SAME token ids.

Usage:
    python scripts/dump_llama_cpp_reference.py \\
        --model models/gguf/tinyllama.gguf \\
        --tokens 1 15043 29871 \\
        --out reference.json

Requires llama-cpp-python built for the same quantization the model
uses (pip install llama-cpp-python). If you only have the llama.cpp
binaries, run `llama-cli --logits-all` and reshape its output into the
JSON above; the harness only depends on the schema, not the producer.
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        from llama_cpp import Llama
    except ImportError:
        sys.exit("llama-cpp-python not installed; pip install llama-cpp-python")

    llm = Llama(
        model_path=args.model,
        n_ctx=512,
        logits_all=True,
        n_threads=1,
        seed=0,
        verbose=False,
    )

    llm.reset()
    llm.eval(args.tokens)

    # Logits for every position: shape [seq, vocab].
    logits = [list(map(float, row)) for row in llm.eval_logits]

    out = {
        "model": args.model,
        "tokens": args.tokens,
        "n_vocab": llm.n_vocab(),
        "n_embd": llm.n_embd(),
        "logits": logits,
    }

    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}: seq={len(logits)} vocab={out['n_vocab']}")


if __name__ == "__main__":
    main()
