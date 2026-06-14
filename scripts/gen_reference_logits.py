#!/usr/bin/env python3
"""Generate a llama.cpp reference logits JSON for Axion validation.

Phase 14.5 (correctness only). This produces the ground-truth file that
`axion_model_validation` compares against. It is fully reproducible: given
the same GGUF, the same token ids, and a deterministic (greedy) llama.cpp
build, the emitted logits are bit-stable.

We deliberately work in *token-id space* and do NOT tokenize text here.
That removes the tokenizer as a variable: Axion and llama.cpp are fed the
exact same integer ids, so any logits divergence is attributable to the
forward pass (RoPE / GQA / KV layout / RMSNorm / dequant), not tokenizer
mismatch.

Reference backend: llama-cpp-python (pip install llama-cpp-python), which
exposes per-position logits via `logits_all=True`.

Usage:
    python scripts/gen_reference_logits.py \\
        --model models/tinyllama-1.1b-chat-q4_k_m.gguf \\
        --tokens 1 15043 29892 590 1024 338 \\
        --out reference.json

Output JSON schema (consumed by tests/model_validation.cpp):
    {
      "model":  "<basename>",
      "arch":   "llama",
      "tokens": [int, ...],          # exact ids fed to both engines
      "n_embd": int,                 # hidden size (config equality)
      "n_vocab": int,                # vocab size (shape equality)
      "n_layer": int,
      "n_head": int,
      "n_head_kv": int,
      "rope_theta": float,
      "logits": [[float, ...], ...]  # [seq][vocab], full per-position
    }
"""

import argparse
import json
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="Path to the TinyLlama GGUF file.")
    p.add_argument("--tokens", required=True, nargs="+", type=int,
                   help="Explicit token ids fed to the model (no tokenizer).")
    p.add_argument("--out", required=True,
                   help="Output reference JSON path.")
    p.add_argument("--n-threads", type=int, default=1,
                   help="llama.cpp threads (1 = most deterministic).")
    p.add_argument("--last-only", action="store_true",
                   help="Emit only the final position's logits (smaller file). "
                        "The harness compares the last position regardless.")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from llama_cpp import Llama
    except ImportError:
        sys.stderr.write(
            "error: llama-cpp-python is required.\n"
            "       pip install llama-cpp-python\n")
        return 2

    if not os.path.isfile(args.model):
        sys.stderr.write(f"error: model not found: {args.model}\n")
        return 2

    tokens = list(args.tokens)
    if not tokens:
        sys.stderr.write("error: --tokens must be non-empty\n")
        return 2

    # logits_all=True keeps per-position logits; n_ctx must cover the seq.
    llm = Llama(
        model_path=args.model,
        n_ctx=max(64, len(tokens) + 8),
        n_threads=args.n_threads,
        logits_all=True,
        verbose=False,
    )

    # Evaluate the exact token ids. eval() fills the logits buffer for
    # every position because logits_all=True.
    llm.reset()
    llm.eval(tokens)

    seq = len(tokens)
    n_vocab = llm.n_vocab()

    # Pull per-position logits out of the llama.cpp scores buffer.
    all_logits = []
    if args.last_only:
        rows = [seq - 1]
    else:
        rows = range(seq)
    for i in rows:
        row = llm.scores[i][:n_vocab]
        all_logits.append([float(x) for x in row])

    # If last-only, pad earlier rows with empty lists so [seq] indexing
    # still works in the harness (it only reads the last row).
    if args.last_only:
        all_logits = [[] for _ in range(seq - 1)] + all_logits

    md = llm.metadata if hasattr(llm, "metadata") else {}

    def md_int(key, default):
        try:
            return int(md.get(key, default))
        except Exception:
            return default

    def md_float(key, default):
        try:
            return float(md.get(key, default))
        except Exception:
            return default

    arch = md.get("general.architecture", "llama")

    out = {
        "model": os.path.basename(args.model),
        "arch": arch,
        "tokens": tokens,
        "n_embd": md_int(f"{arch}.embedding_length", llm.n_embd()),
        "n_vocab": n_vocab,
        "n_layer": md_int(f"{arch}.block_count", 0),
        "n_head": md_int(f"{arch}.attention.head_count", 0),
        "n_head_kv": md_int(f"{arch}.attention.head_count_kv", 0),
        "rope_theta": md_float(f"{arch}.rope.freq_base", 10000.0),
        "logits": all_logits,
    }

    with open(args.out, "w") as f:
        json.dump(out, f)

    sys.stderr.write(
        f"wrote {args.out}: seq={seq} n_vocab={n_vocab} "
        f"arch={arch} (last-only={args.last_only})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
