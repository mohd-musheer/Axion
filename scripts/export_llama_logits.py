#!/usr/bin/env python3
"""Export llama.cpp reference logits straight to reference.json.

DEPRECATED on Windows: this path needs llama-cpp-python, whose wheel
fails to build under MinGW/RTools (ggml-cpu.c /
THREAD_POWER_THROTTLING_STATE). Prefer the native, dependency-free tool
``tools/llama_logits_dump.cpp`` (links the llama.cpp libs you already
built). This script is kept only for environments where a prebuilt
llama-cpp-python wheel is available.

Phase 14.5 (correctness only). This is the single end-to-end producer
for the Axion real-model verification path. It removes the manual
conversion step that ``generate_reference_from_llama_cli.py`` required.

Why this exists
---------------
Modern ``llama-cli`` builds no longer expose ``--logits-all`` /
``--dump-logits`` / ``--save-logits`` on the CLI, so there is no binary
that emits ``llama_logits.f32`` or ``last_logits.txt`` for the converter
to reshape. Instead of patching upstream llama.cpp, we read the logits
through the official Python API (llama-cpp-python), which still exposes
per-position logits via ``logits_all=True``. The GGUF is loaded once and
``reference.json`` is written directly.

    llama.cpp (GGUF) -> export_llama_logits.py -> reference.json
                     -> axion_model_validation.exe -> PASS

Determinism
-----------
We work in *token-id space*: the exact same integer ids are fed to both
llama.cpp and Axion, so the tokenizer is never a variable. Greedy, single
thread, fixed seed -> bit-stable logits.

Schema (identical to gen_reference_logits.py, consumed by the harness)
----------------------------------------------------------------------
    {
      "model":      "<basename>",
      "arch":       "llama",
      "tokens":     [int, ...],          # exact ids fed to BOTH engines
      "n_embd":     int,
      "n_vocab":    int,
      "n_layer":    int,
      "n_head":     int,
      "n_head_kv":  int,
      "rope_theta": float,
      "seq":        int,
      "logits":     [[float, ...], ...]  # [seq][vocab]; with --last-only
                                         # earlier rows are empty.
    }

Usage
-----
Full sequence:
    python scripts/export_llama_logits.py \\
        --model models/tinyllama.Q4_K_M.gguf \\
        --tokens 1 15043 29892 590 1024 338 \\
        --out reference.json

Final-position only (smaller file; the harness compares the last row):
    python scripts/export_llama_logits.py \\
        --model models/tinyllama.Q4_K_M.gguf \\
        --tokens 1 15043 29892 590 1024 338 \\
        --last-only \\
        --out reference.json

Install the backend once:
    pip install llama-cpp-python
"""

import argparse
import json
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="Path to the GGUF file (loaded directly).")
    p.add_argument("--tokens", required=True, nargs="+", type=int,
                   help="Explicit token ids fed to the model (no tokenizer).")
    p.add_argument("--out", required=True,
                   help="Output reference JSON path (reference.json).")
    p.add_argument("--n-threads", type=int, default=1,
                   help="llama.cpp threads (1 = most deterministic).")
    p.add_argument("--n-ctx", type=int, default=0,
                   help="Context size; 0 = auto (len(tokens) + 8, min 64).")
    p.add_argument("--last-only", action="store_true",
                   help="Emit only the final position's logits. The harness "
                        "compares the last position regardless.")
    return p.parse_args()


def _md_int(md, key, default):
    try:
        return int(md.get(key, default))
    except Exception:
        return default


def _md_float(md, key, default):
    try:
        return float(md.get(key, default))
    except Exception:
        return default


def _read_scores(llm, seq, n_vocab, last_only):
    """Pull per-position logits out of the llama.cpp scores buffer.

    Handles both the modern ``llm.scores`` 2-D buffer and the older
    ``llm.eval_logits`` list-of-rows, so the tool works across
    llama-cpp-python versions.
    """
    rows = [seq - 1] if last_only else range(seq)

    scores = getattr(llm, "scores", None)
    if scores is not None:
        out = []
        for i in rows:
            row = scores[i][:n_vocab]
            out.append([float(x) for x in row])
        return out

    eval_logits = getattr(llm, "eval_logits", None)
    if eval_logits is not None:
        out = []
        for i in rows:
            out.append([float(x) for x in list(eval_logits[i])[:n_vocab]])
        return out

    sys.exit("error: this llama-cpp-python build exposes neither "
             "`scores` nor `eval_logits`; cannot read logits.")


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

    seq = len(tokens)
    n_ctx = args.n_ctx if args.n_ctx > 0 else max(64, seq + 8)

    # logits_all=True keeps per-position logits. seed/threads pinned for
    # determinism. We pass token ids directly, never text.
    try:
        llm = Llama(
            model_path=args.model,
            n_ctx=n_ctx,
            n_threads=args.n_threads,
            logits_all=True,
            seed=0,
            verbose=False,
        )
    except TypeError:
        # Older/newer signatures may not accept `seed`; retry without it.
        llm = Llama(
            model_path=args.model,
            n_ctx=n_ctx,
            n_threads=args.n_threads,
            logits_all=True,
            verbose=False,
        )

    llm.reset()
    llm.eval(tokens)

    n_vocab = llm.n_vocab()
    parsed = _read_scores(llm, seq, n_vocab, args.last_only)

    if args.last_only:
        logits = [[] for _ in range(seq - 1)] + parsed
    else:
        logits = parsed

    md = llm.metadata if hasattr(llm, "metadata") else {}
    arch = md.get("general.architecture", "llama")

    out = {
        "model": os.path.basename(args.model),
        "arch": arch,
        "tokens": tokens,
        "n_embd": _md_int(md, f"{arch}.embedding_length", llm.n_embd()),
        "n_vocab": n_vocab,
        "n_layer": _md_int(md, f"{arch}.block_count", 0),
        "n_head": _md_int(md, f"{arch}.attention.head_count", 0),
        "n_head_kv": _md_int(md, f"{arch}.attention.head_count_kv", 0),
        "rope_theta": _md_float(md, f"{arch}.rope.freq_base", 10000.0),
        "seq": seq,
        "logits": logits,
    }

    with open(args.out, "w") as f:
        json.dump(out, f)

    sys.stderr.write(
        f"wrote {args.out}: seq={seq} n_vocab={n_vocab} "
        f"arch={arch} (last-only={args.last_only})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
