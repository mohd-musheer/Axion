#!/usr/bin/env python3
"""Build reference.json from the llama.cpp `llama-cli` / `llama-eval` CLI.

Phase 14.5, correctness only. This is the Windows-friendly alternative
to `gen_reference_logits.py` / `dump_llama_cpp_reference.py`: it has NO
`llama-cpp-python` dependency. Instead it consumes the logits emitted by
the upstream llama.cpp binaries when run with `--logits-all` (or the
`llama-eval`/`llama-perplexity` style logits dump), and reshapes them
into the exact JSON schema the C++ harness expects.

The harness (`axion_model_validation`) and `axion_dump_logits` both use
this schema:

    {
      "model":      "<path or basename>",
      "arch":       "llama",
      "tokens":     [int, ...],          # exact ids fed to BOTH engines
      "n_embd":     int,
      "n_vocab":    int,
      "n_layer":    int,
      "n_head":     int,
      "n_head_kv":  int,
      "rope_theta": float,
      "seq":        int,
      "logits":     [[float, ...], ...]  # [seq][vocab]; earlier rows may
                                         # be empty when --last-only.
    }

We work in token-id space: you pass the SAME ids to llama.cpp and to
Axion, so the tokenizer is never a variable.

--------------------------------------------------------------------
How to produce the llama.cpp logits file
--------------------------------------------------------------------
There is no single universal flag across llama.cpp versions, so this
script accepts two input formats and you pick whichever your build can
emit:

1. --logits-bin <file>  (preferred, exact)
   A raw little-endian float32 dump of shape [seq, n_vocab] (row-major),
   as written by a small patch / the `llama-eval` example's
   `--logits-all` + binary save. You must also pass --n-vocab and
   --seq (or --tokens, whose length gives seq).

2. --logits-text <file>
   A whitespace/CSV text file with `seq * n_vocab` float values
   (row-major). Same shape requirements.

If your build can only print the final-position logits, pass
--last-only; earlier rows are emitted as empty arrays (the harness only
compares the last position).

Example (binary, full sequence):
    python scripts/generate_reference_from_llama_cli.py \\
        --model models/tinyllama.Q4_K_M.gguf \\
        --tokens 1 15043 29892 590 1024 338 \\
        --logits-bin llama_logits.f32 \\
        --n-vocab 32000 \\
        --n-embd 2048 --n-layer 22 --n-head 32 --n-head-kv 4 \\
        --rope-theta 10000 \\
        --out reference.json

Example (final-position only, text):
    python scripts/generate_reference_from_llama_cli.py \\
        --model models/tinyllama.Q4_K_M.gguf \\
        --tokens 1 15043 29892 590 1024 338 \\
        --logits-text last_logits.txt --last-only \\
        --n-vocab 32000 \\
        --out reference.json
"""

import argparse
import json
import os
import struct
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="Model path/basename recorded in the JSON.")
    p.add_argument("--tokens", required=True, nargs="+", type=int,
                   help="Explicit token ids fed to both engines.")
    p.add_argument("--out", required=True, help="Output reference JSON.")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--logits-bin",
                     help="Raw float32 [seq, n_vocab] (row-major) dump.")
    src.add_argument("--logits-text",
                     help="Text file of seq*n_vocab float values.")

    p.add_argument("--n-vocab", type=int, required=True,
                   help="Vocabulary size (column count of the logits).")
    p.add_argument("--last-only", action="store_true",
                   help="Input holds only the final position's logits.")

    # Metadata recorded for the harness's equality checks. Defaults match
    # TinyLlama-1.1B; override if validating another model.
    p.add_argument("--arch", default="llama")
    p.add_argument("--n-embd", type=int, default=2048)
    p.add_argument("--n-layer", type=int, default=22)
    p.add_argument("--n-head", type=int, default=32)
    p.add_argument("--n-head-kv", type=int, default=4)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    return p.parse_args()


def read_floats_bin(path, count):
    with open(path, "rb") as f:
        data = f.read()
    have = len(data) // 4
    if have < count:
        sys.exit(f"error: {path} holds {have} float32 values, "
                 f"need {count}")
    return list(struct.unpack(f"<{count}f", data[: count * 4]))


def read_floats_text(path, count):
    vals = []
    with open(path, "r") as f:
        for line in f:
            line = line.replace(",", " ")
            for tok in line.split():
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
    if len(vals) < count:
        sys.exit(f"error: {path} holds {len(vals)} values, need {count}")
    return vals[:count]


def main():
    args = parse_args()
    tokens = list(args.tokens)
    seq = len(tokens)
    n_vocab = args.n_vocab

    rows_in = 1 if args.last_only else seq
    count = rows_in * n_vocab

    if args.logits_bin:
        flat = read_floats_bin(args.logits_bin, count)
    else:
        flat = read_floats_text(args.logits_text, count)

    # Reshape row-major into [rows_in][n_vocab].
    parsed = [flat[r * n_vocab:(r + 1) * n_vocab] for r in range(rows_in)]

    if args.last_only:
        logits = [[] for _ in range(seq - 1)] + [parsed[0]]
    else:
        logits = parsed

    out = {
        "model": os.path.basename(args.model),
        "arch": args.arch,
        "tokens": tokens,
        "n_embd": args.n_embd,
        "n_vocab": n_vocab,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_head_kv": args.n_head_kv,
        "rope_theta": args.rope_theta,
        "seq": seq,
        "logits": logits,
    }

    with open(args.out, "w") as f:
        json.dump(out, f)

    sys.stderr.write(
        f"wrote {args.out}: seq={seq} n_vocab={n_vocab} "
        f"arch={args.arch} (last-only={args.last_only})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
