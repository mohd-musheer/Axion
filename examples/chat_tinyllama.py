#!/usr/bin/env python3
"""Developer-facing TinyLlama demo for Axion (CPU-first GGUF runtime).

Drives the compiled ``build/axion_cli.exe`` with a real TinyLlama GGUF and
a natural-language prompt. Tokenization, generation, and detokenization
all happen inside Axion using the tokenizer embedded in the GGUF file, so
this script has no third-party dependencies (stdlib only).

Usage:
    python examples/chat_tinyllama.py
    python examples/chat_tinyllama.py --prompt "What is Python?"
    python examples/chat_tinyllama.py --prompt "Hi" --max-new 64 --trace
    python examples/chat_tinyllama.py \\
        --model models/gguf/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \\
        --exe build/axion_cli.exe

The script:
  * loads a real TinyLlama GGUF (via axion_cli),
  * accepts a text prompt,
  * runs generation,
  * prints the generated text,
  * prints generation statistics (when --trace is passed),
  * exits cleanly (0 on success, non-zero on failure).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXE = ROOT / "build" / "axion_cli.exe"
DEFAULT_MODEL = (
    ROOT / "models" / "gguf" / "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Axion TinyLlama chat demo")
    ap.add_argument("--exe", default=str(DEFAULT_EXE),
                    help="Path to axion_cli executable.")
    ap.add_argument("--model", default=str(DEFAULT_MODEL),
                    help="Path to the TinyLlama GGUF model.")
    ap.add_argument("--prompt", default="What is Python?",
                    help="Text prompt to generate from.")
    ap.add_argument("--max-new", type=int, default=64,
                    help="Maximum number of new tokens to generate.")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature (0 = greedy).")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trace", action="store_true",
                    help="Show Axion's execution flow and timing.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    exe = Path(args.exe)
    model = Path(args.model)

    if not exe.exists():
        print(f"error: executable not found: {exe}", file=sys.stderr)
        print("Build it first: cmake --build build --target axion_cli -j8",
              file=sys.stderr)
        return 2
    if not model.exists():
        print(f"error: model not found: {model}", file=sys.stderr)
        return 2

    cmd = [
        str(exe),
        "--model", str(model),
        "--prompt", args.prompt,
        "--max-new", str(args.max_new),
        "--temperature", str(args.temperature),
        "--top-k", str(args.top_k),
        "--top-p", str(args.top_p),
        "--seed", str(args.seed),
    ]
    if args.trace:
        cmd.append("--trace")

    print("=" * 60)
    print(f"Prompt: {args.prompt!r}")
    print("=" * 60)
    sys.stdout.flush()

    t0 = time.time()
    # Stream stderr (trace) live; capture stdout for the result block.
    result = subprocess.run(cmd, text=True, capture_output=True)
    wall = time.time() - t0

    if args.trace and result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    if result.returncode != 0:
        print("\n--- axion_cli FAILED ---", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return result.returncode

    print(result.stdout, end="")
    print("-" * 60)
    print(f"Demo wall time: {wall:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
