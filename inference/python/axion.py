"""Axion high-level Python API (Phase 14 / MR12).

Thin, stable wrapper over the pybind11 ``axion_cpp`` module exposing the
supported entrypoints:

    load_model(path)              -> AxionModel
    AxionModel.predict_next_token(tokens) -> int
    AxionModel.generate(prompt, ...)      -> list[int]
    tokenize(text)                -> list[int]   (placeholder)
    detokenize(ids)               -> str         (placeholder)

Tokenizer note: Axion does not yet ship a BPE tokenizer, so tokenize /
detokenize operate on integer token ids (identity-style). They exist so
callers can write tokenizer-agnostic code today and get real behaviour
when the tokenizer lands, without changing call sites.
"""

from __future__ import annotations

from typing import List

import axion_cpp as _C


class AxionModel:
    """Loaded model handle. Keeps the GGUF loader alive for the runner."""

    def __init__(self, loader, runner):
        self._loader = loader      # keep alive; runner holds a raw ptr
        self._runner = runner

    @property
    def config(self):
        return self._runner.config()

    def predict_next_token(self, tokens: List[int]) -> int:
        return self._runner.predict_next_token(list(tokens))

    def generate(
        self,
        prompt: List[int],
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 0.0,
        seed: int = 0,
    ) -> List[int]:
        params = _C.ModelRunner.GenerationParams()
        params.max_new_tokens = int(max_new_tokens)
        params.sampling.temperature = float(temperature)
        params.sampling.top_k = int(top_k)
        params.sampling.top_p = float(top_p)
        params.sampling.seed = int(seed)
        return list(self._runner.generate(list(prompt), params))


def load_model(path: str) -> AxionModel:
    """Load a GGUF model and return a ready :class:`AxionModel`."""
    loader, runner = _C.load_model(path)
    return AxionModel(loader, runner)


def tokenize(text: str) -> List[int]:
    """Placeholder tokenizer: parse whitespace-separated integer ids.

    Replace with a real BPE tokenizer in a later MR.
    """
    return [int(tok) for tok in text.split()]


def detokenize(ids: List[int]) -> str:
    """Placeholder detokenizer: space-join ids as text."""
    return " ".join(str(int(i)) for i in ids)


# --------------------------------------------------------------------
# Text-prompt CLI (Phase 14.5)
#
# A user-facing entrypoint that layers a real tokenizer over the
# token-id-space C++ runtime:
#
#   python -m inference.python.axion --model model.gguf --prompt "Hello"
#
# Tokenization uses transformers (AutoTokenizer) when --tokenizer is
# given; otherwise the prompt is parsed as whitespace-separated ids so
# the command still works with no tokenizer installed.
# --------------------------------------------------------------------
def _cli() -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        prog="axion", description="Axion text inference CLI")
    ap.add_argument("--model", required=True, help="GGUF model path.")
    ap.add_argument("--prompt", required=True,
                    help="Prompt text, or whitespace-separated token ids "
                         "when no --tokenizer is given.")
    ap.add_argument("--tokenizer", default=None,
                    help="HF tokenizer name/path (e.g. the TinyLlama repo). "
                         "If omitted, --prompt is parsed as token ids.")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = None
    if args.tokenizer:
        from .tokenizer import AxionTokenizer
        tok = AxionTokenizer(args.tokenizer)
        ids = tok.encode(args.prompt)
    else:
        ids = [int(t) for t in args.prompt.split()]

    if not ids:
        sys.stderr.write("error: empty prompt / no tokens\n")
        return 2

    model = load_model(args.model)
    out = model.generate(
        ids,
        max_new_tokens=args.max_new,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    generated = out[len(ids):]

    print("prompt tokens:", ids)
    print("generated tokens:", generated)
    if tok is not None:
        print("output text:", tok.decode(out))
    else:
        print("output text: (pass --tokenizer to render ids as text)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
