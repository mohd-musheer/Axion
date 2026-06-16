"""Axion <-> llama.cpp logits parity (Phase 14.5 correctness gate).

Feeds the IDENTICAL fixed token ids to both engines (no tokenizer in the
loop, so any divergence is attributable to the forward pass: RoPE / GQA /
KV layout / RMSNorm / dequant / logits projection) and compares the
last-position logits.

The reference can come from either:
  * a committed reference JSON next to this test
    (tests/load/data/llama_cpp_reference.json), or
  * llama-cpp-python, generated on the fly when installed.

The test reports divergence statistics (max abs, mean abs, argmax
agreement) and FAILS when divergence exceeds tolerance. It skips when
neither the Axion binary/model nor any reference is available, so CI
without the weights does not fail.

Work in token-id space, deterministic (greedy), single thread.

Run with:
    pytest tests/load/test_llama_cpp_parity.py -v -s
"""

from pathlib import Path
import json
import subprocess
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
DUMP_EXE = ROOT / "build" / "axion_dump_logits.exe"
MODEL = ROOT / "models" / "gguf" / "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
REFERENCE = Path(__file__).resolve().parent / "data" / "llama_cpp_reference.json"

# "What is Python?" tokenized for TinyLlama (BOS + content). Kept fixed
# so both engines see identical ids; update alongside the reference.
TOKENS = [1, 1724, 338, 5132, 29973]

# Logits divergence tolerances. llama.cpp runs fp16/quantized kernels and
# Axion dequantizes to fp32, so small numerical drift is expected; a
# convention/layout bug produces divergence orders of magnitude larger.
MAX_ABS_TOL = 0.75
MEAN_ABS_TOL = 0.08

TIMEOUT_SECONDS = 1800


def _require(path: Path, what: str) -> None:
    if not path.exists():
        pytest.skip(f"{what} not present: {path}")


def _axion_last_logits(tmp: Path):
    out_json = tmp / "axion_logits.json"
    cmd = [str(DUMP_EXE), str(MODEL), str(out_json)] + [str(t) for t in TOKENS]
    res = subprocess.run(
        cmd, text=True, capture_output=True, timeout=TIMEOUT_SECONDS
    )
    assert res.returncode == 0, (
        f"axion_dump_logits exited {res.returncode}\n{res.stderr}"
    )
    data = json.loads(out_json.read_text())
    seq = int(data["seq"])
    return data["logits"][seq - 1]


def _reference_last_logits(tmp: Path):
    # 1. Prefer a committed reference (no llama.cpp dependency in CI).
    if REFERENCE.exists():
        data = json.loads(REFERENCE.read_text())
        if data.get("tokens") != TOKENS:
            pytest.skip(
                "committed reference token ids differ from TOKENS; "
                "regenerate tests/load/data/llama_cpp_reference.json"
            )
        logits = data["logits"]
        return logits[len(logits) - 1]

    # 2. Otherwise generate via llama-cpp-python if available.
    try:
        from llama_cpp import Llama
    except ImportError:
        pytest.skip(
            "no reference: commit tests/load/data/llama_cpp_reference.json "
            "or pip install llama-cpp-python"
        )

    _require(MODEL, "TinyLlama GGUF model")
    llm = Llama(
        model_path=str(MODEL),
        n_ctx=max(64, len(TOKENS) + 8),
        n_threads=1,
        logits_all=True,
        verbose=False,
    )
    llm.reset()
    llm.eval(TOKENS)
    n_vocab = llm.n_vocab()
    return [float(x) for x in llm.scores[len(TOKENS) - 1][:n_vocab]]


def _argmax(xs):
    best_i, best_v = 0, xs[0]
    for i, v in enumerate(xs):
        if v > best_v:
            best_i, best_v = i, v
    return best_i


def test_llama_cpp_logits_parity():
    _require(DUMP_EXE, "axion_dump_logits.exe")
    _require(MODEL, "TinyLlama GGUF model")

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        axion = _axion_last_logits(tmp)
        ref = _reference_last_logits(tmp)

    assert len(axion) == len(ref), (
        f"vocab size mismatch: axion={len(axion)} ref={len(ref)}"
    )

    max_abs = 0.0
    sum_abs = 0.0
    for a, b in zip(axion, ref):
        e = abs(a - b)
        if e > max_abs:
            max_abs = e
        sum_abs += e
    mean_abs = sum_abs / len(axion)

    am_axion = _argmax(axion)
    am_ref = _argmax(ref)

    print("\n=== Axion vs llama.cpp logits parity ===")
    print(f"tokens:          {TOKENS}")
    print(f"vocab:           {len(axion)}")
    print(f"argmax axion:    {am_axion}")
    print(f"argmax ref:      {am_ref}")
    print(f"max abs diff:    {max_abs:.5f}  (tol {MAX_ABS_TOL})")
    print(f"mean abs diff:   {mean_abs:.5f}  (tol {MEAN_ABS_TOL})")

    assert am_axion == am_ref, (
        f"argmax divergence: axion predicts {am_axion}, llama.cpp {am_ref} "
        f"(max_abs={max_abs:.4f}, mean_abs={mean_abs:.4f}). This indicates a "
        "forward-pass math bug (RoPE / GQA / KV / RMSNorm / projection)."
    )
    assert max_abs <= MAX_ABS_TOL and mean_abs <= MEAN_ABS_TOL, (
        f"logits divergence exceeds tolerance: max_abs={max_abs:.4f} "
        f"(tol {MAX_ABS_TOL}), mean_abs={mean_abs:.4f} (tol {MEAN_ABS_TOL})"
    )
