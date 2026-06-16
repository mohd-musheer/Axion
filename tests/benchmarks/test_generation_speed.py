"""Generation-speed benchmark for the Axion CPU runtime.

Drives axion_cli end-to-end against a real TinyLlama GGUF and reports:

  * prompt length      (tokens)
  * generated length   (tokens)
  * generation time    (seconds)
  * tokens/sec

This is a reporting benchmark, not a hard performance gate: it asserts
only that generation runs and produces tokens, then prints the metrics.
No benchmark framework is used; timing comes from the CLI's own --trace
output. Skips when the executable or model is missing.

Run with:
    pytest tests/benchmarks/test_generation_speed.py -v -s
"""

from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "build" / "axion_cli.exe"
MODEL = ROOT / "models" / "gguf" / "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"

PROMPT = "What is Python?"
MAX_NEW = 16
TIMEOUT_SECONDS = 1800


def _require(path: Path, what: str) -> None:
    if not path.exists():
        pytest.skip(f"{what} not present: {path}")


def _run():
    cmd = [
        str(EXE),
        "--model", str(MODEL),
        "--prompt", PROMPT,
        "--max-new", str(MAX_NEW),
        "--trace",
    ]
    return subprocess.run(
        cmd, text=True, capture_output=True, timeout=TIMEOUT_SECONDS
    )


def _count_ids(stdout: str, marker: str) -> int:
    line = next(
        (ln for ln in stdout.splitlines() if ln.startswith(marker)), None
    )
    if line is None:
        return 0
    return len(line.split(":", 1)[1].split())


def test_generation_speed():
    _require(EXE, "axion_cli.exe")
    _require(MODEL, "TinyLlama GGUF model")

    result = _run()
    out = result.stdout
    err = result.stderr
    assert result.returncode == 0, (
        f"axion_cli exited {result.returncode}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
    )

    prompt_len = _count_ids(out, "prompt tokens:")
    generated_len = _count_ids(out, "generated tokens:")
    assert prompt_len >= 2, "tokenizer produced too few prompt tokens:\n" + out
    assert generated_len > 0, "no tokens generated:\n" + out

    gen_time = None
    tokens_per_sec = None
    m_time = re.search(r"generation time:\s*([0-9.]+)\s*s", err)
    m_tps = re.search(r"tokens/sec:\s*([0-9.]+)", err)
    if m_time:
        gen_time = float(m_time.group(1))
    if m_tps:
        tokens_per_sec = float(m_tps.group(1))

    assert gen_time is not None, "trace missing generation time:\n" + err
    assert tokens_per_sec is not None, "trace missing tokens/sec:\n" + err

    print("\n=== Axion generation benchmark ===")
    print(f"prompt:           {PROMPT!r}")
    print(f"prompt length:    {prompt_len} tokens")
    print(f"generated length: {generated_len} tokens")
    print(f"generation time:  {gen_time:.4f} s")
    print(f"tokens/sec:       {tokens_per_sec:.4f}")
