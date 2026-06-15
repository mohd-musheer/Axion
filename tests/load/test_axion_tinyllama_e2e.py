"""End-to-end load test: run the real Axion CLI on a real TinyLlama GGUF.

This is a true integration test. It does NOT use a stub: it invokes the
compiled ``build/axion_cli.exe`` against the real
``models/gguf/tinyllama-1.1b-chat-v1.0.Q8_0.gguf`` model with explicit
token ids and asserts the runtime:

  * loads the GGUF,
  * reports a non-zero vocab in its ModelConfig line,
  * runs the streaming forward + generation pass,
  * exits successfully (no ``rmsnorm weight mismatch`` / no throw).

If the executable or model is missing the test is skipped (so CI without
the weights does not fail), but when both are present any runtime error
fails the test and the full stdout/stderr is printed for diagnosis.

Run with:  pytest tests/load/test_axion_tinyllama_e2e.py -v
"""

from pathlib import Path
import re
import subprocess

import pytest

# Repo root is two levels up from tests/load/.
ROOT = Path(__file__).resolve().parents[2]

EXE = ROOT / "build" / "axion_cli.exe"
MODEL = ROOT / "models" / "gguf" / "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"

# Explicit TinyLlama token ids (BOS + a short prompt). Token-id space is
# deliberate: the C++ runtime works in id space and embeds no tokenizer.
PROMPT_TOKENS = ["1", "15043", "29892", "590", "1024", "338"]

# Keep the run short so the test is fast but still exercises prefill +
# at least one autoregressive decode step.
MAX_NEW = "4"

# Generous: a 1.1B model streaming layer-by-layer on CPU is not fast.
TIMEOUT_SECONDS = 1800


def _require(path: Path, what: str) -> None:
    if not path.exists():
        pytest.skip(f"{what} not present: {path}")


def _run_cli() -> subprocess.CompletedProcess:
    cmd = [
        str(EXE),
        "--model", str(MODEL),
        "--tokens", *PROMPT_TOKENS,
        "--max-new", MAX_NEW,
    ]
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=TIMEOUT_SECONDS,
    )


def test_tinyllama_loads_and_runs_end_to_end():
    _require(EXE, "axion_cli.exe")
    _require(MODEL, "TinyLlama GGUF model")

    result = _run_cli()
    combined = result.stdout + "\n" + result.stderr

    # Always surface output on failure for diagnosis.
    if result.returncode != 0:
        print("=== STDOUT ===\n" + result.stdout)
        print("=== STDERR ===\n" + result.stderr)

    # 1. The runtime must not regress to the original RMSNorm crash.
    assert "rmsnorm weight mismatch" not in combined, (
        "RMSNorm dimension mismatch returned:\n" + combined
    )

    # 2. The process must exit cleanly.
    assert result.returncode == 0, (
        f"axion_cli exited with {result.returncode}.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    # 3. The GGUF must actually load.
    assert "GGUF VERSION:" in combined, (
        "GGUF header was never parsed:\n" + combined
    )
    assert "Parsed tensor directory" in combined, (
        "Tensor directory was never parsed:\n" + combined
    )

    # 4. ModelConfig must report a valid (non-zero) vocab.
    m = re.search(r"vocab=(\d+)", combined)
    assert m is not None, "ModelConfig line with vocab= not found:\n" + combined
    vocab = int(m.group(1))
    assert vocab > 0, f"vocab resolved to {vocab} (expected > 0):\n{combined}"
    # TinyLlama-1.1B uses the LLaMA 32000-token vocabulary.
    assert vocab == 32000, f"unexpected vocab {vocab}:\n{combined}"

    # 5. Generation must actually have produced tokens.
    gen = re.search(r"generated tokens:(.*)", combined)
    assert gen is not None, "no 'generated tokens:' line:\n" + combined
    produced = gen.group(1).split()
    assert len(produced) >= 1, (
        "forward pass completed but produced no tokens:\n" + combined
    )
