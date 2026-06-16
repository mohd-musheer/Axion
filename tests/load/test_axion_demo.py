"""Axion end-to-end demo.

A single self-contained demonstration that Axion can, without manual
debugging:

    1. load a TinyLlama GGUF
    2. accept a text prompt
    3. generate output
    4. report performance

It prints, in order: prompt, generated tokens, generated text,
generation time, tokens/sec, and (when AXION_PROFILE=1) the per-layer
timings emitted by the runtime on stderr.

This is a demo + smoke test: it asserts the run succeeds and produces
tokens, then prints the report. Skips when the executable or model is
missing. Run with:

    pytest tests/load/test_axion_demo.py -v -s
    AXION_PROFILE=1 pytest tests/load/test_axion_demo.py -v -s
"""

from pathlib import Path
import os
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


def _extract_block(text: str, start: str, end: str):
    """Capture a labelled block that may span multiple lines (newlines in
    the decoded text), from ``start`` up to the next marker ``end``."""
    lines = text.splitlines()
    start_idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith(start)), None
    )
    if start_idx is None:
        return None
    collected = [lines[start_idx].split(":", 1)[1]]
    for ln in lines[start_idx + 1:]:
        if ln.startswith(end):
            break
        collected.append(ln)
    return "\n".join(collected)


def _field(stdout: str, marker: str) -> str:
    line = next(
        (ln for ln in stdout.splitlines() if ln.startswith(marker)), None
    )
    return "" if line is None else line.split(":", 1)[1].strip()


def test_axion_demo():
    _require(EXE, "axion_cli.exe")
    _require(MODEL, "TinyLlama GGUF model")

    # Forward AXION_PROFILE so per-layer timings appear on stderr.
    env = dict(os.environ)

    cmd = [
        str(EXE),
        "--model", str(MODEL),
        "--prompt", PROMPT,
        "--max-new", str(MAX_NEW),
        "--trace",
    ]
    result = subprocess.run(
        cmd, text=True, capture_output=True,
        timeout=TIMEOUT_SECONDS, env=env,
    )
    out = result.stdout
    err = result.stderr

    assert result.returncode == 0, (
        f"axion_cli exited {result.returncode}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
    )

    generated_tokens = _field(out, "generated tokens:")
    generated_text = _extract_block(out, "generated text:", "full text:")
    assert generated_text is not None, "no 'generated text:' section:\n" + out
    assert len(generated_text.strip()) > 0, "empty generated text:\n" + out

    m_time = re.search(r"generation time:\s*([0-9.]+)\s*s", err)
    m_tps = re.search(r"tokens/sec:\s*([0-9.]+)", err)
    gen_time = float(m_time.group(1)) if m_time else float("nan")
    tokens_per_sec = float(m_tps.group(1)) if m_tps else float("nan")

    # ---- Demo report ----
    print("\n========== AXION DEMO ==========")
    print(f"1. prompt:           {PROMPT!r}")
    print(f"2. generated tokens: {generated_tokens}")
    print(f"3. generated text:   {generated_text!r}")
    print(f"4. generation time:  {gen_time:.4f} s")
    print(f"5. tokens/sec:       {tokens_per_sec:.4f}")

    profile_enabled = (
        env.get("AXION_PROFILE") not in (None, "", "0")
    )
    if profile_enabled:
        layer_lines = [
            ln for ln in err.splitlines()
            if ln.strip().startswith("Layer ")
            or ln.strip().startswith(("load=", "dequant=", "transpose=",
                                      "attention=", "ffn=", "total="))
        ]
        print("6. per-layer timings (AXION_PROFILE=1):")
        if layer_lines:
            for ln in layer_lines:
                print("   " + ln.strip())
        else:
            print("   (no per-layer timings found on stderr)")
    else:
        print("6. per-layer timings: set AXION_PROFILE=1 to enable")
