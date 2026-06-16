"""End-to-end text-prompt test using Axion's in-process GGUF tokenizer.

Exercises the full developer path: text prompt -> embedded tokenizer
encode -> streaming forward + generation -> embedded tokenizer decode ->
text output. No third-party tokenizer (transformers) is involved; the
GGUF file is the single source of truth.

Skips when the executable or model is missing so CI without the weights
does not fail. Run with:

    pytest tests/load/test_tinyllama_text_prompt.py -v -s
"""

from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "build" / "axion_cli.exe"
MODEL = ROOT / "models" / "gguf" / "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"

PROMPT = "What is Python?"
MAX_NEW = "16"
TIMEOUT_SECONDS = 1800


def _require(path: Path, what: str) -> None:
    if not path.exists():
        pytest.skip(f"{what} not present: {path}")


def _extract_block(text: str, start: str, end: str):
    """Return the text of a labelled block that may span multiple lines.

    The CLI prints the decoded output inline after ``start`` (e.g.
    ``generated text: ...``). Because the model can emit newline tokens,
    the decoded text continues onto following lines until the next
    labelled marker (``end``, e.g. ``full text:``) or end of output.

    Returns the content after ``start`` (including any subsequent lines
    up to ``end``), or ``None`` if the start marker is absent.
    """
    lines = text.splitlines()
    start_idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith(start)), None
    )
    if start_idx is None:
        return None

    # Content on the marker line itself (after the colon).
    first = lines[start_idx].split(":", 1)[1]
    collected = [first]

    for ln in lines[start_idx + 1:]:
        if ln.startswith(end):
            break
        collected.append(ln)

    return "\n".join(collected)


def _run():
    cmd = [
        str(EXE),
        "--model", str(MODEL),
        "--prompt", PROMPT,
        "--max-new", MAX_NEW,
        "--trace",
    ]
    return subprocess.run(
        cmd, text=True, capture_output=True, timeout=TIMEOUT_SECONDS
    )


def test_text_prompt_generates_text():
    _require(EXE, "axion_cli.exe")
    _require(MODEL, "TinyLlama GGUF model")

    result = _run()
    out = result.stdout
    err = result.stderr
    combined = out + "\n" + err

    if result.returncode != 0:
        print("=== STDOUT ===\n" + out)
        print("=== STDERR ===\n" + err)

    # Clean exit, no regression to the RMSNorm crash.
    assert "rmsnorm weight mismatch" not in combined, combined
    assert result.returncode == 0, (
        f"axion_cli exited {result.returncode}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
    )

    # The embedded tokenizer must have encoded the prompt to >1 token
    # (BOS + content), proving text->ids worked in-process.
    prompt_line = next(
        (ln for ln in out.splitlines() if ln.startswith("prompt tokens:")),
        None,
    )
    assert prompt_line is not None, "no 'prompt tokens:' line:\n" + out
    prompt_ids = prompt_line.split(":", 1)[1].split()
    assert len(prompt_ids) >= 2, (
        "tokenizer produced too few tokens (encode failed?):\n" + out
    )

    # A non-empty 'generated text:' section proves decode (ids->text)
    # worked. The decoded text may contain newline tokens, so it spans
    # multiple lines: capture from the 'generated text:' marker up to the
    # next known marker ('full text:'), not just the marker line itself.
    generated_text = _extract_block(
        out, start="generated text:", end="full text:"
    )
    assert generated_text is not None, (
        "no 'generated text:' section:\n" + out
    )
    assert len(generated_text.strip()) > 0, (
        "forward pass completed but produced empty text:\n" + out
    )

    # Trace mode must report throughput.
    assert "tokens/sec:" in err, "trace timing missing:\n" + err

    print("\nPrompt:", PROMPT)
    print("Generated text:", generated_text)
