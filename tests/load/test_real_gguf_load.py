import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

MODEL = ROOT / "models" / "gguf" / "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"

EXE = ROOT / "build" / "axion_runtime_validation.exe"


def test_real_gguf_load():

    assert EXE.exists(), f"Missing executable: {EXE}"

    assert MODEL.exists(), f"Missing model: {MODEL}"

    result = subprocess.run(
        [str(EXE)],
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr

    assert result.returncode == 0

    assert "AXION RUNTIME VALIDATION PASSED" in output

    print(output)