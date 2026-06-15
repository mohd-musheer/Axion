from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "build" / "axion_cli.exe"
MODEL = ROOT / "models" / "gguf" / "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
TOKENS = ["1", "15043", "29892", "590", "1024", "338"]

def main() -> int:
    if not EXE.exists():
        print(f"Missing executable: {EXE}", file=sys.stderr)
        return 2
    if not MODEL.exists():
        print(f"Missing model: {MODEL}", file=sys.stderr)
        return 2

    cmd = [
        str(EXE),
        "--model", str(MODEL),
        "--tokens", *TOKENS,
        "--max-new", "32",
    ]

    result = subprocess.run(cmd, text=True, capture_output=True)
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    return result.returncode

if __name__ == "__main__":
    raise SystemExit(main())