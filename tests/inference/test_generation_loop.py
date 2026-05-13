
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CPP_BUILD = (
    ROOT /
    "inference" /
    "cpp" /
    "build"
)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CPP_BUILD))

from inference.python.generator import (
    AxionGenerator
)

model_path = (
    r"D:\LLM projects\Axion"
    r"\tests\inference\model.safetensors"
)

gen = AxionGenerator(
    model_path=model_path,
    tokenizer_name="gpt2"
)

output = gen.generate(
    prompt="Hello Axion",
    max_new_tokens=10
)

print("\n=== FINAL TEXT ===\n")

print(output)
