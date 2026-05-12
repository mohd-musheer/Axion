
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

sys.path.append(
    str(ROOT / "inference/cpp/build")
)

import axion_cpp

loader = axion_cpp.MMapLoader()

model_path = (
    ROOT
    / "tests"
    / "inference"
    / "adapter_model.safetensors"
)

print(f"\nLoading file:\n{model_path}\n")

success = loader.load_file(
    str(model_path)
)

if not success:

    raise RuntimeError(
        "Failed to load safetensors file"
    )

print("Safetensors loaded successfully.\n")

names = loader.list_tensors()

print(f"Total tensors: {len(names)}\n")

print("FIRST 30 TENSORS:\n")

for i, n in enumerate(names[:30]):

    print(f"{i+1}. {n}")
