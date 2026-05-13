import sys

sys.path.append(
    r"D:\LLM projects\Axion\inference\cpp\build"
)

import axion_cpp

# -------------------------
# LOAD MODEL
# -------------------------

loader = axion_cpp.MMapLoader()

loader.load_file(
    r"D:\LLM projects\Axion\tests\inference\model.safetensors"
)

# -------------------------
# INPUT
# -------------------------

x = axion_cpp.Tensor()

x.shape = [2, 768]

x.data = [0.01] * (2 * 768)

# -------------------------
# STACK
# -------------------------

output = axion_cpp.transformer_stack(
    x,
    loader,
    2,
    12
)

print("\n=== FINAL STACK OUTPUT ===")

print(output.shape)

print(output.data[:20])