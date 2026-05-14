import sys

sys.path.append(
    r"D:\LLM projects\Axion\inference\cpp\build"
)

import axion_cpp

# -------------------------
# INPUT
# -------------------------

x = axion_cpp.Tensor()

x.shape = [2, 4]

x.data = [
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0
]

# -------------------------
# WEIGHT
# -------------------------

w = axion_cpp.Tensor()

w.shape = [4]

w.data = [1.0] * 4

# -------------------------
# BIAS
# -------------------------

b = axion_cpp.Tensor()

b.shape = [4]

b.data = [0.0] * 4

# -------------------------
# LAYERNORM
# -------------------------

out = axion_cpp.layernorm(
    x,
    w,
    b
)

print("\n=== LAYERNORM ===")

print(out.shape)

print(out.data)