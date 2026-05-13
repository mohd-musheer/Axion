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
# -----------------------------------
# LOAD FUSED GPT2 QKV
# -----------------------------------

fused = loader.load_tensor_data(
    "h.0.attn.c_attn.weight"
)

print("\nFUSED SHAPE:")
print(fused.shape)

# -----------------------------------
# SPLIT
# -----------------------------------

qkv = axion_cpp.split_fused_qkv(
    fused
)

print("\nQ SHAPE:")
print(qkv.Q.shape)

print("\nK SHAPE:")
print(qkv.K.shape)

print("\nV SHAPE:")
print(qkv.V.shape)

# -----------------------------------
# INPUT
# -----------------------------------

x = axion_cpp.Tensor()

x.shape = [2, 768]

x.data = [0.01] * (2 * 768)

# -----------------------------------
# TRANSPOSE
# -----------------------------------

Qw = axion_cpp.transpose(qkv.Q)
Kw = axion_cpp.transpose(qkv.K)
Vw = axion_cpp.transpose(qkv.V)

# -----------------------------------
# PROJECTIONS
# -----------------------------------

Q = axion_cpp.matmul(x, Qw)
K = axion_cpp.matmul(x, Kw)
V = axion_cpp.matmul(x, Vw)

print("\nQ PROJECTION:")
print(Q.shape)

# -----------------------------------
# ATTENTION
# -----------------------------------

output = axion_cpp.multihead_attention(
    Q,
    K,
    V,
    12
)

print("\n=== REAL ATTENTION OUTPUT ===")

print(output.shape)

print(output.data[:20])