import sys

sys.path.append(
    r"D:\LLM projects\Axion\inference\cpp\build"
)

import axion_cpp

# -------------------------
# MODEL
# -------------------------

loader = axion_cpp.MMapLoader()

loader.load_file(
    r"D:\LLM projects\Axion\tests\inference\model.safetensors"
)

# -------------------------
# CACHE
# -------------------------

cache = axion_cpp.LayerKVCache()

# -------------------------
# TOKEN 1
# -------------------------

x1 = axion_cpp.Tensor()

x1.shape = [1, 768]

x1.data = [0.1] * 768

out1 = axion_cpp.cached_attention(
    x1,
    loader,
    cache,
    "h.0.attn.c_attn.weight",
    12
)

print("\nSTEP 1")

print("CACHE KEYS:")

print(cache.keys.shape)

# -------------------------
# TOKEN 2
# -------------------------

x2 = axion_cpp.Tensor()

x2.shape = [1, 768]

x2.data = [0.2] * 768

out2 = axion_cpp.cached_attention(
    x2,
    loader,
    cache,
    "h.0.attn.c_attn.weight",
    12
)


print("\nSTEP 2")

print("CACHE KEYS:")

print(cache.keys.shape)

print("\nOUTPUT SHAPE:")

print(out2.shape)