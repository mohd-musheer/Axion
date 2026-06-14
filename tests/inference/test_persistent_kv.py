import sys

sys.path.append(
    r"D:\LLM projects\Axion\inference\cpp\build"
)

import axion_cpp

# -------------------------
# KV STATE
# -------------------------

state = axion_cpp.KVState(2)

# -------------------------
# TOKEN 1
# -------------------------

K1 = axion_cpp.create_owned_tensor(
    [1, 4]
)

K1.data = [
    1, 2, 3, 4
]

V1 = axion_cpp.create_owned_tensor(
    [1, 4]
)

V1.data = [
    5, 6, 7, 8
]

axion_cpp.append_kv_cache(
    state.layers[0],
    K1,
    V1
)

# -------------------------
# TOKEN 2
# -------------------------

K2 = axion_cpp.create_owned_tensor(
    [1, 4]
)

K2.data = [
    9, 10, 11, 12
]

V2 = axion_cpp.create_owned_tensor(
    [1, 4]
)

V2.data = [
    13, 14, 15, 16
]

axion_cpp.append_kv_cache(
    state.layers[0],
    K2,
    V2
)

# -------------------------
# RESULTS
# -------------------------

print("\n=== KEYS ===")

print(
    state.layers[0].keys.shape
)

print(
    state.layers[0].keys.data
)

print("\n=== VALUES ===")

print(
    state.layers[0].values.shape
)

print(
    state.layers[0].values.data
)