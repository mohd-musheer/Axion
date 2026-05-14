import sys

sys.path.append(
    r"D:\LLM projects\Axion\inference\cpp\build"
)
import axion_cpp

cache = axion_cpp.KVCache()

# -------------------------
# STEP 1
# -------------------------

K1 = axion_cpp.create_owned_tensor(
    [1, 4]
)

K1.data = [
    1.0, 2.0, 3.0, 4.0
]

V1 = axion_cpp.create_owned_tensor(
    [1, 4]
)

V1.data = [
    5.0, 6.0, 7.0, 8.0
]

cache.add(
    K1,
    V1
)

# -------------------------
# STEP 2
# -------------------------

K2 = axion_cpp.create_owned_tensor(
    [1, 4]
)

K2.data = [
    9.0, 10.0, 11.0, 12.0
]

V2 = axion_cpp.create_owned_tensor(
    [1, 4]
)

V2.data = [
    13.0, 14.0, 15.0, 16.0
]

cache.add(
    K2,
    V2
)

# -------------------------
# GET CACHE
# -------------------------

all_k = cache.get_all_keys()

all_v = cache.get_all_values()

print("\n=== ALL KEYS ===")

print(all_k.shape)

print(all_k.data)

print("\n=== ALL VALUES ===")

print(all_v.shape)

print(all_v.data)
