import sys

sys.path.append(
    r"D:\LLM projects\Axion\inference\cpp\build"
)

import axion_cpp

cache = axion_cpp.PagedKVCache()

cache.initialize(
    4,
    2
)

for i in range(5):

    k = axion_cpp.create_owned_tensor([1, 4])
    v = axion_cpp.create_owned_tensor([1, 4])

    k.data = [
        i, i, i, i
    ]

    v.data = [
        i + 100,
        i + 100,
        i + 100,
        i + 100
    ]

    cache.append(k, v)

keys = cache.materialize_keys()

values = cache.materialize_values()

print("\nKEYS:")
print(keys.shape)
print(keys.data)

print("\nVALUES:")
print(values.shape)
print(values.data)