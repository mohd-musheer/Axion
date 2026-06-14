import sys

sys.path.append(
    r"D:\LLM projects\Axion\inference\cpp\build"
)
import axion_cpp

q = axion_cpp.create_owned_tensor(
    [1, 8]
)
k = axion_cpp.create_owned_tensor(
    [1, 8]
)

q.data = [
    1.0, 2.0,
    3.0, 4.0,
    5.0, 6.0,
    7.0, 8.0
]

k.data = [
    2.0, 1.0,
    4.0, 3.0,
    6.0, 5.0,
    8.0, 7.0
]

print("Before RoPE")
print(q.data)
print(k.data)

axion_cpp.apply_rope(
    q,
    k,
    5,
    8,
    10000.0
)

print("\nAfter RoPE")
print(q.data)
print(k.data)
