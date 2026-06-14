import sys

sys.path.append(
    r"D:\LLM projects\Axion\inference\cpp\build"
)

import axion_cpp

arena = axion_cpp.Arena(
    1024 * 1024
)

A = axion_cpp.create_tensor(
    arena,
    [2, 3]
)

B = axion_cpp.create_tensor(
    arena,
    [3, 2]
)

A.data = [
    1, 2, 3,
    4, 5, 6
]

B.data = [
    7, 8,
    9, 10,
    11, 12
]

C = axion_cpp.matmul_arena(
    A,
    B,
    arena
)

print("\nOUTPUT:")
print(C.shape)
print(C.data)