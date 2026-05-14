import sys

sys.path.append(
    r"D:\LLM projects\Axion\inference\cpp\build"
)

import axion_cpp

arena = axion_cpp.Arena(
    1024
)

x = axion_cpp.create_tensor(
    arena,
    [2, 4],
    
)

tmp = x.data

for i in range(8):
    tmp[i] = float(i)

x.data = tmp

print("\nTENSOR:")
print(x.data)

arena.reset()

y = axion_cpp.create_tensor(
    arena,
    [2, 4]
)

print("\nSECOND TENSOR:")
print(y.data)