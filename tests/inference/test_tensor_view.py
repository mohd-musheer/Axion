import sys

sys.path.append(
    r"D:\LLM projects\Axion\inference\cpp\build"
)

import axion_cpp

x = axion_cpp.Tensor()

x.name = "base"

x.shape = [8]

x.data = [
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0
]

view = axion_cpp.tensor_view(
    x,
    2,
    [4]
)

print("\nBASE:")
print(x.data)

print("\nVIEW:")
print(view.data)