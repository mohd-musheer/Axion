import sys
import numpy as np


sys.path.append(
    r"D:\LLM projects\Axion\inference\cpp\build"
)

import axion_cpp

x = axion_cpp.create_owned_tensor(
    [2, 4]
)

x.name = "input"

x.data = [
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0
]

weight = axion_cpp.create_owned_tensor(
    [4]
)

weight.data = [
    1.0,
    1.0,
    1.0,
    1.0
]

output = axion_cpp.rmsnorm(
    x,
    weight,
    1e-6
)

print("OUTPUT:")
print(output.data)


