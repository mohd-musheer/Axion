import sys
import numpy as np

sys.path.append("../../inference/cpp/build")

import axion_cpp

x = axion_cpp.Tensor()

x.name = "input"

x.shape = [2, 4]

x.data = [
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0
]

weight = axion_cpp.Tensor()

weight.name = "weight"

weight.shape = [4]

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


# output :
    
#     PS D:\LLM projects\Axion\inference\cpp\build> python -u "d:\LLM projects\Axion\tests\inference\test_rmsnorm.py"
# OUTPUT:
# [0.3651483654975891, 0.7302967309951782, 1.095445156097412, 1.4605934619903564, 0.7580980658531189, 0.9097176790237427, 1.0613372325897217, 1.2129569053649902]
# PS D:\LLM projects\Axion\inference\cpp\build> 