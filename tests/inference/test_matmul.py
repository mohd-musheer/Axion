import sys
sys.path.append(
    "../../inference/cpp/build"
)

import axion_cpp

# Create tensor A

A = axion_cpp.Tensor()

A.name = "A"

A.shape = [2, 3]

A.data = [
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0
]

# Create tensor B

B = axion_cpp.Tensor()

B.name = "B"

B.shape = [3, 2]

B.data = [
    7.0, 8.0,
    9.0, 10.0,
    11.0, 12.0
]

# Run matmul

C = axion_cpp.matmul(A, B)

print("Output Shape:")
print(C.shape)

print("\nOutput Data:")
print(C.data)




# PS D:\LLM projects\Axion\inference\cpp\build> python -u "d:\LLM projects\Axion\tests\inference\test_matmul.py"
# Output Shape:
# [2, 2]

# Output Data:
# [58.0, 64.0, 139.0, 154.0]
# PS D:\LLM projects\Axion\inference\cpp\build> 

