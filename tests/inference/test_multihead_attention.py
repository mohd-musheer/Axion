import sys

sys.path.append(
    r"D:\LLM projects\Axion\inference\cpp\build"
)
import axion_cpp

Q = axion_cpp.create_owned_tensor(
    [2, 8]
)

Q.data = [

    1,2,3,4,5,6,7,8,

    9,10,11,12,13,14,15,16
]

K = axion_cpp.create_owned_tensor(
    [2, 8]
)

K.data = [

    2,1,0,1,1,0,2,3,

    3,2,1,0,4,3,2,1
]

V = axion_cpp.create_owned_tensor(
    [2, 8]
)

V.data = [

    1,1,1,1,2,2,2,2,

    3,3,3,3,4,4,4,4
]

output = axion_cpp.multihead_attention(
    Q,
    K,
    V,
    2
)

print("\n=== MULTIHEAD OUTPUT ===")

print(output.shape)

print(output.data)