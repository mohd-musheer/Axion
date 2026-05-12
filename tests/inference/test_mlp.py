
import axion_cpp

gate = axion_cpp.Tensor()

gate.shape = [2, 4]

gate.data = [
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0
]

up = axion_cpp.Tensor()

up.shape = [2, 4]

up.data = [
    2.0, 2.0, 2.0, 2.0,
    3.0, 3.0, 3.0, 3.0
]

output = axion_cpp.mlp_block(
    gate,
    up
)

print("\n=== MLP OUTPUT ===")

print("Shape:", output.shape)

print("\nValues:")

for v in output.data:
    print(round(v, 4))
