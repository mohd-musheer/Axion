
import axion_cpp

x = axion_cpp.Tensor()

x.name = "scores"

x.shape = [2, 2]

x.data = [
    4.0,
    9.5,

    12.0,
    21.5
]

output = axion_cpp.softmax(
    x
)

print("\n=== SOFTMAX OUTPUT ===")

print("Shape:", output.shape)

print("\nValues:")

for v in output.data:
    print(round(v, 6))
