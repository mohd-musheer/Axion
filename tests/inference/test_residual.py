
import axion_cpp

x = axion_cpp.Tensor()

x.name = "hidden_state"

x.shape = [2, 4]

x.data = [
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0
]

attention = axion_cpp.Tensor()

attention.name = "attention_output"

attention.shape = [2, 4]

attention.data = [
    4.9837, 5.9837, 6.9837, 7.9837,
    4.9997, 5.9997, 6.9997, 7.9997
]

output = axion_cpp.residual_add(
    x,
    attention
)

print("\n=== RESIDUAL OUTPUT ===")

print("Shape:", output.shape)

print("\nValues:")

for v in output.data:
    print(round(v, 4))
