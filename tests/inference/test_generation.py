import sys

sys.path.append(
    r"D:\LLM projects\Axion\inference\cpp\build"
)
import axion_cpp

x = axion_cpp.create_owned_tensor(
    [2, 4]
)

x.name = "input_hidden"

x.shape = [2, 4]

x.data = [
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0
]

output = axion_cpp.generate_tokens(
    x,
    5
)

print("\n=== FINAL GENERATED STATE ===")

print("Shape:", output.shape)

print("\nValues:")

for v in output.data:
    print(round(v, 4))
