import sys

sys.path.append(
    r"D:\LLM projects\Axion\inference\cpp\build"
)
import axion_cpp

Q = axion_cpp.create_owned_tensor(
    [2, 4]
)
Q.name = "Q"

Q.data = [
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0
]

K = axion_cpp.create_owned_tensor(
    [2, 4]
)
K.name = "K"

K.data = [
    2.0, 1.0, 0.0, 1.0,
    1.0, 0.0, 2.0, 3.0
]

scores = axion_cpp.attention_scores(
    Q,
    K
)

print("\n=== ATTENTION SCORES ===")

print("Shape:", scores.shape)

print("\nValues:")

for v in scores.data:
    print(round(v, 4))
