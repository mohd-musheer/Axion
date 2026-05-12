
import axion_cpp

Q = axion_cpp.Tensor()
Q.name = "Q"
Q.shape = [2, 4]

Q.data = [
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0
]

K = axion_cpp.Tensor()
K.name = "K"
K.shape = [2, 4]

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
