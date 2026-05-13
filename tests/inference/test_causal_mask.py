
import axion_cpp

scores = axion_cpp.Tensor()

scores.name = "attention_scores"

scores.shape = [3, 3]

scores.data = [

    1.0, 2.0, 3.0,

    4.0, 5.0, 6.0,

    7.0, 8.0, 9.0
]

masked = axion_cpp.causal_mask(
    scores
)

print("\n=== CAUSAL MASK ===")

print("Shape:", masked.shape)

rows = masked.shape[0]
cols = masked.shape[1]

for r in range(rows):

    row = []

    for c in range(cols):

        v = masked.data[
            r * cols + c
        ]

        row.append(v)

    print(row)
