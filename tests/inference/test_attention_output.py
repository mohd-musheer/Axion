
import axion_cpp

# -------------------------
# ATTENTION PROBABILITIES
# -------------------------

probs = axion_cpp.Tensor()

probs.name = "attention_probs"

probs.shape = [2, 2]

probs.data = [
    0.00407,
    0.99593,

    0.000075,
    0.999925
]

# -------------------------
# VALUE MATRIX
# -------------------------

V = axion_cpp.Tensor()

V.name = "V"

V.shape = [2, 4]

V.data = [
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0
]

# -------------------------
# COMPUTE ATTENTION OUTPUT
# -------------------------

output = axion_cpp.attention_output(
    probs,
    V
)

print("\n=== ATTENTION OUTPUT ===")

print("Shape:", output.shape)

print("\nValues:")

for v in output.data:
    print(round(v, 4))