
import axion_cpp

MODEL_PATH = r"D:\LLM projects\Axion\tests\inference\adapter_model.safetensors"

loader = axion_cpp.MMapLoader()

loader.load_file(MODEL_PATH)

# -------------------------
# LOAD LORA MATRICES
# -------------------------

A = loader.load_tensor_data(
    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
)

B = loader.load_tensor_data(
    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
)

print("\n=== ORIGINAL SHAPES ===")

print("A shape:", A.shape)
print("B shape:", B.shape)

# -------------------------
# TRANSPOSE
# -------------------------

A_t = axion_cpp.transpose(A)
B_t = axion_cpp.transpose(B)

print("\n=== TRANSPOSED SHAPES ===")

print("A_t shape:", A_t.shape)
print("B_t shape:", B_t.shape)

# -------------------------
# PRINT SAMPLE VALUES
# -------------------------

print("\nA first values:")
print(A_t.data[:10])

print("\nB first values:")
print(B_t.data[:10])

# -------------------------
# COMPUTE DELTA WEIGHT
# -------------------------

delta = axion_cpp.matmul(
    B_t,
    A_t
)

print("\n=== DELTA WEIGHT ===")

print("Delta shape:", delta.shape)

print("Delta first values:")
print(delta.data[:20])
