
import axion_cpp

model_path = r"D:\LLM projects\Axion\tests\inference\model.safetensors"

loader = axion_cpp.MMapLoader()

loader.load_file(model_path)

# -----------------------------------
# LOAD LORA A
# -----------------------------------

A_name = (
    "base_model.model.model.layers.0."
    "self_attn.q_proj.lora_A.weight"
)

A = loader.load_tensor_data(
    A_name
)

# -----------------------------------
# LOAD LORA B
# -----------------------------------

B_name = (
    "base_model.model.model.layers.0."
    "self_attn.q_proj.lora_B.weight"
)

B = loader.load_tensor_data(
    B_name
)

print("\n=== LORA A ===")

print(A.shape)

print(A.data[:10])

print("\n=== LORA B ===")

print(B.shape)

print(B.data[:10])

# -----------------------------------
# TRANSPOSE
# -----------------------------------

A_t = axion_cpp.transpose(
    A
)

B_t = axion_cpp.transpose(
    B
)

print("\n=== TRANSPOSED ===")

print("A_t:", A_t.shape)

print("B_t:", B_t.shape)

# -----------------------------------
# DELTA W = B @ A
# -----------------------------------

delta = axion_cpp.matmul(
    B_t,
    A_t
)

print("\n=== DELTA ===")

print(delta.shape)

print(delta.data[:20])

# -----------------------------------
# INPUT HIDDEN STATE
# -----------------------------------

x = axion_cpp.Tensor()

x.shape = [2, 8]

x.data = [

    1,2,3,4,5,6,7,8,

    9,10,11,12,13,14,15,16
]

# -----------------------------------
# REAL PROJECTION
# -----------------------------------

Q = axion_cpp.matmul(
    x,
    delta
)

print("\n=== REAL Q PROJECTION ===")

print(Q.shape)

print(Q.data[:20])
