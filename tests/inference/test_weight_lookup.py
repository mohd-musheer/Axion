
import axion_cpp

tensor_names = [

    "model.layers.0.self_attn.q_proj.weight",

    "model.layers.0.self_attn.k_proj.weight",

    "model.layers.0.self_attn.v_proj.weight",

    "model.layers.1.self_attn.q_proj.weight",

    "model.layers.1.mlp.up_proj.weight"
]

q_proj = axion_cpp.find_tensor_by_suffix(
    tensor_names,
    "layers.0",
    "q_proj"
)

k_proj = axion_cpp.find_tensor_by_suffix(
    tensor_names,
    "layers.0",
    "k_proj"
)

mlp = axion_cpp.find_tensor_by_suffix(
    tensor_names,
    "layers.1",
    "up_proj"
)

print("\n=== FOUND TENSORS ===")

print(q_proj)

print(k_proj)

print(mlp)
