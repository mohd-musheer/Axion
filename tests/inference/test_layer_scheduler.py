
import axion_cpp

tensor_names = [

    "model.layers.0.self_attn.q_proj.weight",

    "model.layers.0.self_attn.k_proj.weight",

    "model.layers.1.self_attn.q_proj.weight",

    "model.layers.1.self_attn.v_proj.weight",

    "model.layers.10.mlp.up_proj.weight"
]

layers = axion_cpp.discover_layers(
    tensor_names
)

print("\n=== DISCOVERED LAYERS ===")

for l in layers:
    print(l)
