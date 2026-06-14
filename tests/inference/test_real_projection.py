
import axion_cpp

model_path = r"D:\LLM projects\Axion\tests\inference\adapter_model.safetensors"

loader = axion_cpp.MMapLoader()

loader.load_file(model_path)

tensor_name = (
    "base_model.model.model.layers.0."
    "self_attn.q_proj.lora_A.weight"
)

W = loader.load_tensor_data(
    tensor_name
)

print("\n=== REAL WEIGHT ===")

print("Shape:", W.shape)

print("Numel:", len(W.data))

print("\nFirst values:")

print(W.data[:20])
