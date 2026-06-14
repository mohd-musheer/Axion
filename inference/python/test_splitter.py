from safetensors.torch import load_file

path = "models/cache/qwen2.5-3b/layers/layer_0.safetensors"

weights = load_file(path)

print(weights.keys())