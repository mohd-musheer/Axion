
import sys

sys.path.append(
    "inference/cpp/build"
)

import axion_cpp

loader = axion_cpp.MMapLoader()

path = (
    "models/cache/qwen2.5-3b/"
    "layers/layer_0.safetensors"
)

loader.load_file(path)

tensor = loader.load_tensor_data(
    "self_attn.q_proj.weight"
)

tensor.print_info()

print("\nFIRST 10 VALUES:\n")

for i in range(10):
    print(tensor.data[i])
