import sys

sys.path.append(
    r"D:\LLM projects\Axion\inference\cpp\build"
)

import axion_cpp

loader = axion_cpp.GGUFLoader()

loader.load_file(
    "tests/inference/gguf/inferra-q4.gguf"
)

executor = axion_cpp.StreamingExecutor(
    loader
)

x = axion_cpp.create_owned_tensor(
    [1, 4096]
)


# x.data = [0.1] * 4096

output = executor.forward(
    x,
    2
)

print(output.shape)