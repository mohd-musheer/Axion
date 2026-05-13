import sys

sys.path.append(
    r"D:\LLM projects\Axion\inference\cpp\build"
)

import axion_cpp

loader = axion_cpp.MMapLoader()

loader.load_file(
    r"D:\LLM projects\Axion\tests\inference\model.safetensors"
)

names = loader.list_tensors()

print("\n=== POSITION TENSORS ===\n")

for n in names:

    if "wpe" in n:

        print(n)