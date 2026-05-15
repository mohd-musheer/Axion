import sys

sys.path.append(
    r"D:\LLM projects\Axion\inference\cpp\build"
)
import axion_cpp

model_path = r"D:\LLM projects\Axion\tests\inference\adapter_model.safetensors"

loader = axion_cpp.MMapLoader()

loader.load_file(model_path)

names = loader.list_tensors()

print("\n=== TOTAL TENSORS ===")

print(len(names))

print("\n=== FIRST 100 TENSORS ===\n")

for i, name in enumerate(names[:100]):

    print(f"{i+1}. {name}")
