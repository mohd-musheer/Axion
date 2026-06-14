
import axion_cpp

model_path = r"D:\LLM projects\Axion\tests\inference\adapter_model.safetensors"

x = axion_cpp.Tensor()

x.shape = [2, 8]

x.data = [

    1,2,3,4,5,6,7,8,

    9,10,11,12,13,14,15,16
]

output = axion_cpp.execute_model(
    model_path,
    x
)

print("\n=== FINAL OUTPUT ===")

print(output.shape)

print(output.data)
