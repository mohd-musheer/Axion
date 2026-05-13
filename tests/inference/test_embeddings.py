
import axion_cpp

model_path = r"D:\LLM projects\Axion\tests\inference\model.safetensors"

loader = axion_cpp.MMapLoader()

loader.load_file(model_path)

# GPT2 embedding matrix

embedding_name = "wte.weight"

embedding_matrix = loader.load_tensor_data(
    embedding_name
)

print("\n=== EMBEDDING MATRIX ===")

print(embedding_matrix.shape)

# tokens:
# Hello Axion

token_ids = [
    15496,
    12176,
    295
]

hidden = axion_cpp.embedding_lookup(
    embedding_matrix,
    token_ids
)

print("\n=== EMBEDDING OUTPUT ===")

print(hidden.shape)

print(hidden.data[:20])
