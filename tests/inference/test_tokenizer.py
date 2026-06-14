
from inference.python.tokenizer import AxionTokenizer
from huggingface_hub import login
login("hf_****************")
tok = AxionTokenizer(
    "openai-community/gpt2"
)

text = "Hello Axion"

ids = tok.encode(text)

print("\n=== TOKEN IDS ===")

print(ids)

decoded = tok.decode(ids)

print("\n=== DECODED ===")

print(decoded)

print("\n=== VOCAB SIZE ===")

print(tok.vocab_size)
