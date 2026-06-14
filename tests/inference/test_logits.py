
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CPP_BUILD = (
    ROOT /
    "inference" /
    "cpp" /
    "build"
)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CPP_BUILD))

import axion_cpp

from inference.python.tokenizer import (
    AxionTokenizer
)

tok = AxionTokenizer(
    "gpt2"
)

model_path = (
    r"D:\LLM projects\Axion"
    r"\tests\inference\adapter_model.safetensors"
)

loader = axion_cpp.MMapLoader()

loader.load_file(model_path)

# -----------------------------------
# LOAD EMBEDDINGS
# -----------------------------------

embedding_matrix = (
    loader.load_tensor_data(
        "wte.weight"
    )
)

# -----------------------------------
# TOKENIZE
# -----------------------------------

text = "Hello Axion"

token_ids = tok.encode(text)

print("\nTOKENS:")

print(token_ids)

# -----------------------------------
# EMBEDDINGS
# -----------------------------------

hidden = axion_cpp.embedding_lookup(
    embedding_matrix,
    token_ids
)

print("\nHIDDEN SHAPE:")

print(hidden.shape)

# -----------------------------------
# LOGITS
# -----------------------------------

logits = axion_cpp.compute_logits(
    hidden,
    embedding_matrix
)

print("\nLOGITS SHAPE:")

print(logits.shape)

# -----------------------------------
# NEXT TOKEN
# -----------------------------------

next_token = axion_cpp.argmax(
    logits
)

print("\nNEXT TOKEN ID:")

print(next_token)

decoded = tok.decode(
    [next_token]
)

print("\nPREDICTED TOKEN:")

print(decoded)
