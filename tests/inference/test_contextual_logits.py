
import sys

sys.path.append(
    r"D:\LLM projects\Axion\inference\cpp\build"
)

import axion_cpp

from inference.python.tokenizer import (
    AxionTokenizer
)

# -------------------------
# TOKENIZER
# -------------------------

tok = AxionTokenizer(
    "gpt2"
)

# -------------------------
# MODEL
# -------------------------

loader = axion_cpp.MMapLoader()

loader.load_file(
    r"D:\LLM projects\Axion\tests\inference\model.safetensors"
)

# -------------------------
# LOAD MATRICES
# -------------------------

embedding_matrix = (
    loader.load_tensor_data(
        "wte.weight"
    )
)

position_matrix = (
    loader.load_tensor_data(
        "wpe.weight"
    )
)

# -------------------------
# TOKENS
# -------------------------

text = "Hello Axion"

token_ids = tok.encode(text)

print("\nTOKENS:")

print(token_ids)

# -------------------------
# TOKEN EMBEDDINGS
# -------------------------

embeddings = (
    axion_cpp.embedding_lookup(
        embedding_matrix,
        token_ids
    )
)

# -------------------------
# POSITION EMBEDDINGS
# -------------------------

position_embeddings = (
    axion_cpp.position_embedding_lookup(
        position_matrix,
        len(token_ids)
    )
)

# -------------------------
# COMBINE
# -------------------------

embeddings = (
    axion_cpp.add(
        embeddings,
        position_embeddings
    )
)

print("\nEMBED SHAPE:")

print(embeddings.shape)

# -------------------------
# FULL FORWARD
# -------------------------

logits = axion_cpp.full_forward(
    embeddings,
    loader,
    2,
    12,
    embedding_matrix
)

print("\nLOGITS SHAPE:")

print(logits.shape)

# -------------------------
# LAST TOKEN
# -------------------------

last_logits = (
    axion_cpp.last_token(
        logits
    )
)

# -------------------------
# NEXT TOKEN
# -------------------------

next_token = (
    axion_cpp.argmax(
        last_logits
    )
)

print("\nNEXT TOKEN:")

print(next_token)

decoded = tok.decode(
    [next_token]
)

print("\nPREDICTED:")

print(decoded)
