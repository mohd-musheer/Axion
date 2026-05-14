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
# KV CACHE
# -------------------------

kv_state = (
    axion_cpp.KVState(2)
)

# -------------------------
# TOKENS
# -------------------------

tokens = tok.encode(
    "Hello Axion"
)

print("\nTOKENS:")

print(tokens)

# -------------------------
# INCREMENTAL LOOP
# -------------------------

for pos, token in enumerate(tokens):

    logits = (
        axion_cpp.incremental_forward(
            token,
            pos,
            embedding_matrix,
            position_matrix,
            loader,
            kv_state,
            2,
            12
        )
    )

    print(f"\nSTEP {pos}")

    print("LOGITS:")

    print(logits.shape)

# -------------------------
# NEXT TOKEN
# -------------------------

last_logits = (
    axion_cpp.last_token(
        logits
    )
)

next_token = (
    axion_cpp.argmax(
        last_logits
    )
)

print("\nNEXT TOKEN:")

print(next_token)

print(
    tok.decode([next_token])
)