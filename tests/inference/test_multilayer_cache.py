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
# KV STATE
# -------------------------

num_layers = 2

kv_state = (
    axion_cpp.KVState(
        num_layers
    )
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
# RUN TOKENS
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
            num_layers,
            12
        )
    )

# -------------------------
# INSPECT CACHE
# -------------------------

print("\n=== LAYER CACHE STATE ===")

for layer in range(num_layers):

    cache = (
        kv_state.layers[layer]
    )

    print(f"\nLAYER {layer}")

    print("KEY SHAPE:")

    print(cache.keys.shape)

    print("VALUE SHAPE:")

    print(cache.values.shape)

    print("KEY VALUES:")

    print(cache.keys.data[:10])