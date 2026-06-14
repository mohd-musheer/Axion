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
# PROMPT
# -------------------------

text = "Hello Axion"

tokens = tok.encode(text)

print("\nINITIAL TOKENS:")

print(tokens)

# -------------------------
# PREFILL
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

# -------------------------
# GENERATION
# -------------------------

generated = list(tokens)

current_token = tokens[-1]

for step in range(10):

    logits = (
        axion_cpp.incremental_forward(
            current_token,
            len(generated) - 1,
            embedding_matrix,
            position_matrix,
            loader,
            kv_state,
            2,
            12
        )
    )

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

    generated.append(
        next_token
    )

    decoded = tok.decode(
        [next_token]
    )

    print(f"\nSTEP {step+1}")

    print("TOKEN:", next_token)

    print("TEXT:", repr(decoded))

    current_token = next_token

# -------------------------
# FINAL TEXT
# -------------------------

final_text = tok.decode(
    generated
)

print("\n=== FINAL TEXT ===\n")

print(final_text)