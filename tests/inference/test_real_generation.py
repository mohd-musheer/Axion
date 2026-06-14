
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
# MATRICES
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
# INITIAL TEXT
# -------------------------

text = "Hello Axion"

token_ids = tok.encode(text)

print("\nINITIAL TOKENS:")

print(token_ids)

# -------------------------
# GENERATION LOOP
# -------------------------

for step in range(10):

    print(f"\nSTEP {step+1}")

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

    pos = (
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
            pos
        )
    )

    # -------------------------
    # TRANSFORMER
    # -------------------------

    logits = (
        axion_cpp.full_forward(
            embeddings,
            loader,
            2,
            12,
            embedding_matrix
        )
    )

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

    token_ids.append(
        next_token
    )

    decoded = tok.decode(
        [next_token]
    )

    print("TOKEN:", next_token)

    print("TEXT:", repr(decoded))

# -------------------------
# FINAL TEXT
# -------------------------

final_text = tok.decode(
    token_ids
)

print("\n=== FINAL TEXT ===\n")

print(final_text)
