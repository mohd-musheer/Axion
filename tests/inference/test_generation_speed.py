import sys
import time

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
# PROMPT
# -------------------------

prompt = (
    "Axion is building a transformer "
    "runtime from scratch"
)

tokens = tok.encode(prompt)

# ==================================================
# FULL RECOMPUTE
# ==================================================

print("\n=== FULL RECOMPUTE ===")

full_tokens = list(tokens)

start = time.time()

for step in range(20):

    embeddings = (
        axion_cpp.embedding_lookup(
            embedding_matrix,
            full_tokens
        )
    )

    pos = (
        axion_cpp.position_embedding_lookup(
            position_matrix,
            len(full_tokens)
        )
    )

    hidden = (
        axion_cpp.add(
            embeddings,
            pos
        )
    )

    logits = (
        axion_cpp.full_forward(
            hidden,
            loader,
            2,
            12,
            embedding_matrix
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

    full_tokens.append(
        next_token
    )

end = time.time()

full_time = end - start

print(f"\nFULL TIME: {full_time:.4f} sec")

# ==================================================
# CACHED GENERATION
# ==================================================

print("\n=== CACHED GENERATION ===")

cached_tokens = list(tokens)

kv_state = (
    axion_cpp.KVState(2)
)

start = time.time()

# -------------------------
# PREFILL
# -------------------------

for pos, token in enumerate(cached_tokens):

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

current_token = cached_tokens[-1]

# -------------------------
# GENERATION
# -------------------------

for step in range(20):

    logits = (
        axion_cpp.incremental_forward(
            current_token,
            len(cached_tokens) - 1,
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

    cached_tokens.append(
        next_token
    )

    current_token = next_token

end = time.time()

cached_time = end - start

print(f"\nCACHED TIME: {cached_time:.4f} sec")

# ==================================================
# SPEEDUP
# ==================================================

speedup = (
    full_time / cached_time
)

print(f"\nSPEEDUP: {speedup:.2f}x")