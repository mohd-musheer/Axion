# Phase 14.5 — TinyLlama Real-Model Verification (full procedure)

> See the bottom of this file for the original short procedure. This
> section is the complete, authoritative reference.

## Required files

- A TinyLlama GGUF, e.g. `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`.
- `scripts/gen_reference_logits.py` (needs `llama-cpp-python`), OR
- `scripts/generate_reference_from_llama_cli.py` (no Python binding;
  consumes a llama.cpp binary logits dump). **Preferred on Windows.**
- C++ executables (built with `-DAXION_BUILD_MODEL_VALIDATION=ON`):
  - `axion_model_validation` — in-process logits/metadata comparison.
  - `axion_tinyllama_validation` — regression suite.
  - `axion_dump_logits` — dumps Axion logits to JSON.
  - `axion_cli` — user-facing inference driver.

## Build

```
cmake -S inference/cpp -B build \
    -DAXION_BUILD_PYTHON_MODULE=OFF \
    -DAXION_BUILD_VALIDATION=OFF \
    -DAXION_BUILD_MODEL_VALIDATION=ON
cmake --build build -j
```

## Reference generation

### Option A — llama-cpp-python (Linux/macOS, easiest)

```
python scripts/gen_reference_logits.py \
    --model models/tinyllama.Q4_K_M.gguf \
    --tokens 1 15043 29892 590 1024 338 \
    --n-threads 1 --out reference.json
```

### Option B — llama.cpp binaries (Windows-friendly, no binding)

Run upstream `llama-cli`/`llama-eval` with `--logits-all` and save the
logits as raw float32 `[seq, n_vocab]` (or text). Then:

```
python scripts/generate_reference_from_llama_cli.py \
    --model models/tinyllama.Q4_K_M.gguf \
    --tokens 1 15043 29892 590 1024 338 \
    --logits-bin llama_logits.f32 --n-vocab 32000 \
    --n-embd 2048 --n-layer 22 --n-head 32 --n-head-kv 4 \
    --rope-theta 10000 --out reference.json
```

Use `--last-only` if your build only emits the final position's logits.

## Validation commands

```
# In-process logits + metadata comparison
./build/axion_model_validation models/tinyllama.Q4_K_M.gguf reference.json

# Full regression suite (load/metadata/shape/top-1/decode/KV reuse)
./build/axion_tinyllama_validation models/tinyllama.Q4_K_M.gguf reference.json

# Standalone Axion logits dump (offline diff)
./build/axion_dump_logits models/tinyllama.Q4_K_M.gguf axion.json \
    1 15043 29892 590 1024 338

# User-facing inference
./build/axion_cli --model models/tinyllama.Q4_K_M.gguf \
    --tokens 1 15043 29892 590 1024 338 --max-new 16

# Text prompt via Python wrapper (full tokenizer)
python -m inference.python.axion \
    --model models/tinyllama.Q4_K_M.gguf \
    --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --prompt "Hello, my name is"
```

## Expected outputs

`axion_tinyllama_validation`:

```
PASS  TinyLlama GGUF load
PASS  TinyLlama metadata validation
PASS  TinyLlama logits shape validation
PASS  TinyLlama top-1 validation
      logits[last] max_abs=... mean_abs=...
PASS  TinyLlama logits tolerance
PASS  TinyLlama incremental decode validation
PASS  TinyLlama KV reuse validation
      next(full)=... next(kv)=...

AXION TINYLLAMA VALIDATION PASSED (6/6 checks passed)
```

`axion_model_validation`: all `PASS` lines (meta n_embd/n_layer/n_head/
n_head_kv/n_vocab/arch/rope_theta/seq_len, logits shape, logits[last],
top-1) and `AXION MODEL VALIDATION PASSED (0 failures)`.

## Pass criteria

- Metadata equality: `arch`, `n_embd`, `n_layer`, `n_head`, `n_head_kv`,
  `n_vocab`, `rope_theta`, `seq_len`.
- Shape equality: `[seq, vocab]` matches the reference.
- Last-position logits within tolerance (max 0.75 / mean 0.08).
- **top-1 token exact match** (hard floor).
- Incremental decode and KV-reuse predictions equal the full-prefill
  greedy prediction.

## Troubleshooting

- **top-1 mismatch, large max_abs** — suspect RoPE convention
  (`RopeType::NEOX` is correct for TinyLlama) or GQA grouping
  (`n_head % n_kv_head`). Check `mha.cpp`.
- **Shape `[seq, 0]` / wrong vocab** — LM head selection; verify
  `output.weight` vs tied `token_embd.weight` in `model_runner.cpp`.
- **"GGUF dequant not implemented"** — the model uses a tensor type with
  no decode path; re-quantize to Q4_K_M / Q8_0 / F16.
- **Reference seq mismatch** — the token ids passed to the reference
  generator differ from those given to Axion; they must be identical.
- **`llama-cpp-python` won't install (Windows)** — use Option B with the
  prebuilt llama.cpp binaries and `generate_reference_from_llama_cli.py`.
- **KV reuse FAIL but full-prefill PASS** — bug in the incremental path
  (`decode_layer`/`mha_attention_incremental`) or `rope_apply_row_at`
  position handling.

---

## Original short procedure


Correctness only. No CUDA, no backend abstraction, no benchmarking, no
speculative decoding, no distributed inference.

Goal: prove Axion produces the same logits as llama.cpp for a real
TinyLlama GGUF, within a defined tolerance, with exact top-1 token
agreement, and matching shapes/metadata.

## Why token-id space

We feed **explicit integer token ids** to both engines, never raw text.
This removes the tokenizer as a variable: any divergence is attributable
to the forward pass (RoPE, GQA, KV layout, RMSNorm, dequant), not to a
tokenizer mismatch. A real tokenizer is out of scope for this phase.

## Pipeline

```
TinyLlama GGUF -> llama.cpp     -> reference logits (reference.json)
TinyLlama GGUF -> Axion runner  -> generated logits (axion.json)
                       |
                       v
        model_validation compares last-position logits
```

## Prerequisites

- A TinyLlama GGUF, e.g. `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`
  (Q4_K / Q6_K / F16 tensors are all supported by the loader).
- `pip install llama-cpp-python` for the reference.
- A C++17 toolchain + OpenMP.

## Steps

1. Build the harness and dumper:

   ```
   cmake -S inference/cpp -B build \
       -DAXION_BUILD_PYTHON_MODULE=OFF \
       -DAXION_BUILD_VALIDATION=OFF \
       -DAXION_BUILD_MODEL_VALIDATION=ON
   cmake --build build -j
   ```

2. Generate the llama.cpp reference (greedy, single-thread = stable):

   ```
   python scripts/gen_reference_logits.py \
       --model models/tinyllama.Q4_K_M.gguf \
       --tokens 1 15043 29892 590 1024 338 \
       --n-threads 1 \
       --out reference.json
   ```

3. Run the in-process comparison:

   ```
   ./build/axion_model_validation models/tinyllama.Q4_K_M.gguf reference.json
   ```

4. (Optional) Dump Axion logits standalone and diff offline:

   ```
   ./build/axion_dump_logits models/tinyllama.Q4_K_M.gguf axion.json \
       1 15043 29892 590 1024 338
   ```

## Tolerances

Defined in `tests/model_validation.cpp`:

- `LOGITS_MAX_ABS_TOL  = 0.75`  (max abs error over the last-position row)
- `LOGITS_MEAN_ABS_TOL = 0.08`  (mean abs error over the last-position row)

These absorb f16/f32 accumulation differences between Axion (f32) and
llama.cpp (mixed) while still catching a structural bug.

## Success criteria

Exit code 0 with all of:

- metadata equality: `arch`, `n_embd`, `n_layer`, `n_head`, `n_head_kv`,
  `rope_theta`
- shape equality: `[seq, vocab]` matches the reference
- last-position logits within tolerance
- **top-1 token exact match** (hard floor)

## Reproducibility

Given the same GGUF + same token ids + single-thread greedy llama.cpp,
`reference.json` is bit-stable. Axion's forward pass is deterministic
(no sampling on this path), so re-runs produce identical logits.
