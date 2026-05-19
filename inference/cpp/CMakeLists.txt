cmake_minimum_required(VERSION 3.14)

project(axion_cpp)

set(CMAKE_CXX_STANDARD 17)

set(PYBIND11_FINDPYTHON ON)

find_package(Python 3.11 EXACT REQUIRED
    COMPONENTS Interpreter Development
)

find_package(pybind11 REQUIRED)

find_package(OpenMP REQUIRED)

include_directories(
    third_party
)

pybind11_add_module(axion_cpp

    bindings/pybind_module.cpp

    core/tensor.cpp
    quantization/q8.cpp
    core/fp16.cpp
    core/mmap_loader.cpp
    kernels/attention_output.cpp
    kernels/blas.cpp
    runtime/paged_kv.cpp
    runtime/transformer_stack.cpp
    runtime/residual.cpp
    kernels/rmsnorm.cpp
    runtime/generation.cpp
    runtime/embedding.cpp
    runtime/streaming_executor.cpp
    runtime/tensor_evictor.cpp
    core/scheduler.cpp
    runtime/linear.cpp
    kernels/gelu.cpp
    quantization/q4.cpp
    runtime/last_token.cpp
    gguf/gguf.cpp
    core/arena.cpp
    core/tensor_factory.cpp
    kernels/layernorm.cpp
    runtime/token_embedding.cpp
    core/tensor_view.cpp
    runtime/incremental_forward.cpp
    runtime/gpt2_ln.cpp
    runtime/single_position.cpp
    runtime/cached_attention.cpp
    kernels/add.cpp
    runtime/weight_lookup.cpp
    runtime/position_embedding.cpp
    runtime/real_attention.cpp
    runtime/fused_qkv.cpp
    runtime/final_norm.cpp
    runtime/execution_graph.cpp
    runtime/logits.cpp
    kernels/multihead.cpp
    runtime/tensor_lifetime.cpp
    runtime/full_forward.cpp
    runtime/kv_append.cpp
    runtime/layer_scheduler.cpp
    kernels/multihead_attention.cpp
    kernels/causal_mask.cpp
    kernels/softmax.cpp
    kernels/attention.cpp
    runtime/kv_cache.cpp
    runtime/transformer_layer.cpp
    kernels/silu.cpp
    kernels/elementwise.cpp
    kernels/mlp.cpp
    kernels/transpose.cpp
    kernels/rope.cpp
    runtime/inference_engine.cpp
)

target_link_libraries(axion_cpp PRIVATE
    Python::Python
    OpenMP::OpenMP_CXX
)