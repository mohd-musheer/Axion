cmake_minimum_required(VERSION 3.14)

project(axion_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(AXION_BUILD_PYTHON_MODULE "Build the pybind11 Python module" ON)
option(AXION_BUILD_VALIDATION "Build the runtime validation harness" ON)
option(AXION_BUILD_EXAMPLES "Build the command-line example drivers" OFF)

find_package(OpenMP REQUIRED)

include_directories(
    third_party
)

if(AXION_BUILD_PYTHON_MODULE)

    set(PYBIND11_FINDPYTHON ON)

    find_package(Python 3.11 EXACT REQUIRED
        COMPONENTS Interpreter Development
    )

    find_package(pybind11 REQUIRED)

    pybind11_add_module(axion_cpp

        bindings/pybind_module.cpp

        core/allocator.cpp
        core/mapped_file.cpp
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
        gguf/tokenizer.cpp
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
        kernels/simd_dot.cpp
        runtime/inference_engine.cpp
        runtime/model_runner.cpp
        runtime/sampler.cpp
        runtime/quantized_matmul.cpp
        runtime/mha.cpp
    )

    target_link_libraries(axion_cpp PRIVATE
        Python::Python
        OpenMP::OpenMP_CXX
    )

endif()

if(AXION_BUILD_VALIDATION)

    # Pure C++ validation harness: no Python / pybind dependency,
    # so CI can build and run it on a plain compiler image.
    add_executable(axion_runtime_validation

        tests/runtime_validation.cpp

        core/allocator.cpp
        core/mapped_file.cpp
        core/mmap_loader.cpp
        gguf/gguf.cpp
        core/tensor.cpp
        core/fp16.cpp
        core/tensor_view.cpp
        core/tensor_factory.cpp
        core/arena.cpp
        core/scheduler.cpp
        runtime/tensor_lifetime.cpp
        runtime/tensor_evictor.cpp
        runtime/kv_append.cpp
        runtime/kv_cache.cpp
        runtime/paged_kv.cpp
        runtime/token_embedding.cpp
        runtime/streaming_executor.cpp
        runtime/linear.cpp
        runtime/residual.cpp
        runtime/final_norm.cpp
        runtime/model_runner.cpp
        kernels/blas.cpp
        kernels/rmsnorm.cpp
        kernels/transpose.cpp
        kernels/attention.cpp
        kernels/softmax.cpp
        kernels/silu.cpp
        kernels/elementwise.cpp
        runtime/logits.cpp
        runtime/sampler.cpp
        runtime/quantized_matmul.cpp
        kernels/simd_dot.cpp
        runtime/mha.cpp
    )

endif()

option(AXION_BUILD_MODEL_VALIDATION "Build the real-model validation harness" OFF)

if(AXION_BUILD_MODEL_VALIDATION)

    # Real-model harness: loads a GGUF + a llama.cpp reference JSON and
    # compares logits. Not part of CI (needs a model file); built and
    # run manually. Reuses the validation source set.
    add_executable(axion_model_validation
        tests/model_validation.cpp
        core/allocator.cpp
        core/mapped_file.cpp
        core/mmap_loader.cpp
        gguf/gguf.cpp
        core/tensor.cpp
        core/fp16.cpp
        core/tensor_view.cpp
        core/tensor_factory.cpp
        core/arena.cpp
        core/scheduler.cpp
        runtime/tensor_lifetime.cpp
        runtime/tensor_evictor.cpp
        runtime/kv_append.cpp
        runtime/kv_cache.cpp
        runtime/paged_kv.cpp
        runtime/token_embedding.cpp
        runtime/streaming_executor.cpp
        runtime/linear.cpp
        runtime/residual.cpp
        runtime/final_norm.cpp
        runtime/model_runner.cpp
        runtime/logits.cpp
        runtime/sampler.cpp
        runtime/quantized_matmul.cpp
        kernels/simd_dot.cpp
        runtime/mha.cpp
        kernels/blas.cpp
        kernels/rmsnorm.cpp
        kernels/transpose.cpp
        kernels/attention.cpp
        kernels/softmax.cpp
        kernels/silu.cpp
        kernels/elementwise.cpp
    )

    target_include_directories(axion_model_validation PRIVATE third_party)
    target_link_libraries(axion_model_validation PRIVATE OpenMP::OpenMP_CXX)

    # Standalone logits dumper: same forward path, JSON output matching
    # the llama.cpp reference schema for offline diffing.
    add_executable(axion_dump_logits
        tests/dump_logits.cpp
        core/allocator.cpp
        core/mapped_file.cpp
        core/mmap_loader.cpp
        gguf/gguf.cpp
        core/tensor.cpp
        core/fp16.cpp
        core/tensor_view.cpp
        core/tensor_factory.cpp
        core/arena.cpp
        core/scheduler.cpp
        runtime/tensor_lifetime.cpp
        runtime/tensor_evictor.cpp
        runtime/kv_append.cpp
        runtime/kv_cache.cpp
        runtime/paged_kv.cpp
        runtime/token_embedding.cpp
        runtime/streaming_executor.cpp
        runtime/linear.cpp
        runtime/residual.cpp
        runtime/final_norm.cpp
        runtime/model_runner.cpp
        runtime/logits.cpp
        runtime/sampler.cpp
        runtime/quantized_matmul.cpp
        kernels/simd_dot.cpp
        runtime/mha.cpp
        kernels/blas.cpp
        kernels/rmsnorm.cpp
        kernels/transpose.cpp
        kernels/attention.cpp
        kernels/softmax.cpp
        kernels/silu.cpp
        kernels/elementwise.cpp
    )
    target_include_directories(axion_dump_logits PRIVATE third_party)
    target_link_libraries(axion_dump_logits PRIVATE OpenMP::OpenMP_CXX)

    # Shared source set for the real-model executables below.
    set(AXION_MODEL_SOURCES
        core/allocator.cpp
        core/mapped_file.cpp
        core/mmap_loader.cpp
        gguf/gguf.cpp
        gguf/tokenizer.cpp
        core/tensor.cpp
        core/fp16.cpp
        core/tensor_view.cpp
        core/tensor_factory.cpp
        core/arena.cpp
        core/scheduler.cpp
        runtime/tensor_lifetime.cpp
        runtime/tensor_evictor.cpp
        runtime/kv_append.cpp
        runtime/kv_cache.cpp
        runtime/paged_kv.cpp
        runtime/token_embedding.cpp
        runtime/streaming_executor.cpp
        runtime/linear.cpp
        runtime/residual.cpp
        runtime/final_norm.cpp
        runtime/model_runner.cpp
        runtime/logits.cpp
        runtime/sampler.cpp
        runtime/quantized_matmul.cpp
        kernels/simd_dot.cpp
        runtime/mha.cpp
        kernels/blas.cpp
        kernels/rmsnorm.cpp
        kernels/transpose.cpp
        kernels/attention.cpp
        kernels/softmax.cpp
        kernels/silu.cpp
        kernels/elementwise.cpp
    )

    # TinyLlama regression suite (load/metadata/shape/top-1/decode/KV).
    add_executable(axion_tinyllama_validation
        tests/tinyllama_validation.cpp
        ${AXION_MODEL_SOURCES}
    )
    target_include_directories(axion_tinyllama_validation PRIVATE third_party)
    target_link_libraries(axion_tinyllama_validation PRIVATE OpenMP::OpenMP_CXX)

    # User-facing inference CLI.
    add_executable(axion_cli
        cli/axion_cli.cpp
        ${AXION_MODEL_SOURCES}
    )
    target_link_libraries(axion_cli PRIVATE OpenMP::OpenMP_CXX)

endif()

if(AXION_BUILD_EXAMPLES)

    # Minimal end-to-end inference drivers. Reuse the same source set as
    # the validation harness so they link the full streaming pipeline.
    set(AXION_E2E_SOURCES
        core/allocator.cpp
        core/mapped_file.cpp
        core/mmap_loader.cpp
        gguf/gguf.cpp
        core/tensor.cpp
        core/fp16.cpp
        core/tensor_view.cpp
        core/tensor_factory.cpp
        core/arena.cpp
        core/scheduler.cpp
        runtime/tensor_lifetime.cpp
        runtime/tensor_evictor.cpp
        runtime/kv_append.cpp
        runtime/kv_cache.cpp
        runtime/paged_kv.cpp
        runtime/token_embedding.cpp
        runtime/streaming_executor.cpp
        runtime/linear.cpp
        runtime/residual.cpp
        runtime/final_norm.cpp
        runtime/model_runner.cpp
        runtime/logits.cpp
        runtime/sampler.cpp
        runtime/quantized_matmul.cpp
        kernels/simd_dot.cpp
        runtime/mha.cpp
        kernels/blas.cpp
        kernels/rmsnorm.cpp
        kernels/transpose.cpp
        kernels/attention.cpp
        kernels/softmax.cpp
        kernels/silu.cpp
        kernels/elementwise.cpp
    )

    find_package(OpenMP REQUIRED)

    add_executable(axion_single_token
        examples/single_token.cpp
        ${AXION_E2E_SOURCES}
    )
    target_link_libraries(axion_single_token PRIVATE OpenMP::OpenMP_CXX)

    # Production CLI runtime.
    add_executable(axion
        main.cpp
        ${AXION_E2E_SOURCES}
    )
    target_link_libraries(axion PRIVATE OpenMP::OpenMP_CXX)

    # Attention benchmark (old single-head vs new MHA+GQA+RoPE).
    add_executable(axion_bench_attention
        examples/bench_attention.cpp
        ${AXION_E2E_SOURCES}
    )
    target_link_libraries(axion_bench_attention PRIVATE OpenMP::OpenMP_CXX)

endif()
