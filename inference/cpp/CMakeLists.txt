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
    core/mmap_loader.cpp
    kernels/attention_output.cpp
    kernels/blas.cpp
    kernels/rmsnorm.cpp
    runtime/generation.cpp
    runtime/embedding.cpp
    runtime/weight_lookup.cpp
    kernels/residual.cpp
    runtime/execution_graph.cpp
    kernels/multihead.cpp
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