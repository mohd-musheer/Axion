
cmake_minimum_required(VERSION 3.14)

project(axion_cpp)

set(CMAKE_CXX_STANDARD 17)

set(PYBIND11_FINDPYTHON ON)

find_package(Python 3.11 EXACT REQUIRED
    COMPONENTS Interpreter Development
)

find_package(pybind11 REQUIRED)

find_package(OpenMP REQUIRED)

target_link_libraries(axion_cpp PRIVATE
    Python::Python
    OpenMP::OpenMP_CXX
)

include_directories(
    third_party
)

pybind11_add_module(axion_cpp

    bindings/pybind_module.cpp

    core/tensor.cpp
    core/mmap_loader.cpp

    kernels/blas.cpp

    runtime/inference_engine.cpp
)

