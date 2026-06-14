# Standalone build for the native llama.cpp reference-logits dumper.
#
# This intentionally does NOT depend on the Axion tree. It links against
# the llama.cpp libraries you already built. Point LLAMA_CPP_DIR at your
# llama.cpp checkout/build so we can find <llama.h> and libllama/libggml.
#
# Configure & build (out-of-source):
#   cmake -S tools -B tools/build \
#         -DLLAMA_CPP_DIR=/path/to/llama.cpp \
#         -DLLAMA_CPP_BUILD=/path/to/llama.cpp/build
#   cmake --build tools/build --config Release

cmake_minimum_required(VERSION 3.14)
project(llama_logits_dump CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Locate llama.cpp ---
set(LLAMA_CPP_DIR "" CACHE PATH "Root of the llama.cpp source checkout")
set(LLAMA_CPP_BUILD "" CACHE PATH "llama.cpp build dir containing the libs")

if(NOT LLAMA_CPP_DIR)
    message(FATAL_ERROR
        "Set -DLLAMA_CPP_DIR=/path/to/llama.cpp (for llama.h).")
endif()
if(NOT LLAMA_CPP_BUILD)
    set(LLAMA_CPP_BUILD "${LLAMA_CPP_DIR}/build")
endif()

# Header search: cover both modern (include/) and legacy (root) layouts.
find_path(LLAMA_HEADER llama.h
    HINTS
        "${LLAMA_CPP_DIR}/include"
        "${LLAMA_CPP_DIR}"
    NO_DEFAULT_PATH
)
if(NOT LLAMA_HEADER)
    message(FATAL_ERROR "llama.h not found under ${LLAMA_CPP_DIR}")
endif()

# Library search: libllama + libggml live in the build dir; layouts vary
# (bin/, lib/, src/, ggml/src/), so search broadly.
find_library(LLAMA_LIB NAMES llama libllama
    HINTS
        "${LLAMA_CPP_BUILD}"
        "${LLAMA_CPP_BUILD}/bin"
        "${LLAMA_CPP_BUILD}/lib"
        "${LLAMA_CPP_BUILD}/src"
)
if(NOT LLAMA_LIB)
    message(FATAL_ERROR "libllama not found under ${LLAMA_CPP_BUILD}")
endif()

# ggml may be split into several libs in recent builds; link whatever exists.
set(GGML_LIBS "")
foreach(name ggml ggml-base ggml-cpu)
    find_library(GGML_${name}_LIB NAMES ${name} lib${name}
        HINTS
            "${LLAMA_CPP_BUILD}"
            "${LLAMA_CPP_BUILD}/bin"
            "${LLAMA_CPP_BUILD}/lib"
            "${LLAMA_CPP_BUILD}/ggml/src"
            "${LLAMA_CPP_BUILD}/src"
    )
    if(GGML_${name}_LIB)
        list(APPEND GGML_LIBS ${GGML_${name}_LIB})
    endif()
endforeach()

add_executable(llama_logits_dump llama_logits_dump.cpp)

# nlohmann/json is vendored in the Axion tree.
target_include_directories(llama_logits_dump PRIVATE
    "${LLAMA_HEADER}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../inference/cpp/third_party"
)

target_link_libraries(llama_logits_dump PRIVATE
    ${LLAMA_LIB}
    ${GGML_LIBS}
)

find_package(Threads REQUIRED)
target_link_libraries(llama_logits_dump PRIVATE Threads::Threads)
