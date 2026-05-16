
#include "blas.hpp"
#include "../core/tensor_factory.hpp"
#include <stdexcept>
#include <iostream>

#include "../core/tensor_factory.hpp"
#include "../core/arena.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace axion {

Tensor matmul(
    const Tensor& a,
    const Tensor& b,
    RuntimeMemoryScheduler* scheduler
) {
    if (!a.valid() || !b.valid()) {

        throw std::runtime_error(
            "Invalid tensor in matmul"
        );
    }

    if (a.shape.size() != 2 ||
        b.shape.size() != 2) {

        throw std::runtime_error(
            "matmul only supports 2D tensors"
        );
    }

    int64_t M = a.shape[0];
    int64_t K = a.shape[1];

    int64_t K2 = b.shape[0];
    int64_t N = b.shape[1];

    if (K != K2) {

        throw std::runtime_error(
            "matmul dimension mismatch"
        );
    }

Tensor output;

if (scheduler != nullptr) {

    output =
        scheduler->request_tensor(
            "matmul_output",
            {M, N},
            DType::FLOAT32
        );
}
else {

    output =
        create_owned_tensor(
            {M, N},
            DType::FLOAT32
        );
}
output.name =
    "matmul_output";

    #pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < M; i++) {

        for (int64_t j = 0; j < N; j++) {

            float sum = 0.0f;

            for (int64_t k = 0; k < K; k++) {

                float a_val =
                    a.value(i * K + k);

                float b_val =
                    b.value(k * N + j);

                sum += a_val * b_val;
            }

            output.data()[i * N + j] = sum;
        }
    }

    return output;
}
Tensor matmul_arena(
    const Tensor& a,
    const Tensor& b,
    Arena& arena
) {

    if (a.shape.size() != 2 ||
        b.shape.size() != 2) {

        throw std::runtime_error(
            "matmul only supports 2D tensors"
        );
    }

    int64_t M = a.shape[0];
    int64_t K = a.shape[1];

    int64_t K2 = b.shape[0];
    int64_t N = b.shape[1];

    if (K != K2) {

        throw std::runtime_error(
            "matmul dimension mismatch"
        );
    }

    Tensor output =
        create_tensor(
            arena,
            {M, N},
            DType::FLOAT32
        );

    output.name =
        "matmul_arena_output";

    #pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < M; i++) {

        for (int64_t j = 0; j < N; j++) {

            float sum = 0.0f;

            for (int64_t k = 0; k < K; k++) {

                sum +=
                    a.value(i * K + k) *
                    b.value(k * N + j);
            }

            output.data_ptr[
                i * N + j
            ] = sum;
        }
    }

    return output;
}

}
