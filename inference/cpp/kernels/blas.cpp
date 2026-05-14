
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
    const Tensor& A,
    const Tensor& B
) {
    if (!A.valid() || !B.valid()) {

        throw std::runtime_error(
            "Invalid tensor in matmul"
        );
    }

    if (A.shape.size() != 2 ||
        B.shape.size() != 2) {

        throw std::runtime_error(
            "matmul only supports 2D tensors"
        );
    }

    int64_t M = A.shape[0];
    int64_t K = A.shape[1];

    int64_t K2 = B.shape[0];
    int64_t N = B.shape[1];

    if (K != K2) {

        throw std::runtime_error(
            "matmul dimension mismatch"
        );
    }

Tensor output =
    create_owned_tensor(
        {M, N},
        DType::FLOAT32
    );

output.name =
    "matmul_output";

    #pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < M; i++) {

        for (int64_t j = 0; j < N; j++) {

            float sum = 0.0f;

            for (int64_t k = 0; k < K; k++) {

                float a =
                    A.value(i * K + k);

                float b =
                    B.value(k * N + j);

                sum += a * b;
            }

            output.data()[i * N + j] = sum;
        }
    }

    return output;
}
Tensor matmul_arena(
    const Tensor& A,
    const Tensor& B,
    Arena& arena
) {

    if (A.shape.size() != 2 ||
        B.shape.size() != 2) {

        throw std::runtime_error(
            "matmul only supports 2D tensors"
        );
    }

    int64_t M = A.shape[0];
    int64_t K = A.shape[1];

    int64_t K2 = B.shape[0];
    int64_t N = B.shape[1];

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
                    A.value(i * K + k) *
                    B.value(k * N + j);
            }

            output.data_ptr[
                i * N + j
            ] = sum;
        }
    }

    return output;
}

}
