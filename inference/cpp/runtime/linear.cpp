
#include "linear.hpp"

#include "../kernels/blas.hpp"
#include "../core/tensor_factory.hpp"

#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace axion {

Tensor linear(
    const Tensor& input,
    const Tensor& weight,
    RuntimeMemoryScheduler* scheduler
) {

    return matmul(
        input,
        weight
    );
}

Tensor linear_from_gguf(
    const Tensor& input,
    const Tensor& weight
) {
    if (input.shape.size() != 2 || weight.shape.size() != 2) {
        throw std::runtime_error(
            "linear_from_gguf expects 2D input [M,K] and weight [N,K]");
    }

    int64_t M = input.shape[0];
    int64_t K = input.shape[1];
    int64_t N = weight.shape[0];
    int64_t Kw = weight.shape[1];

    if (K != Kw) {
        throw std::runtime_error(
            "linear_from_gguf dimension mismatch: input K != weight K");
    }

    Tensor output =
        create_owned_tensor({M, N}, DType::FLOAT32);
    output.name = "linear_from_gguf_output";

    const float* in_data = input.data();
    const float* w_data  = weight.data();
    float*       out_data = output.data();

    // out[i, j] = sum_k input[i, k] * weight[j, k]
    // Both input row i and weight row j are contiguous length-K spans,
    // so each dot product walks two sequential buffers (cache-friendly).
    #pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            const float* a = in_data + i * K;
            const float* b = w_data  + j * K;
            float acc = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                acc += a[k] * b[k];
            }
            out_data[i * N + j] = acc;
        }
    }

    return output;
}

}
