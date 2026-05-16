#include "attention.hpp"
#include "blas.hpp"
#include "transpose.hpp"

#include <cmath>
#include <stdexcept>

namespace axion {

Tensor attention_scores(
    const Tensor& Q,
    const Tensor& K,
    RuntimeMemoryScheduler* scheduler
) {

    if (Q.shape.size() != 2 ||
        K.shape.size() != 2) {

        throw std::runtime_error(
            "attention_scores expects 2D tensors"
        );
    }

    int64_t q_dim =
        Q.shape[1];

    int64_t k_dim =
        K.shape[1];

    if (q_dim != k_dim) {

        throw std::runtime_error(
            "Q/K hidden dimension mismatch"
        );
    }

    // --------------------------------
    // TRANSPOSE
    // --------------------------------

    Tensor K_t =
        transpose(
            K,
            scheduler
        );

    // --------------------------------
    // MATMUL
    // --------------------------------

    Tensor scores =
        matmul(
            Q,
            K_t,
            scheduler
        );

    // --------------------------------
    // RELEASE TEMP
    // --------------------------------

    if (scheduler != nullptr) {

        scheduler->release_tensor(
            K_t.name
        );
    }

    float scale =
        1.0f / std::sqrt(
            static_cast<float>(q_dim)
        );

    for (int64_t i = 0;
         i < scores.numel();
         i++) {

        scores.data()[i] *= scale;
    }

    scores.name =
        "attention_scores";

    return scores;
}

}