
#include "attention.hpp"
#include "blas.hpp"
#include "transpose.hpp"

#include <cmath>
#include <stdexcept>

namespace axion {

Tensor attention_scores(
    const Tensor& Q,
    const Tensor& K
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

    Tensor K_t =
        transpose(K);

    Tensor scores =
        matmul(Q, K_t);

    float scale =
        1.0f / std::sqrt(
            static_cast<float>(q_dim)
        );

    for (size_t i = 0;
         i < scores.data.size();
         i++) {

        scores.data[i] *= scale;
    }

    scores.name =
        "attention_scores";

    return scores;
}

}
