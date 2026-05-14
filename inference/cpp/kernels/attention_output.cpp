
#include "attention_output.hpp"
#include "blas.hpp"

#include <stdexcept>

namespace axion {

Tensor attention_output(
    const Tensor& attention_probs,
    const Tensor& V
) {

    if (attention_probs.shape.size() != 2 ||
        V.shape.size() != 2) {

        throw std::runtime_error(
            "attention_output expects 2D tensors"
        );
    }

    int64_t probs_cols =
        attention_probs.shape[1];

    int64_t v_rows =
        V.shape[0];

    if (probs_cols != v_rows) {

        throw std::runtime_error(
            "attention_probs/V mismatch"
        );
    }

    Tensor output =
        matmul(
            attention_probs,
            V
        );

    output.name =
        "attention_output";

    output.dtype =
        DType::FLOAT32;

    return output;
}

}
