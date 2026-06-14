
#pragma once

#include "../core/tensor.hpp"

namespace axion {

Tensor compute_logits(
    const Tensor& hidden_states,
    const Tensor& embedding_matrix
);

int argmax(
    const Tensor& logits
);

}
