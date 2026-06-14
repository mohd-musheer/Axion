#pragma once

#include "../core/tensor.hpp"

namespace axion {

Tensor token_embedding(
    const Tensor& embedding_matrix,
    int token_id
);

}