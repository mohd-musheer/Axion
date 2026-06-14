#pragma once

#include "../core/tensor.hpp"

#include <vector>

namespace axion {

Tensor position_embedding_lookup(
    const Tensor& position_matrix,
    int seq_len
);

}