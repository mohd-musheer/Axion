
#pragma once

#include "../core/tensor.hpp"

#include <vector>

namespace axion {

Tensor embedding_lookup(
    const Tensor& embedding_matrix,
    const std::vector<int>& token_ids
);

}
