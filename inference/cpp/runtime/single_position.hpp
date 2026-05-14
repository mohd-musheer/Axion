#pragma once

#include "../core/tensor.hpp"

namespace axion {

Tensor single_position_embedding(
    const Tensor& position_matrix,
    int position
);

}