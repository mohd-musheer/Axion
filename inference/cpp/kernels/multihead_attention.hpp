
#pragma once

#include "../core/tensor.hpp"

namespace axion {

Tensor multihead_attention(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    int num_heads
);

}
