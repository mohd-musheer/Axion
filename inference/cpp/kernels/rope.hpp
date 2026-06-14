#pragma once

#include "../core/tensor.hpp"

namespace axion {

void apply_rope(
    Tensor& q,
    Tensor& k,
    int position,
    int head_dim,
    float theta = 10000.0f
);

}