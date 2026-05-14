#pragma once

#include "../core/tensor.hpp"

namespace axion {

Tensor layernorm(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    float eps = 1e-5f
);

}