#pragma once

#include "../core/tensor.hpp"

namespace axion {

Tensor rmsnorm(
    const Tensor& input,
    const Tensor& weight,
    float eps = 1e-6f
);

}