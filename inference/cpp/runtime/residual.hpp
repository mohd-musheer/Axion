#pragma once

#include "../core/tensor.hpp"

namespace axion {

Tensor residual_add(
    const Tensor& a,
    const Tensor& b
);

}