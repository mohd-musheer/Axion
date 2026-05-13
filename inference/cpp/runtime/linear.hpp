#pragma once

#include "../core/tensor.hpp"

namespace axion {

Tensor linear(
    const Tensor& input,
    const Tensor& weight
);

}