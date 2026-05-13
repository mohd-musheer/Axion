#pragma once

#include "../core/tensor.hpp"

namespace axion {

Tensor gpt2_ln(
    const Tensor& input,
    const Tensor& weight
);

}