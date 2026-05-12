
#pragma once

#include "../core/tensor.hpp"

namespace axion {

Tensor generate_tokens(
    const Tensor& input,
    int steps
);

}
