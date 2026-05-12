
#pragma once

#include "../core/tensor.hpp"

namespace axion {

Tensor mlp_block(
    const Tensor& gate,
    const Tensor& up
);

}
