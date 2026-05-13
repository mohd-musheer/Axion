
#pragma once

#include "../core/tensor.hpp"

namespace axion {

Tensor causal_mask(
    const Tensor& scores
);

}
