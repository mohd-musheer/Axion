
#pragma once

#include "../core/tensor.hpp"

namespace axion {

Tensor attention_scores(
    const Tensor& Q,
    const Tensor& K
);

}
