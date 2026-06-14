#pragma once

#include "../core/tensor.hpp"

namespace axion {

Tensor last_token(
    const Tensor& logits
);

}