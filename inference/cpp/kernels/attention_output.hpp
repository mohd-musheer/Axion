
#pragma once

#include "../core/tensor.hpp"

namespace axion {

Tensor attention_output(
    const Tensor& attention_probs,
    const Tensor& V
);

}
