
#pragma once

#include "../core/tensor.hpp"

namespace axion {

Tensor matmul(
    const Tensor& A,
    const Tensor& B
);
Tensor matmul_arena(
    const Tensor& A,
    const Tensor& B,
    Arena& arena
);

}
