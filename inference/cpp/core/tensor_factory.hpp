#pragma once

#include "tensor.hpp"
#include "arena.hpp"

#include <vector>

namespace axion {

Tensor create_tensor(
    Arena& arena,
    const std::vector<int64_t>& shape,
    DType dtype = DType::FLOAT32
);

Tensor create_owned_tensor(
    const std::vector<int64_t>& shape,
    DType dtype = DType::FLOAT32
);

}