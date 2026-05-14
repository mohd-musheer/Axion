#pragma once

#include "tensor.hpp"

namespace axion {

Tensor tensor_view(
    const Tensor& base,
    int64_t offset,
    std::vector<int64_t> shape
);

}