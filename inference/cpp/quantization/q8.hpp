#pragma once

#include "../core/tensor.hpp"

#include <vector>
#include <cstdint>

namespace axion {

struct Q8Tensor {

    std::vector<int8_t> data;

    std::vector<int64_t> shape;

    float scale = 1.0f;
};

Q8Tensor quantize_q8(
    const Tensor& input
);

Tensor dequantize_q8(
    const Q8Tensor& q
);

}