#pragma once

#include "../core/tensor.hpp"

#include <vector>
#include <cstdint>

namespace axion {

struct Q4Tensor {

    std::vector<uint8_t> data;

    std::vector<int64_t> shape;

    float scale = 1.0f;
};

Q4Tensor quantize_q4(
    const Tensor& input
);

Tensor dequantize_q4(
    const Q4Tensor& q
);

}