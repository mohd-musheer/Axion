#pragma once

#include "../core/tensor.hpp"
#include "../core/scheduler.hpp"

#include <vector>

namespace axion {

std::vector<Tensor> split_heads(
    const Tensor& input,
    int num_heads
);

Tensor merge_heads(
    const std::vector<Tensor>& heads,
    RuntimeMemoryScheduler* scheduler = nullptr
);

}