#pragma once

#include "../core/tensor.hpp"
#include "../core/scheduler.hpp"

namespace axion {

Tensor causal_mask(
    const Tensor& scores,
    RuntimeMemoryScheduler* scheduler = nullptr
);

}