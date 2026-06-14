#pragma once

#include "../core/tensor.hpp"
#include "../core/scheduler.hpp"

namespace axion {

Tensor silu(
    const Tensor& input,
    RuntimeMemoryScheduler* scheduler = nullptr
);

}