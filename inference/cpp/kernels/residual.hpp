#pragma once

#include "../core/tensor.hpp"
#include "../core/scheduler.hpp"

namespace axion {

Tensor residual_add(
    const Tensor& x,
    const Tensor& y,
    RuntimeMemoryScheduler* scheduler = nullptr
);

}