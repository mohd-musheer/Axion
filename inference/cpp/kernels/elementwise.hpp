#pragma once

#include "../core/tensor.hpp"
#include "../core/scheduler.hpp"

namespace axion {

Tensor elementwise_mul(
    const Tensor& a,
    const Tensor& b,
    RuntimeMemoryScheduler* scheduler = nullptr
);

}