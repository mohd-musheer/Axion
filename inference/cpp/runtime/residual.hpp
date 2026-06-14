#pragma once

#include "../core/tensor.hpp"
#include "../core/scheduler.hpp"
namespace axion {

Tensor residual_add(
    const Tensor& a,
    const Tensor& b,
    RuntimeMemoryScheduler* scheduler = nullptr
);

}