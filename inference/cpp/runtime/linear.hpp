#pragma once

#include "../core/tensor.hpp"
#include "../core/scheduler.hpp"
namespace axion {

Tensor linear(
    const Tensor& input,
    const Tensor& weight,
    RuntimeMemoryScheduler* scheduler = nullptr
);

}