#pragma once

#include "../core/tensor.hpp"
#include "../core/scheduler.hpp"

namespace axion {

Tensor transpose(
    const Tensor& input,
    RuntimeMemoryScheduler* scheduler = nullptr
);

}