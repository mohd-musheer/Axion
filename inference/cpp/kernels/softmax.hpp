#pragma once

#include "../core/tensor.hpp"
#include "../core/scheduler.hpp"

namespace axion {

Tensor softmax(
    const Tensor& input,
    RuntimeMemoryScheduler* scheduler = nullptr
);

}