#pragma once

#include "../core/tensor.hpp"
#include "../core/scheduler.hpp"

namespace axion {

Tensor mlp_block(
    const Tensor& gate,
    const Tensor& up,
    RuntimeMemoryScheduler* scheduler = nullptr
);

}