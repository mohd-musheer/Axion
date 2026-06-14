#pragma once

#include "../core/tensor.hpp"
#include "../core/arena.hpp"
#include "../core/scheduler.hpp"

namespace axion {

Tensor matmul(
    const Tensor& a,
    const Tensor& b,
    RuntimeMemoryScheduler* scheduler = nullptr
);

Tensor matmul_arena(
    const Tensor& a,
    const Tensor& b,
    Arena& arena
);

}