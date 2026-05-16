#pragma once

#include "../core/tensor.hpp"
#include "../core/scheduler.hpp"

namespace axion {

Tensor attention_scores(
    const Tensor& Q,
    const Tensor& K,
    RuntimeMemoryScheduler* scheduler = nullptr
);

}