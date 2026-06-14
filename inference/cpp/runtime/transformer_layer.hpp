#pragma once

#include "../core/tensor.hpp"
#include "../core/scheduler.hpp"

namespace axion {

Tensor transformer_layer(
    const Tensor& input,
    RuntimeMemoryScheduler* scheduler = nullptr
);

}