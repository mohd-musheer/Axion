#pragma once

#include "../core/tensor.hpp"
#include "../core/scheduler.hpp"

namespace axion {

Tensor attention_output(
    const Tensor& attention_probs,
    const Tensor& V,
    RuntimeMemoryScheduler* scheduler = nullptr
);

}