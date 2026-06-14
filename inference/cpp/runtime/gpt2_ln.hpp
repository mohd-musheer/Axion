#pragma once

#include "../core/tensor.hpp"
#include "../core/scheduler.hpp"
namespace axion {

Tensor gpt2_ln(
    const Tensor& input,
    const Tensor& weight,
        RuntimeMemoryScheduler* scheduler = nullptr
);

}