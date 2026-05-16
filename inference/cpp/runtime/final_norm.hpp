#pragma once

#include "../core/tensor.hpp"
#include "../core/scheduler.hpp"
namespace axion {

Tensor final_norm(
    const Tensor& input,
    const Tensor& weight,
     RuntimeMemoryScheduler* scheduler = nullptr
);

}