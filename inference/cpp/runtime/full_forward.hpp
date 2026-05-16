#pragma once

#include "../core/tensor.hpp"
#include "../core/mmap_loader.hpp"
#include "../core/scheduler.hpp"

namespace axion {

Tensor full_forward(
    const Tensor& input_ids,
    MMapLoader& loader,
    int num_layers,
    int num_heads,
    const Tensor& final_norm_weight,
    RuntimeMemoryScheduler* scheduler = nullptr
);

}