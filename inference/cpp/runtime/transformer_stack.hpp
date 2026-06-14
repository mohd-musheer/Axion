#pragma once

#include "../core/tensor.hpp"
#include "../core/mmap_loader.hpp"
#include "../core/scheduler.hpp"

namespace axion {

Tensor transformer_stack(
    const Tensor& input,
    MMapLoader& loader,
    int num_layers,
    int num_heads,
    RuntimeMemoryScheduler* scheduler = nullptr
);

}