#pragma once

#include "../core/tensor.hpp"
#include "../core/mmap_loader.hpp"
#include "../core/scheduler.hpp"
namespace axion {

Tensor real_attention(
    const Tensor& hidden,
    MMapLoader& loader,
    const std::string& q_name,
    const std::string& k_name,
    const std::string& v_name,
    int num_heads,
        RuntimeMemoryScheduler* scheduler = nullptr
);

}