
#pragma once

#include "../core/tensor.hpp"
#include "../core/mmap_loader.hpp"

#include "kv_state.hpp"

#include <string>

namespace axion {

Tensor cached_attention(
    const Tensor& hidden,
    MMapLoader& loader,
    LayerKVCache& cache,
    const std::string& fused_name,
    int num_heads
);

}