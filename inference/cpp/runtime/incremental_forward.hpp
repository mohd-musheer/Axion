#pragma once

#include "../core/tensor.hpp"
#include "../core/mmap_loader.hpp"

#include "kv_state.hpp"

namespace axion {

Tensor incremental_forward(
    int token_id,
    int position,
    const Tensor& embedding_matrix,
    const Tensor& position_matrix,
    MMapLoader& loader,
    KVState& kv_state,
    int num_layers,
    int num_heads
);

}