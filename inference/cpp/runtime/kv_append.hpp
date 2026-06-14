#pragma once

#include "kv_state.hpp"

namespace axion {

void append_kv_cache(
    LayerKVCache& cache,
    const Tensor& K,
    const Tensor& V
);

}