#pragma once

#include "tensor.hpp"
#include "arena.hpp"

#include <vector>

namespace axion {

class IAllocator;

Tensor create_tensor(
    Arena& arena,
    const std::vector<int64_t>& shape,
    DType dtype = DType::FLOAT32
);

Tensor create_owned_tensor(
    const std::vector<int64_t>& shape,
    DType dtype = DType::FLOAT32
);

// Phase 12: aligned, allocator-backed tensor with shared ownership
// (TensorBuffer). Default allocator is SystemAllocator::instance().
Tensor create_buffer_tensor(
    const std::vector<int64_t>& shape,
    DType dtype = DType::FLOAT32,
    IAllocator* allocator = nullptr
);

}
