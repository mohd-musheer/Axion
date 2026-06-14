#include "tensor_factory.hpp"
#include "allocator.hpp"
#include "tensor_buffer.hpp"

#include <stdexcept>

namespace axion {

Tensor create_tensor(
    Arena& arena,
    const std::vector<int64_t>& shape,
    DType dtype
) {

    Tensor t;
    t.storage =
        TensorStorage::ARENA;

    t.shape = shape;

    t.dtype = dtype;

    t.arena = &arena;

    int64_t total = 1;

    for (auto s : shape) {
        total *= s;
    }

    t.data_ptr =
        arena.allocate(total);

    return t;
}

Tensor create_owned_tensor(
    const std::vector<int64_t>& shape,
    DType dtype
) {

    Tensor t;
    t.storage =
        TensorStorage::OWNED;

    t.shape = shape;

    t.dtype = dtype;

    int64_t total = 1;

    for (auto s : shape) {
        total *= s;
    }

    t.owned_data.resize(total);

    return t;
}

Tensor create_buffer_tensor(
    const std::vector<int64_t>& shape,
    DType dtype,
    IAllocator* allocator
) {

    if (dtype != DType::FLOAT32) {

        throw std::runtime_error(
            "create_buffer_tensor: only FLOAT32 is supported "
            "until quantized buffers land (Phase 16)"
        );
    }

    Tensor t;

    t.storage =
        TensorStorage::BUFFER;

    t.shape = shape;

    t.dtype = dtype;

    int64_t total = 1;

    for (auto s : shape) {
        total *= s;
    }

    IAllocator& alloc =
        (allocator != nullptr)
            ? *allocator
            : SystemAllocator::instance();

    t.buffer =
        TensorBuffer::create(
            alloc,
            static_cast<size_t>(total) *
            sizeof(float)
        );

    // data_ptr aliases the buffer so data()/value()/valid()
    // work unchanged through the existing pointer path.
    t.data_ptr =
        t.buffer->data();

    return t;
}

}
