#include "tensor_factory.hpp"

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

}