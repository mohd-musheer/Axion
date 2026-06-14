#include "tensor_view.hpp"
#include "scheduler.hpp"

#include <stdexcept>

namespace axion {

Tensor tensor_view(
    const Tensor& base,
    int64_t offset,
    std::vector<int64_t> shape
) {

    Tensor view;

    view.storage =
        TensorStorage::VIEW;

    view.parent_tensor =
        base.name;

    view.name =
        base.name + "_view";

    view.dtype =
        base.dtype;

    view.shape =
        shape;

    view.fp16_ptr =
        base.fp16_ptr;

    view.is_fp16 =
        base.is_fp16;

    int64_t total = 1;

    for (auto s : shape) {
        total *= s;
    }

    if (offset < 0 ||
        offset + total > base.numel()) {

        throw std::out_of_range(
            "tensor_view: range exceeds base tensor '" +
            base.name + "'"
        );
    }

// --------------------------------
// SHARED STORAGE
//
// Views always resolve through a raw pointer. We must NOT store
// a pointer to base.owned_data: Tensor is a value type, so that
// pointer dangles as soon as the base copy is destroyed or moved.
// Lifetime is guaranteed either by the caller (this overload) or
// by scheduler pinning (the managed overload below).
// --------------------------------

    if (base.is_fp16 &&
        base.data_ptr == nullptr) {

        // FP16-only tensor: element access goes through
        // fp16_ptr + view_offset, no float pointer exists.
        view.data_ptr = nullptr;
    }
    else {

        view.data_ptr =
            const_cast<float*>(
                base.data()
            );
    }

    view.parent_owned_data = nullptr;

    // Phase 12: buffer-backed bases share the control block, so
    // the storage cannot be freed while any view is alive.
    view.buffer = base.buffer;

    view.is_view = true;

    view.view_offset = offset;

    view.view_numel = total;

    return view;
}

Tensor tensor_view(
    const Tensor& base,
    int64_t offset,
    std::vector<int64_t> shape,
    RuntimeMemoryScheduler* scheduler
) {

    Tensor view =
        tensor_view(
            base,
            offset,
            shape
        );

    if (scheduler != nullptr) {

        scheduler->pin_tensor(
            base.name
        );
    }

    return view;
}

}
