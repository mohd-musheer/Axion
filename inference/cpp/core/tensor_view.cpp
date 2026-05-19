#include "tensor_view.hpp"

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

// --------------------------------
// SHARED STORAGE
// --------------------------------

    if (base.owns_data()) {

        view.parent_owned_data =
            const_cast<std::vector<float>*>(
                &base.owned_data
            );
    }
    else {

        view.data_ptr =
            const_cast<float*>(
                base.data()
            );
    }

    view.is_view = true;

    view.view_offset = offset;

    int64_t total = 1;

    for (auto s : shape) {
        total *= s;
    }

    view.view_numel = total;

    return view;
}

}