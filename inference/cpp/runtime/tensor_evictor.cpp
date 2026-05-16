#include "tensor_evictor.hpp"

namespace axion {

void TensorEvictor::register_tensor(
    Tensor* t
) {

    tensors.push_back(t);
}

void TensorEvictor::evict_all() {

    for (Tensor* t : tensors) {

        if (t == nullptr) {
            continue;
        }

        // --------------------------------
        // CLEAR OWNED STORAGE
        // --------------------------------

        t->owned_data.clear();

        t->owned_data.shrink_to_fit();

        // --------------------------------
        // RESET POINTERS
        // --------------------------------

        t->data_ptr = nullptr;

        t->fp16_ptr = nullptr;

        t->parent_owned_data = nullptr;
    }

    tensors.clear();
}

}