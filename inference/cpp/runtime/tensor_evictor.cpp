#include "tensor_evictor.hpp"

#include "../core/tensor_buffer.hpp"

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
        // SAFETY GUARDS
        // --------------------------------

        // Pinned tensors (e.g., KV-referenced or viewed)
        // must survive eviction.
        if (t->pin_count > 0) {
            continue;
        }

        // Scheduler-backed memory is owned by the scheduler;
        // evicting it here would corrupt block reuse accounting.
        if (t->storage == TensorStorage::SCHEDULER) {
            continue;
        }

        // Buffer-level pins are shared across all copies and
        // views; a pinned buffer must survive eviction.
        if (t->buffer && t->buffer->pins() > 0) {
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

        // Drop this tensor's reference to shared storage; the
        // memory returns to its allocator when the last
        // reference dies.
        t->buffer.reset();
    }

    tensors.clear();
}

}
