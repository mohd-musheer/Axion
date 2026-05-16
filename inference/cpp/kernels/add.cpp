#include "add.hpp"

#include "../core/tensor_factory.hpp"

#include <stdexcept>

namespace axion {

Tensor add(
    const Tensor& a,
    const Tensor& b,
    RuntimeMemoryScheduler* scheduler
) {

    if (!a.valid() || !b.valid()) {

        throw std::runtime_error(
            "Invalid tensor in add"
        );
    }

    if (a.shape != b.shape) {

        throw std::runtime_error(
            "Add shape mismatch"
        );
    }

    Tensor out;

    // --------------------------------
    // SCHEDULER ALLOCATION
    // --------------------------------

    if (scheduler != nullptr) {

        out =
            scheduler->request_tensor(
                "add_output",
                a.shape,
                a.dtype
            );
    }

    // --------------------------------
    // FALLBACK
    // --------------------------------

    else {

        out =
            create_owned_tensor(
                a.shape,
                a.dtype
            );
    }

    out.name =
        "add_output";

    for (int64_t i = 0;
         i < a.numel();
         i++) {

        out.data()[i] =
            a.value(i) +
            b.value(i);
    }

    return out;
}

}