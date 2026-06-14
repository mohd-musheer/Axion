#include "residual.hpp"

#include "../core/tensor_factory.hpp"

#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace axion {

Tensor residual_add(
    const Tensor& x,
    const Tensor& y,
    RuntimeMemoryScheduler* scheduler
) {

    if (!x.valid() || !y.valid()) {

        throw std::runtime_error(
            "Invalid tensor in residual_add"
        );
    }

    if (x.shape != y.shape) {

        throw std::runtime_error(
            "Residual shape mismatch"
        );
    }

    Tensor output;

    // --------------------------------
    // SCHEDULER ALLOCATION
    // --------------------------------

    if (scheduler != nullptr) {

        output =
            scheduler->request_tensor(
                "residual_output",
                x.shape,
                x.dtype
            );
    }

    // --------------------------------
    // FALLBACK
    // --------------------------------

    else {

        output =
            create_owned_tensor(
                x.shape,
                x.dtype
            );
    }

    output.name =
        "residual_output";

    #pragma omp parallel for
    for (int64_t i = 0;
         i < x.numel();
         i++) {

        output.data()[i] =
            x.value(i) +
            y.value(i);
    }

    return output;
}

}