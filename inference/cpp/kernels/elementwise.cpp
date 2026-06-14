#include "elementwise.hpp"

#include "../core/tensor_factory.hpp"

#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace axion {

Tensor elementwise_mul(
    const Tensor& a,
    const Tensor& b,
    RuntimeMemoryScheduler* scheduler
) {

    if (!a.valid() || !b.valid()) {

        throw std::runtime_error(
            "Invalid tensor in elementwise_mul"
        );
    }

    if (a.shape != b.shape) {

        throw std::runtime_error(
            "elementwise shape mismatch"
        );
    }

    Tensor output;

    if (scheduler != nullptr) {

        output =
            scheduler->request_tensor(
                "elementwise_mul",
                a.shape,
                a.dtype
            );
    }
    else {

        output =
            create_owned_tensor(
                a.shape,
                a.dtype
            );
    }

    output.name =
        "elementwise_mul";

    #pragma omp parallel for
    for (int64_t i = 0;
         i < static_cast<int64_t>(a.numel());
         i++) {

        output.data()[i] =
            a.value(i) *
            b.value(i);
    }

    return output;
}

}