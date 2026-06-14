#include "silu.hpp"
#include "../core/tensor_factory.hpp"

#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace axion {

Tensor silu(
    const Tensor& input,
    RuntimeMemoryScheduler* scheduler
){

    if (!input.valid()) {

        throw std::runtime_error(
            "Invalid tensor in silu"
        );
    }

    Tensor output;

    if (scheduler != nullptr) {

        output =
            scheduler->request_tensor(
                "silu_output",
                input.shape,
                input.dtype
            );
    }
    else {

        output =
            create_owned_tensor(
                input.shape,
                input.dtype
            );
    }

    output.name =
        "silu_output";

    #pragma omp parallel for
    for (int64_t i = 0;
         i < input.numel();
         i++) {

        float x =
            input.value(i);

        output.data()[i] =
            x /
            (
                1.0f +
                std::exp(-x)
            );
    }

    return output;
}

}