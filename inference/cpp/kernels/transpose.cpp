
#include "transpose.hpp"
#include "../core/tensor_factory.hpp"
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace axion {

Tensor transpose(
    const Tensor& input,
    RuntimeMemoryScheduler* scheduler
){
    if (!input.valid()) {

        throw std::runtime_error(
            "Invalid tensor in matmul"
        );
    }

    if (input.shape.size() != 2) {

        throw std::runtime_error(
            "transpose only supports 2D tensors"
        );
    }

    int64_t rows = input.shape[0];
    int64_t cols = input.shape[1];

Tensor output;

if (scheduler != nullptr) {

    output =
        scheduler->request_tensor(
            "transpose_output",
            {
                cols,
                rows
            },
            input.dtype
        );
}
else {

    output =
        create_owned_tensor(
            {
                cols,
                rows
            },
            input.dtype
        );
}
    output.name =
        "transpose_output";

    #pragma omp parallel for collapse(2)
    for (int64_t r = 0; r < rows; r++) {

        for (int64_t c = 0; c < cols; c++) {

            output.data()[
                c * rows + r
            ] =
            input.value(
                r * cols + c
            );
        }
    }

    return output;
}

}
