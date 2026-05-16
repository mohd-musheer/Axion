#include "softmax.hpp"
#include "../core/tensor_factory.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace axion {

Tensor softmax(
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
            "softmax expects 2D tensor"
        );
    }

    int64_t rows =
        input.shape[0];

    int64_t cols =
        input.shape[1];

    Tensor output;

    if (scheduler != nullptr) {

        output =
            scheduler->request_tensor(
                "softmax_output",
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

    #pragma omp parallel for
    for (int64_t r = 0; r < rows; r++) {

        float max_val =
            -1e30f;

        // -------------------------
        // FIND MAX
        // -------------------------

        for (int64_t c = 0; c < cols; c++) {

            float val =
                input.value(
                    r * cols + c
                );

            if (val > max_val) {
                max_val = val;
            }
        }

        // -------------------------
        // EXP
        // -------------------------

        float sum =
            0.0f;

        for (int64_t c = 0; c < cols; c++) {

            float exp_val =
                std::exp(
                    input.value(
                        r * cols + c
                    ) - max_val
                );

            output.data()[
                r * cols + c
            ] = exp_val;

            sum += exp_val;
        }

        // -------------------------
        // NORMALIZE
        // -------------------------

        for (int64_t c = 0; c < cols; c++) {

            output.data()[
                r * cols + c
            ] /= sum;
        }
    }

    return output;
}

}