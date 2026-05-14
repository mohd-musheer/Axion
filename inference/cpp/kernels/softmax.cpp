
#include "softmax.hpp"

#include <cmath>
#include <stdexcept>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace axion {

Tensor softmax(
    const Tensor& input
) {

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

    output.name =
        "softmax_output";

    output.dtype =
        input.dtype;

    output.shape =
        input.shape;

    output.owned_data.resize(
        input.numel()
    );

    #pragma omp parallel for
    for (int64_t r = 0; r < rows; r++) {

        float max_val =
            -1e30f;

        for (int64_t c = 0; c < cols; c++) {

            float val =
                input.data()[
                    r * cols + c
                ];

            if (val > max_val) {
                max_val = val;
            }
        }

        float sum =
            0.0f;

        for (int64_t c = 0; c < cols; c++) {

            float exp_val =
                std::exp(
                    input.data()[
                        r * cols + c
                    ] - max_val
                );

            output.data()[
                r * cols + c
            ] = exp_val;

            sum += exp_val;
        }

        for (int64_t c = 0; c < cols; c++) {

            output.data()[
                r * cols + c
            ] /= sum;
        }
    }

    return output;
}

}
