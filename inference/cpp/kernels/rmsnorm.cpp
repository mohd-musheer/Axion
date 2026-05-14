
#include "rmsnorm.hpp"

#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace axion {

Tensor rmsnorm(
    const Tensor& input,
    const Tensor& weight,
    float eps
) {

    if (input.shape.size() != 2) {

        throw std::runtime_error(
            "rmsnorm expects 2D tensor"
        );
    }

    int64_t rows =
        input.shape[0];

    int64_t cols =
        input.shape[1];

    if (weight.numel() != cols) {

        throw std::runtime_error(
            "rmsnorm weight mismatch"
        );
    }

    Tensor output;

    output.name =
        "rmsnorm_output";

    output.dtype =
        DType::FLOAT32;

    output.shape =
        input.shape;

    output.owned_data.resize(
        input.numel()
    );

    #pragma omp parallel for
    for (int64_t r = 0;
         r < rows;
         r++) {

        float mean_square = 0.0f;

        // -------------------------
        // COMPUTE RMS
        // -------------------------

        for (int64_t c = 0;
             c < cols;
             c++) {

            float val =
                input.data()[
                    r * cols + c
                ];

            mean_square += val * val;
        }

        mean_square /= cols;

        float inv_rms =
            1.0f /
            std::sqrt(
                mean_square + eps
            );

        // -------------------------
        // NORMALIZE
        // -------------------------

        for (int64_t c = 0;
             c < cols;
             c++) {

            float normalized =
                input.data()[
                    r * cols + c
                ] * inv_rms;

            output.data()[
                r * cols + c
            ] =
                normalized *
                weight.data()[c];
        }
    }

    return output;
}

}
