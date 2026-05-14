#include "layernorm.hpp"
#include "../core/tensor_factory.hpp"
#include <cmath>
#include <stdexcept>

namespace axion {

Tensor layernorm(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    float eps
) {

    if (input.shape.size() != 2) {

        throw std::runtime_error(
            "LayerNorm expects 2D tensor"
        );
    }

    int64_t rows =
        input.shape[0];

    int64_t hidden =
        input.shape[1];

    Tensor output =
        create_owned_tensor(
            input.shape,
            DType::FLOAT32
        );
    output.name =
        "layernorm_output";

    for (int64_t r = 0;
         r < rows;
         r++) {

        // -------------------------
        // MEAN
        // -------------------------

        float mean = 0.0f;

        for (int64_t h = 0;
             h < hidden;
             h++) {

            mean += input.value(
                r * hidden + h
            );
        }

        mean /= hidden;

        // -------------------------
        // VARIANCE
        // -------------------------

        float var = 0.0f;

        for (int64_t h = 0;
             h < hidden;
             h++) {

            float x =
                input.value(
                    r * hidden + h
                ) - mean;

            var += x * x;
        }

        var /= hidden;

        float inv_std =
            1.0f /
            std::sqrt(
                var + eps
            );

        // -------------------------
        // NORMALIZE
        // -------------------------

        for (int64_t h = 0;
             h < hidden;
             h++) {

            float x =
                input.value(
                    r * hidden + h
                );

            float norm =
                (x - mean) *
                inv_std;

            norm *=
                weight.value(h);

            norm +=
                bias.value(h);

            output.data()[
                r * hidden + h
            ] = norm;
        }
    }

    return output;
}

}