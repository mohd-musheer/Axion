#include "gelu.hpp"
#include "../core/tensor_factory.hpp"

#include <cmath>
#include <stdexcept>

namespace axion {

Tensor gelu(
    const Tensor& input
) {

    if (!input.valid()) {

        throw std::runtime_error(
            "Invalid tensor in gelu"
        );
    }

    Tensor out =
        create_owned_tensor(
            input.shape,
            DType::FLOAT32
        );

    out.name =
        "gelu_output";

    for (int64_t i = 0;
         i < input.numel();
         i++) {

        float x =
            input.value(i);

        out.data()[i] =
            0.5f *
            x *
            (
                1.0f +
                std::tanh(
                    std::sqrt(
                        2.0f / M_PI
                    ) *
                    (
                        x +
                        0.044715f *
                        x *
                        x *
                        x
                    )
                )
            );
    }

    return out;
}

}