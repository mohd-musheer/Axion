
#include "gelu.hpp"

#include <cmath>

namespace axion {

Tensor gelu(
    const Tensor& input
) {

    Tensor out;

    out.name =
        "gelu_output";

    out.dtype =
        input.dtype;

    out.shape =
        input.shape;

    out.owned_data.resize(
        input.numel()
    );

    for (int64_t i = 0;
         i < input.numel();
         i++) {

        float x =
            input.data()[i];

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
