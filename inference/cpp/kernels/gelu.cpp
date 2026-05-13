#include "gelu.hpp"

#include <cmath>

namespace axion {

Tensor gelu(
    const Tensor& input
) {

    Tensor out = input;

    for (size_t i = 0;
         i < input.data.size();
         i++) {

        float x =
            input.data[i];

        out.data[i] =
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