#include "q8.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace axion {

Q8Tensor quantize_q8(
    const Tensor& input
) {

    Q8Tensor q;

    q.shape =
        input.shape;

    // -------------------------
    // FIND MAX ABS
    // -------------------------

    float max_abs = 0.0f;

    for (int64_t i = 0;
         i < input.numel();
         i++) {

        float v =
            std::abs(
                input.value(i)
            );

        if (v > max_abs) {
            max_abs = v;
        }
    }

    // avoid divide by zero

    if (max_abs < 1e-8f) {
        max_abs = 1e-8f;
    }

    q.scale =
        max_abs / 127.0f;

    q.data.resize(
        input.numel()
    );

    // -------------------------
    // QUANTIZE
    // -------------------------

    for (int64_t i = 0;
         i < input.numel();
         i++) {

        float v =
            input.value(i);

        int8_t quantized =
            static_cast<int8_t>(
                std::round(
                    v / q.scale
                )
            );

        q.data[i] =
            quantized;
    }

    return q;
}

Tensor dequantize_q8(
    const Q8Tensor& q
) {

    Tensor output;

    output.name =
        "dequantized_q8";

    output.dtype =
        DType::FLOAT32;

    output.shape =
        q.shape;

    output.owned_data.resize(
        q.data.size()
    );

    for (int64_t i = 0;
         i < (int64_t)q.data.size();
         i++) {

        output.data()[i] =
            (float)q.data[i] *
            q.scale;
    }

    return output;
}

}