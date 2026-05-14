#include "q4.hpp"

#include "../core/tensor_factory.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace axion {

Q4Tensor quantize_q4(
    const Tensor& input
) {

    Q4Tensor q;

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

    if (max_abs < 1e-8f) {
        max_abs = 1e-8f;
    }

    q.scale =
        max_abs / 7.0f;

    // two q4 values per byte

    int64_t packed_size =
        (input.numel() + 1) / 2;

    q.data.resize(
        packed_size
    );

    // -------------------------
    // PACK
    // -------------------------

    for (int64_t i = 0;
         i < input.numel();
         i += 2) {

        int8_t q1 =
            static_cast<int8_t>(
                std::round(
                    input.value(i) /
                    q.scale
                )
            );

        q1 =
            std::max(
                (int8_t)-8,
                std::min(
                    (int8_t)7,
                    q1
                )
            );

        uint8_t packed =
            (q1 + 8) & 0x0F;

        // second nibble

        if (i + 1 < input.numel()) {

            int8_t q2 =
                static_cast<int8_t>(
                    std::round(
                        input.value(i + 1) /
                        q.scale
                    )
                );

            q2 =
                std::max(
                    (int8_t)-8,
                    std::min(
                        (int8_t)7,
                        q2
                    )
                );

            packed |=
                ((q2 + 8) & 0x0F)
                << 4;
        }

        q.data[i / 2] =
            packed;
    }

    return q;
}

Tensor dequantize_q4(
    const Q4Tensor& q
) {

    Tensor output =
        create_owned_tensor(
            q.shape,
            DType::FLOAT32
        );

    output.name =
        "dequantized_q4";

    int64_t total =
        output.numel();

    for (int64_t i = 0;
         i < total;
         i += 2) {

        uint8_t packed =
            q.data[i / 2];

        int8_t q1 =
            (packed & 0x0F) - 8;

        output.data()[i] =
            q1 * q.scale;

        if (i + 1 < total) {

            int8_t q2 =
                ((packed >> 4) & 0x0F) - 8;

            output.data()[i + 1] =
                q2 * q.scale;
        }
    }

    return output;
}

}