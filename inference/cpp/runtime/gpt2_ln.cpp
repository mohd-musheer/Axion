#include "gpt2_ln.hpp"

#include <cmath>

namespace axion {

Tensor gpt2_ln(
    const Tensor& input,
    const Tensor& weight
) {

    Tensor out = input;

    int64_t rows =
        input.shape[0];

    int64_t cols =
        input.shape[1];

    for (int64_t r = 0;
         r < rows;
         r++) {

        float mean = 0.0f;

        for (int64_t c = 0;
             c < cols;
             c++) {

            mean += input.data()[
                r * cols + c
            ];
        }

        mean /= cols;

        float var = 0.0f;

        for (int64_t c = 0;
             c < cols;
             c++) {

            float v =
                input.data()[
                    r * cols + c
                ] - mean;

            var += v * v;
        }

        var /= cols;

        float inv_std =
            1.0f /
            std::sqrt(var + 1e-5f);

        for (int64_t c = 0;
             c < cols;
             c++) {

            float norm =
                (
                    input.data()[
                        r * cols + c
                    ] - mean
                ) * inv_std;

            out.data()[
                r * cols + c
            ] =
                norm *
                weight.data()[c];
        }
    }

    return out;
}

}