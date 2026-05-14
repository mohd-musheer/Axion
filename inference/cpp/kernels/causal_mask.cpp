
#include "causal_mask.hpp"

#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace axion {

Tensor causal_mask(
    const Tensor& scores
) {

    if (scores.shape.size() != 2) {

        throw std::runtime_error(
            "causal_mask expects 2D tensor"
        );
    }

    int64_t rows =
        scores.shape[0];

    int64_t cols =
        scores.shape[1];

    Tensor output =
        scores;

    #pragma omp parallel for collapse(2)
    for (int64_t r = 0; r < rows; r++) {

        for (int64_t c = 0; c < cols; c++) {

            // mask future tokens

            if (c > r) {

                output.data()[
                    r * cols + c
                ] = -1e30f;
            }
        }
    }

    output.name =
        "causal_masked_scores";

    return output;
}

}
