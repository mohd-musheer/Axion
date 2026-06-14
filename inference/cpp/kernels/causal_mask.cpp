
#include "causal_mask.hpp"
#include "../core/tensor_factory.hpp"
#include <stdexcept>
#include "../core/scheduler.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace axion {

Tensor causal_mask(
    const Tensor& scores,
    RuntimeMemoryScheduler* scheduler
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

    Tensor masked;

    if (scheduler != nullptr) {

        masked =
            scheduler->request_tensor(
                "causal_mask_output",
                scores.shape,
                scores.dtype
            );
    }
    else {

        masked =
            create_owned_tensor(
                scores.shape,
                scores.dtype
            );
    }

    for (int64_t i = 0;
        i < scores.numel();
        i++) {

        masked.data()[i] =
            scores.value(i);
    }

    #pragma omp parallel for collapse(2)
    for (int64_t r = 0; r < rows; r++) {

        for (int64_t c = 0; c < cols; c++) {

            // mask future tokens

            if (c > r) {

                masked.data()[
                    r * cols + c
                ] = -1e30f;
            }
        }
    }

    masked.name =
        "causal_masked_scores";

    return masked;
}

}
