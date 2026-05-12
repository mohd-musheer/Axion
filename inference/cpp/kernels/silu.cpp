
#include "silu.hpp"

#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace axion {

Tensor silu(
    const Tensor& input
) {

    Tensor output;

    output.name =
        "silu_output";

    output.dtype =
        input.dtype;

    output.shape =
        input.shape;

    output.data.resize(
        input.data.size()
    );

    #pragma omp parallel for
    for (int64_t i = 0;
         i < static_cast<int64_t>(input.data.size());
         i++) {

        float x =
            input.data[i];

        float sigmoid =
            1.0f /
            (
                1.0f +
                std::exp(-x)
            );

        output.data[i] =
            x * sigmoid;
    }

    return output;
}

}
