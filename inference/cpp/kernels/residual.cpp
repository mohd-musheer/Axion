#include "residual.hpp"

#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace axion {

Tensor residual_add(
    const Tensor& x,
    const Tensor& y
) {

    if (x.shape != y.shape) {

        throw std::runtime_error(
            "Residual shape mismatch"
        );
    }

    Tensor output;

    output.name =
        "residual_output";

    output.dtype =
        x.dtype;

    output.shape =
        x.shape;

    output.owned_data.resize(
        x.numel()
    );

    #pragma omp parallel for
    for (int64_t i = 0;
         i < x.numel();
         i++) {

        output.data()[i] =
            x.value(i) +
            y.value(i);
    }

    return output;
}

}