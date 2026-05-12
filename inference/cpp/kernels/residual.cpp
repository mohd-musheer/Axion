
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

    output.data.resize(
        x.data.size()
    );

    #pragma omp parallel for
    for (int64_t i = 0;
         i < static_cast<int64_t>(x.data.size());
         i++) {

        output.data[i] =
            x.data[i] + y.data[i];
    }

    return output;
}

}
