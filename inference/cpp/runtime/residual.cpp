#include "residual.hpp"

#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace axion {

Tensor residual_add(
    const Tensor& a,
    const Tensor& b
) {

    if (a.shape != b.shape) {

        throw std::runtime_error(
            "Residual shape mismatch"
        );
    }

    Tensor out;

    out.name =
        "residual_output";

    out.shape =
        a.shape;

    out.dtype =
        a.dtype;

    out.data.resize(
        a.data.size()
    );

    #pragma omp parallel for
    for (int64_t i = 0;
         i < (int64_t)a.data.size();
         i++) {

        out.data[i] =
            a.data[i] +
            b.data[i];
    }

    return out;
}

}