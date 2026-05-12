
#include "elementwise.hpp"

#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace axion {

Tensor elementwise_mul(
    const Tensor& a,
    const Tensor& b
) {

    if (a.shape != b.shape) {

        throw std::runtime_error(
            "elementwise shape mismatch"
        );
    }

    Tensor output;

    output.name =
        "elementwise_mul";

    output.dtype =
        a.dtype;

    output.shape =
        a.shape;

    output.data.resize(
        a.data.size()
    );

    #pragma omp parallel for
    for (int64_t i = 0;
         i < static_cast<int64_t>(a.data.size());
         i++) {

        output.data[i] =
            a.data[i] * b.data[i];
    }

    return output;
}

}

