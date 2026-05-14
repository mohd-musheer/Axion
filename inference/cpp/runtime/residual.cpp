
#include "residual.hpp"

#include <stdexcept>

namespace axion {

Tensor residual_add(
    const Tensor& a,
    const Tensor& b
) {

    if (a.shape != b.shape) {

        throw std::runtime_error(
            "Residual add shape mismatch"
        );
    }

    Tensor out;

    out.name =
        "residual_output";

    out.dtype =
        a.dtype;

    out.shape =
        a.shape;

    out.owned_data.resize(
        a.numel()
    );

    for (int64_t i = 0;
         i < a.numel();
         i++) {

        out.data()[i] =
            a.data()[i] +
            b.data()[i];
    }

    return out;
}

}
