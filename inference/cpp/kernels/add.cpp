
#include "add.hpp"

#include <stdexcept>

namespace axion {

Tensor add(
    const Tensor& a,
    const Tensor& b
) {

    if (a.shape != b.shape) {

        throw std::runtime_error(
            "Add shape mismatch"
        );
    }

    Tensor out;

    out.name =
        "add_output";

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
            a.value(i) +
            b.value(i);
    }

    return out;
}

}
