#include "add.hpp"

#include "../core/tensor_factory.hpp"

#include <stdexcept>

namespace axion {

Tensor add(
    const Tensor& a,
    const Tensor& b
) {

    if (!a.valid() || !b.valid()) {

        throw std::runtime_error(
            "Invalid tensor in matmul"
        );
    }
    if (a.shape != b.shape) {

        throw std::runtime_error(
            "Add shape mismatch"
        );
    }

    Tensor out =
        create_owned_tensor(
            a.shape,
            a.dtype
        );

    out.name =
        "add_output";

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