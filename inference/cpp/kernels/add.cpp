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

    Tensor out = a;

    for (size_t i = 0;
         i < a.data.size();
         i++) {

        out.data[i] =
            a.data[i] +
            b.data[i];
    }

    return out;
}

}