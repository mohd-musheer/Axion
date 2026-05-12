
#include "transpose.hpp"

#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace axion {

Tensor transpose(
    const Tensor& input
) {

    if (input.shape.size() != 2) {

        throw std::runtime_error(
            "transpose only supports 2D tensors"
        );
    }

    int64_t rows = input.shape[0];
    int64_t cols = input.shape[1];

    Tensor output;

    output.name = input.name + "_T";

    output.dtype = input.dtype;

    output.shape = {
        cols,
        rows
    };

    output.data.resize(
        rows * cols
    );

    #pragma omp parallel for collapse(2)
    for (int64_t r = 0; r < rows; r++) {

        for (int64_t c = 0; c < cols; c++) {

            output.data[
                c * rows + r
            ] =
            input.data[
                r * cols + c
            ];
        }
    }

    return output;
}

}
