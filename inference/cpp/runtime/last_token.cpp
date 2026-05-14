#include "last_token.hpp"

#include <stdexcept>

namespace axion {

Tensor last_token(
    const Tensor& logits
) {

    if (logits.shape.size() != 2) {

        throw std::runtime_error(
            "Expected 2D logits tensor"
        );
    }

    int64_t rows =
        logits.shape[0];

    int64_t cols =
        logits.shape[1];

    Tensor out;

    out.name =
        "last_token_logits";

    out.shape = {
        1,
        cols
    };

    out.dtype =
        logits.dtype;

    out.owned_data.resize(cols);

    int64_t last_row =
        rows - 1;

    for (int64_t c = 0;
         c < cols;
         c++) {

        out.data()[c] =
            logits.data()[
                last_row * cols + c
            ];
    }

    return out;
}

}