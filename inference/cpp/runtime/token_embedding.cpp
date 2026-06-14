#include "token_embedding.hpp"

#include <stdexcept>

namespace axion {

Tensor token_embedding(
    const Tensor& embedding_matrix,
    int token_id
) {

    if (embedding_matrix.shape.size() != 2) {

        throw std::runtime_error(
            "Embedding matrix must be 2D"
        );
    }

    int64_t vocab =
        embedding_matrix.shape[0];

    int64_t hidden =
        embedding_matrix.shape[1];

    if (token_id < 0 ||
        token_id >= vocab) {

        throw std::runtime_error(
            "Invalid token id"
        );
    }

    Tensor out;

    out.name =
        "single_token_embedding";

    out.shape = {
        1,
        hidden
    };

    out.dtype =
        embedding_matrix.dtype;

    out.owned_data.resize(hidden);

    for (int64_t h = 0;
         h < hidden;
         h++) {

        out.data()[h] =
            embedding_matrix.data()[
                token_id * hidden + h
            ];
    }

    return out;
}

}