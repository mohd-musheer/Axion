#include "single_position.hpp"

#include <stdexcept>

namespace axion {

Tensor single_position_embedding(
    const Tensor& position_matrix,
    int position
) {

    if (position_matrix.shape.size() != 2) {

        throw std::runtime_error(
            "Position matrix must be 2D"
        );
    }

    int64_t max_pos =
        position_matrix.shape[0];

    int64_t hidden =
        position_matrix.shape[1];

    if (position >= max_pos) {

        throw std::runtime_error(
            "Position exceeds limit"
        );
    }

    Tensor out;

    out.name =
        "single_position_embedding";

    out.shape = {
        1,
        hidden
    };

    out.dtype =
        position_matrix.dtype;

    out.data.resize(hidden);

    for (int64_t h = 0;
         h < hidden;
         h++) {

        out.data[h] =
            position_matrix.data[
                position * hidden + h
            ];
    }

    return out;
}

}