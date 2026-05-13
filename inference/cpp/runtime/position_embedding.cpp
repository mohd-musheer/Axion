#include "position_embedding.hpp"

#include <stdexcept>

namespace axion {

Tensor position_embedding_lookup(
    const Tensor& position_matrix,
    int seq_len
) {

    if (position_matrix.shape.size() != 2) {

        throw std::runtime_error(
            "Position matrix must be 2D"
        );
    }

    int64_t max_positions =
        position_matrix.shape[0];

    int64_t hidden_size =
        position_matrix.shape[1];

    if (seq_len > max_positions) {

        throw std::runtime_error(
            "Sequence exceeds max positions"
        );
    }

    Tensor out;

    out.name =
        "position_embeddings";

    out.shape = {
        seq_len,
        hidden_size
    };

    out.dtype =
        position_matrix.dtype;

    out.data.resize(
        seq_len * hidden_size
    );

    for (int pos = 0;
         pos < seq_len;
         pos++) {

        for (int64_t h = 0;
             h < hidden_size;
             h++) {

            out.data[
                pos * hidden_size + h
            ] =
            position_matrix.data[
                pos * hidden_size + h
            ];
        }
    }

    return out;
}

}