
#include "embedding.hpp"

#include <stdexcept>

namespace axion {

Tensor embedding_lookup(
    const Tensor& embedding_matrix,
    const std::vector<int>& token_ids
) {

    if (embedding_matrix.shape.size() != 2) {

        throw std::runtime_error(
            "Embedding matrix must be 2D"
        );
    }

    int64_t vocab_size =
        embedding_matrix.shape[0];

    int64_t hidden_size =
        embedding_matrix.shape[1];

    Tensor output;

    output.name =
        "token_embeddings";

    output.shape = {
        (int64_t)token_ids.size(),
        hidden_size
    };

    output.data.resize(
        token_ids.size() *
        hidden_size
    );

    for (size_t t = 0;
         t < token_ids.size();
         t++) {

        int token_id =
            token_ids[t];

        if (token_id < 0 ||
            token_id >= vocab_size) {

            throw std::runtime_error(
                "Invalid token id"
            );
        }

        int64_t row_start =
            token_id * hidden_size;

        int64_t out_start =
            t * hidden_size;

        for (int64_t h = 0;
             h < hidden_size;
             h++) {

            output.data[
                out_start + h
            ] =
                embedding_matrix.data[
                    row_start + h
                ];
        }
    }

    return output;
}

}
