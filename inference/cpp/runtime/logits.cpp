
#include "logits.hpp"
#include "../kernels/transpose.hpp"

#include "../kernels/blas.hpp"

#include <limits>

namespace axion {

Tensor compute_logits(
    const Tensor& hidden_states,
    const Tensor& embedding_matrix
) {

    // embedding:
    // [vocab, hidden]

    // need transpose:
    // [hidden, vocab]

    Tensor W_t =
        transpose(
            embedding_matrix
        );

    // logits:
    // [tokens, vocab]

    return matmul(
        hidden_states,
        W_t
    );
}

int argmax(
    const Tensor& logits
) {

    if (logits.numel() == 0) {

        return -1;
    }

    // use last token logits

    int64_t vocab_size =
        logits.shape[1];

    int64_t last_row =
        logits.shape[0] - 1;

    int64_t offset =
        last_row * vocab_size;

    float best =
        -std::numeric_limits<float>::infinity();

    int best_idx = 0;

    for (int64_t i = 0;
         i < vocab_size;
         i++) {

        float val =
            logits.data()[
                offset + i
            ];

        if (val > best) {

            best = val;

            best_idx = i;
        }
    }

    return best_idx;
}

}
