#include "full_forward.hpp"

#include "transformer_stack.hpp"
#include "logits.hpp"

namespace axion {

Tensor full_forward(
    const Tensor& embeddings,
    MMapLoader& loader,
    int num_layers,
    int num_heads,
    const Tensor& embedding_matrix
) {

    // -------------------------
    // TRANSFORMER
    // -------------------------

    Tensor hidden =
        transformer_stack(
            embeddings,
            loader,
            num_layers,
            num_heads
        );

    // -------------------------
    // LOGITS
    // -------------------------

    Tensor logits =
        compute_logits(
            hidden,
            embedding_matrix
        );

    logits.name =
        "final_logits";

    return logits;
}

}