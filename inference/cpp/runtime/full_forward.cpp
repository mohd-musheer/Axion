#include "full_forward.hpp"

#include "embedding.hpp"
#include "transformer_stack.hpp"
#include "final_norm.hpp"

namespace axion {

Tensor full_forward(
    const Tensor& input_ids,
    MMapLoader& loader,
    int num_layers,
    int num_heads,
    const Tensor& final_norm_weight,
    RuntimeMemoryScheduler* scheduler
) {

    // --------------------------------
    // TOKEN EMBEDDINGS
    // --------------------------------

    Tensor token_embedding_weight =
        loader.load_tensor_data(
            "wte.weight"
        );

    std::vector<int> token_ids;

    for (int64_t i = 0;
        i < input_ids.numel();
        i++) {

        token_ids.push_back(
            static_cast<int>(
                input_ids.value(i)
            )
        );
    }

    Tensor embeddings =
        embedding_lookup(
            token_embedding_weight,
            token_ids
        );

    // --------------------------------
    // TRANSFORMER
    // --------------------------------

    Tensor hidden =
        transformer_stack(
            embeddings,
            loader,
            num_layers,
            num_heads,
            scheduler
        );

    // embeddings no longer needed

    if (scheduler != nullptr) {

        scheduler->release_tensor(
            embeddings.name
        );
    }

    // --------------------------------
    // FINAL NORM
    // --------------------------------

    Tensor output =
        final_norm(
            hidden,
            final_norm_weight,
            scheduler
        );

    if (scheduler != nullptr) {

        scheduler->release_tensor(
            hidden.name
        );
    }

    output.name =
        "full_forward_output";

    return output;
}

}