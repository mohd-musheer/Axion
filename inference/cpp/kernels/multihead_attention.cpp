#include "multihead_attention.hpp"

#include "multihead.hpp"
#include "attention.hpp"
#include "causal_mask.hpp"
#include "softmax.hpp"
#include "attention_output.hpp"

#include <vector>

namespace axion {

Tensor multihead_attention(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    int num_heads,
    RuntimeMemoryScheduler* scheduler
) {

    // -------------------------
    // SPLIT HEADS
    // -------------------------

    std::vector<Tensor> q_heads =
        split_heads(Q, num_heads);

    std::vector<Tensor> k_heads =
        split_heads(K, num_heads);

    std::vector<Tensor> v_heads =
        split_heads(V, num_heads);

    // -------------------------
    // PROCESS EACH HEAD
    // -------------------------

    std::vector<Tensor> outputs;

    for (int h = 0;
         h < num_heads;
         h++) {

        // --------------------------------
        // QK^T
        // --------------------------------

        Tensor scores =
            attention_scores(
                q_heads[h],
                k_heads[h],
                scheduler
            );

        // --------------------------------
        // MASK
        // --------------------------------

        Tensor masked =
            causal_mask(
                scores,
                scheduler
            );

        // scores no longer needed

        if (scheduler != nullptr) {

            scheduler->release_tensor(
                scores.name
            );
        }

        // --------------------------------
        // SOFTMAX
        // --------------------------------

        Tensor probs =
            softmax(
                masked,
                scheduler
            );

        // --------------------------------
        // ATTENTION OUTPUT
        // --------------------------------

        Tensor out =
            attention_output(
                probs,
                v_heads[h],
                scheduler
            );

        // probs no longer needed

        if (scheduler != nullptr) {

            scheduler->release_tensor(
                probs.name
            );
        }

        outputs.push_back(out);
    }

    // -------------------------
    // MERGE HEADS
    // -------------------------

    Tensor merged =
        merge_heads(outputs);

    merged.name =
        "multihead_attention_output";

    return merged;
}

}