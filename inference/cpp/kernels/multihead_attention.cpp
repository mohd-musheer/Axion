
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
    int num_heads
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

        // QK^T

        Tensor scores =
            attention_scores(
                q_heads[h],
                k_heads[h]
            );

        // causal mask

        Tensor masked =
            causal_mask(
                scores
            );

        // softmax

        Tensor probs =
            softmax(
                masked
            );

        // weighted values

        Tensor out =
            attention_output(
                probs,
                v_heads[h]
            );

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
