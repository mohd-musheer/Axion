
#include "transformer_layer.hpp"

#include "../kernels/rmsnorm.hpp"
#include "../kernels/attention.hpp"
#include "../kernels/softmax.hpp"
#include "../kernels/attention_output.hpp"
#include "../kernels/residual.hpp"
#include "../kernels/mlp.hpp"

namespace axion {

Tensor transformer_layer(
    const Tensor& input
) {

    // -------------------------
    // RMSNORM 1
    // -------------------------

    Tensor rms_weight;

    rms_weight.shape = {
        input.shape[1]
    };

    rms_weight.data.resize(
        input.shape[1],
        1.0f
    );

    Tensor norm1 =
        rmsnorm(
            input,
            rms_weight
        );

    // -------------------------
    // ATTENTION
    // -------------------------

    Tensor scores =
        attention_scores(
            norm1,
            norm1
        );

    Tensor probs =
        softmax(scores);

    Tensor attn =
        attention_output(
            probs,
            norm1
        );

    // -------------------------
    // RESIDUAL 1
    // -------------------------

    Tensor residual1 =
        residual_add(
            input,
            attn
        );

    // -------------------------
    // RMSNORM 2
    // -------------------------

    Tensor norm2 =
        rmsnorm(
            residual1,
            rms_weight
        );

    // -------------------------
    // MLP
    // -------------------------

    Tensor mlp =
        mlp_block(
            norm2,
            norm2
        );

    // -------------------------
    // RESIDUAL 2
    // -------------------------

    Tensor output =
        residual_add(
            residual1,
            mlp
        );

    output.name =
        "transformer_layer_output";

    return output;
}

}
