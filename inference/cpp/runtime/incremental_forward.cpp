#include "incremental_forward.hpp"

#include "token_embedding.hpp"
#include "single_position.hpp"
#include "cached_attention.hpp"
#include "linear.hpp"
#include "logits.hpp"

#include "../kernels/add.hpp"
#include "../kernels/transpose.hpp"
#include "../kernels/rmsnorm.hpp"
#include "../kernels/gelu.hpp"

namespace axion {

Tensor incremental_forward(
    int token_id,
    int position,
    const Tensor& embedding_matrix,
    const Tensor& position_matrix,
    MMapLoader& loader,
    KVState& kv_state,
    int num_layers,
    int num_heads
) {

    // -------------------------
    // TOKEN EMBEDDING
    // -------------------------

    Tensor hidden =
        token_embedding(
            embedding_matrix,
            token_id
        );

    // -------------------------
    // POSITION EMBEDDING
    // -------------------------

    Tensor pos =
        single_position_embedding(
            position_matrix,
            position
        );

    hidden =
        add(hidden, pos);

    // -------------------------
    // TRANSFORMER LAYERS
    // -------------------------

    for (int layer = 0;
         layer < num_layers;
         layer++) {

        // -------------------------
        // ATTENTION
        // -------------------------

        Tensor attn =
            cached_attention(
                hidden,
                loader,
                kv_state.layers[layer],
                "h." +
                std::to_string(layer) +
                ".attn.c_attn.weight",
                num_heads
            );

        // -------------------------
        // ATTN PROJ
        // -------------------------

        Tensor proj_weight =
            loader.load_tensor_data(
                "h." +
                std::to_string(layer) +
                ".attn.c_proj.weight"
            );

        Tensor proj_t =
            transpose(
                proj_weight
            );

        Tensor projected =
            linear(
                attn,
                proj_t
            );

        hidden =
            add(
                hidden,
                projected
            );

        // -------------------------
        // MLP
        // -------------------------

        Tensor fc_weight =
            loader.load_tensor_data(
                "h." +
                std::to_string(layer) +
                ".mlp.c_fc.weight"
            );

        Tensor fc_out =
            linear(
                hidden,
                fc_weight
            );

        Tensor activated =
            gelu(
                fc_out
            );

        Tensor proj_weight2 =
            loader.load_tensor_data(
                "h." +
                std::to_string(layer) +
                ".mlp.c_proj.weight"
            );

        Tensor mlp_out =
            linear(
                activated,
                proj_weight2
            );

        hidden =
            add(
                hidden,
                mlp_out
            );
    }

    // -------------------------
    // FINAL LOGITS
    // -------------------------

    Tensor logits =
        compute_logits(
            hidden,
            embedding_matrix
        );

    return logits;
}

}