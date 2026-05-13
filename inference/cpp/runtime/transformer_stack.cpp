
#include "transformer_stack.hpp"
#include "../kernels/gelu.hpp"
#include "real_attention.hpp"
#include "../kernels/transpose.hpp"
#include "residual.hpp"
#include "final_norm.hpp"
#include "linear.hpp"
#include "gpt2_ln.hpp"

#include <iostream>

namespace axion {

Tensor transformer_stack(
    const Tensor& input,
    MMapLoader& loader,
    int num_layers,
    int num_heads
) {

    Tensor hidden =
        input;

    for (int layer = 0;
         layer < num_layers;
         layer++) {

        std::cout
            << "Running layer: "
            << layer
            << std::endl;

        // ---------------------------------
        // LOAD LN1 WEIGHT
        // ---------------------------------

        std::string ln1_name =
            "h." +
            std::to_string(layer) +
            ".ln_1.weight";

        Tensor ln1_weight =
            loader.load_tensor_data(
                ln1_name
            );

        // ---------------------------------
        // LAYER NORM
        // ---------------------------------

        Tensor ln_hidden =
            gpt2_ln(
                hidden,
                ln1_weight
            );

        // ---------------------------------
        // ATTENTION
        // ---------------------------------

        std::string attn_name =
            "h." +
            std::to_string(layer) +
            ".attn.c_attn.weight";

        Tensor attn_output =
            real_attention(
                ln_hidden,
                loader,
                attn_name,
                attn_name,
                attn_name,
                num_heads
            );

        // ---------------------------------
        // LOAD ATTN PROJ
        // ---------------------------------

        std::string proj_name =
            "h." +
            std::to_string(layer) +
            ".attn.c_proj.weight";

        Tensor proj_weight =
            loader.load_tensor_data(
                proj_name
            );

        // ---------------------------------
        // OUTPUT PROJECTION
        // ---------------------------------

        
        Tensor proj_weight_t =
            transpose(
                proj_weight
            );

        Tensor projected =
            linear(
                attn_output,
                proj_weight_t
            );



        // ---------------------------------
        // RESIDUAL
        // ---------------------------------

        hidden =
            residual_add(
                hidden,
                projected
            );

        // ---------------------------------
        // LN2
        // ---------------------------------

        std::string ln2_name =
            "h." +
            std::to_string(layer) +
            ".ln_2.weight";

        Tensor ln2_weight =
            loader.load_tensor_data(
                ln2_name
            );

        Tensor mlp_input =
            gpt2_ln(
                hidden,
                ln2_weight
            );

        // ---------------------------------
        // MLP FC
        // ---------------------------------

        std::string fc_name =
            "h." +
            std::to_string(layer) +
            ".mlp.c_fc.weight";

        Tensor fc_weight =
            loader.load_tensor_data(
                fc_name
            );

        Tensor fc_out =
            linear(
                mlp_input,
                fc_weight
            );

        // ---------------------------------
        // GELU
        // ---------------------------------

        Tensor activated =
            gelu(
                fc_out
            );

        // ---------------------------------
        // MLP PROJ
        // ---------------------------------

        std::string mlp_proj_name =
            "h." +
            std::to_string(layer) +
            ".mlp.c_proj.weight";

        Tensor mlp_proj_weight =
            loader.load_tensor_data(
                mlp_proj_name
            );

        Tensor mlp_out =
            linear(
                activated,
                mlp_proj_weight
            );

        // ---------------------------------
        // RESIDUAL
        // ---------------------------------

        hidden =
            residual_add(
                hidden,
                mlp_out
            );





        std::cout
            << "Output shape: ["
            << hidden.shape[0]
            << ", "
            << hidden.shape[1]
            << "]"
            << std::endl;
    }

    // ---------------------------------
    // FINAL NORM
    // ---------------------------------

    hidden =
        final_norm(
            hidden
        );

    return hidden;
}

}
