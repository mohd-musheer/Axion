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
    int num_heads,
    RuntimeMemoryScheduler* scheduler
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
        // LN1 WEIGHT
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
        // LN1
        // ---------------------------------

        Tensor ln_hidden =
            gpt2_ln(
                hidden,
                ln1_weight,
                scheduler
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
                num_heads,
                scheduler
            );

        // release ln_hidden

        if (scheduler != nullptr) {

            scheduler->release_tensor(
                ln_hidden.name
            );
        }

        // ---------------------------------
        // PROJ WEIGHT
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
        // TRANSPOSE
        // ---------------------------------

        Tensor proj_weight_t =
            transpose(
                proj_weight,
                scheduler
            );

        // ---------------------------------
        // LINEAR
        // ---------------------------------

        Tensor projected =
            linear(
                attn_output,
                proj_weight_t,
                scheduler
            );

        if (scheduler != nullptr) {

            scheduler->release_tensor(
                proj_weight_t.name
            );

            scheduler->release_tensor(
                attn_output.name
            );
        }

        // ---------------------------------
        // RESIDUAL 1
        // ---------------------------------

        Tensor residual1 =
            residual_add(
                hidden,
                projected,
                scheduler
            );

        if (scheduler != nullptr) {

            scheduler->release_tensor(
                projected.name
            );

            if (hidden.name != input.name) {

                scheduler->release_tensor(
                    hidden.name
                );
            }
        }

        hidden =
            residual1;

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
                ln2_weight,
                scheduler
            );

        // ---------------------------------
        // FC WEIGHT
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
                fc_weight,
                scheduler
            );

        if (scheduler != nullptr) {

            scheduler->release_tensor(
                mlp_input.name
            );
        }

        // ---------------------------------
        // GELU
        // ---------------------------------

        Tensor activated =
            gelu(
                fc_out,
                scheduler
            );

        if (scheduler != nullptr) {

            scheduler->release_tensor(
                fc_out.name
            );
        }

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
                mlp_proj_weight,
                scheduler
            );

        if (scheduler != nullptr) {

            scheduler->release_tensor(
                activated.name
            );
        }

        // ---------------------------------
        // RESIDUAL 2
        // ---------------------------------

        Tensor residual2 =
            residual_add(
                hidden,
                mlp_out,
                scheduler
            );

        if (scheduler != nullptr) {

            scheduler->release_tensor(
                hidden.name
            );

            scheduler->release_tensor(
                mlp_out.name
            );
        }

        hidden =
            residual2;

        std::cout
            << "Output shape: ["
            << hidden.shape[0]
            << ", "
            << hidden.shape[1]
            << "]"
            << std::endl;
    }

    // ---------------------------------
    // FINAL NORM WEIGHT
    // ---------------------------------

    Tensor final_norm_weight =
        loader.load_tensor_data(
            "ln_f.weight"
        );

    // ---------------------------------
    // FINAL NORM
    // ---------------------------------

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

    return output;
}

}