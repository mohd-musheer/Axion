#include "streaming_executor.hpp"

#include "../runtime/linear.hpp"
#include "../runtime/residual.hpp"
#include "../kernels/rmsnorm.hpp"

#include <iostream>

namespace axion {

StreamingExecutor::StreamingExecutor(
    GGUFLoader* loader
) : loader(loader) {}

RuntimeMemoryScheduler scheduler;

Tensor StreamingExecutor::forward(
    const Tensor& input,
    int num_layers
) {

    Tensor hidden = input;

    for (int i = 0; i < num_layers; i++) {

        std::cout
            << "Streaming Layer "
            << i
            << std::endl;

        hidden =
            execute_layer(
                hidden,
                i
            );
    }

    return hidden;
}

Tensor StreamingExecutor::execute_layer(
    const Tensor& hidden,
    int layer_idx
) {

    // --------------------------------
    // LOAD ONLY CURRENT LAYER
    // --------------------------------

    std::string prefix =
        "blk." +
        std::to_string(layer_idx);

    Tensor attn_norm =
        loader->load_tensor(
            prefix + ".attn_norm.weight"
        );

    Tensor ffn_norm =
        loader->load_tensor(
            prefix + ".ffn_norm.weight"
        );

    Tensor q_weight =
        loader->load_tensor(
            prefix + ".attn_q.weight"
        );

    Tensor k_weight =
        loader->load_tensor(
            prefix + ".attn_k.weight"
        );

    Tensor v_weight =
        loader->load_tensor(
            prefix + ".attn_v.weight"
        );

    Tensor out_weight =
        loader->load_tensor(
            prefix + ".attn_output.weight"
        );

    std::cout
        << "Loaded layer tensors"
        << std::endl;

    // --------------------------------
    // PLACEHOLDER EXECUTION
    // --------------------------------

    Tensor output = hidden;




    evictor.register_tensor(
        &attn_norm
    );

    evictor.register_tensor(
        &ffn_norm
    );

    evictor.register_tensor(
        &q_weight
    );

    evictor.register_tensor(
        &k_weight
    );

    evictor.register_tensor(
        &v_weight
    );

    evictor.register_tensor(
        &out_weight
    );

    // --------------------------------
    // UNLOAD
    // --------------------------------

    evictor.evict_all();

    return output;
}

void StreamingExecutor::unload_tensor(
    Tensor& t
) {

    t.owned_data.clear();

    t.owned_data.shrink_to_fit();
}

}