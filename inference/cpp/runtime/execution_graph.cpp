
#include "execution_graph.hpp"

#include "../core/mmap_loader.hpp"

#include "layer_scheduler.hpp"
#include "weight_lookup.hpp"

#include "../runtime/transformer_layer.hpp"

#include <iostream>

namespace axion {

Tensor execute_model(
    const std::string& safetensor_path,
    const Tensor& input
) {

    // -------------------------
    // LOAD MODEL
    // -------------------------

    MMapLoader loader;

    loader.load_file(
        safetensor_path
    );

    std::vector<std::string>
    tensor_names =
        loader.list_tensors();

    // -------------------------
    // DISCOVER LAYERS
    // -------------------------

    std::vector<std::string>
    layers =
        discover_layers(
            tensor_names
        );

    std::cout
        << "\nDiscovered "
        << layers.size()
        << " layers\n";

    // -------------------------
    // FORWARD PASS
    // -------------------------

    Tensor hidden =
        input;

    for (const auto& layer : layers) {

        std::cout
            << "\nExecuting "
            << layer
            << std::endl;

        // resolve tensors

        try {

            std::string q_proj =
                find_tensor_by_suffix(
                    tensor_names,
                    layer,
                    "q_proj"
                );

            std::string k_proj =
                find_tensor_by_suffix(
                    tensor_names,
                    layer,
                    "k_proj"
                );

            std::string v_proj =
                find_tensor_by_suffix(
                    tensor_names,
                    layer,
                    "v_proj"
                );

            std::cout
                << "Q: "
                << q_proj
                << std::endl;

            std::cout
                << "K: "
                << k_proj
                << std::endl;

            std::cout
                << "V: "
                << v_proj
                << std::endl;

        } catch (...) {

            std::cout
                << "Skipping incomplete layer"
                << std::endl;

            continue;
        }

        // simulated execution

        hidden =
            transformer_layer(
                hidden
            );
    }

    hidden.name =
        "model_output";

    return hidden;
}

}
