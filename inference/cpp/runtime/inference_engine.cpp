#include "inference_engine.hpp"
#include "../core/mmap_loader.hpp"

#include <iostream>

namespace axion {

Tensor InferenceEngine::forward_layer(
    const Tensor& input,
    const std::string& layer_path
) {

    MMapLoader loader;

    bool success = loader.load_file(layer_path);

    if (!success) {
        throw std::runtime_error(
            "Failed to load layer file"
        );
    }

    auto tensor = loader.load_tensor(
        "self_attn.q_proj.weight"
    );

    std::cout << "Loaded layer: "
              << layer_path
              << std::endl;

    tensor.print_info();

    Tensor output = input;

    return output;
}

}