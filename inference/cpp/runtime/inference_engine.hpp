#pragma once

#include "../core/tensor.hpp"
#include <string>

namespace axion {

class InferenceEngine {
public:
    Tensor forward_layer(
        const Tensor& input,
        const std::string& layer_path
    );
};

}