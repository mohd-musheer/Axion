#pragma once

#include "../gguf/gguf.hpp"
#include "../core/tensor.hpp"
#include "tensor_evictor.hpp"
#include <string>
#include <vector>
#include "../core/scheduler.hpp"

namespace axion {

class StreamingExecutor {

public:

    StreamingExecutor(
        GGUFLoader* loader
    );

    Tensor forward(
        const Tensor& input,
        int num_layers
    );

private:

    GGUFLoader* loader;
    TensorEvictor evictor;

    Tensor execute_layer(
        const Tensor& hidden,
        int layer_idx
    );

    void unload_tensor(
        Tensor& t
    );
};

}