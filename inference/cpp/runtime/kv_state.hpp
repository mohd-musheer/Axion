#pragma once

#include "../core/tensor.hpp"

#include <vector>

namespace axion {

struct LayerKVCache {

    Tensor keys;

    Tensor values;
};

class KVState {

public:

    std::vector<LayerKVCache> layers;

    KVState(
        int num_layers
    ) {

        layers.resize(
            num_layers
        );
    }
};

}