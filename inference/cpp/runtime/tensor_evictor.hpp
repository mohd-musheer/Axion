#pragma once

#include "../core/tensor.hpp"

#include <vector>

namespace axion {

class TensorEvictor {

public:

    void register_tensor(
        Tensor* t
    );

    void evict_all();

private:

    std::vector<Tensor*> tensors;
};

}