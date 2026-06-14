
#pragma once

#include "../core/tensor.hpp"

#include <vector>

namespace axion {

class KVCache {

public:

    std::vector<Tensor> key_cache;

    std::vector<Tensor> value_cache;

    void add(
        const Tensor& K,
        const Tensor& V
    );

    Tensor get_all_keys() const;

    Tensor get_all_values() const;

    void clear();
};

}
