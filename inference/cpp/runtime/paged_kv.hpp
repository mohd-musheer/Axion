#pragma once

#include "../core/tensor.hpp"

#include <vector>

namespace axion {

struct KVPage {

    Tensor keys;

    Tensor values;

    int64_t used = 0;
};

class PagedKVCache {

public:

    int64_t page_size = 128;

    int64_t hidden_size = 0;

    std::vector<KVPage> pages;

    void initialize(
        int64_t hidden,
        int64_t page
    );

    void append(
        const Tensor& K,
        const Tensor& V
    );

    Tensor materialize_keys();

    Tensor materialize_values();
};

}