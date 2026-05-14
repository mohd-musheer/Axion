
#include "kv_cache.hpp"

#include <stdexcept>

namespace axion {

void KVCache::add(
    const Tensor& K,
    const Tensor& V
) {

    key_cache.push_back(K);

    value_cache.push_back(V);
}

Tensor KVCache::get_all_keys() const {

    if (key_cache.empty()) {

        throw std::runtime_error(
            "Key cache empty"
        );
    }

    Tensor output =
        key_cache[0];

    for (size_t i = 1;
         i < key_cache.size();
         i++) {

        output.owned_data.insert(
            output.owned_data.end(),
            key_cache[i].owned_data.begin(),
            key_cache[i].owned_data.end()
        );

        output.shape[0] +=
            key_cache[i].shape[0];
    }

    output.name =
        "cached_keys";

    return output;
}

Tensor KVCache::get_all_values() const {

    if (value_cache.empty()) {

        throw std::runtime_error(
            "Value cache empty"
        );
    }

    Tensor output =
        value_cache[0];

    for (size_t i = 1;
         i < value_cache.size();
         i++) {

        output.owned_data.insert(
            output.owned_data.end(),
            value_cache[i].owned_data.begin(),
            value_cache[i].owned_data.end()
        );

        output.shape[0] +=
            value_cache[i].shape[0];
    }

    output.name =
        "cached_values";

    return output;
}

void KVCache::clear() {

    key_cache.clear();

    value_cache.clear();
}

}
