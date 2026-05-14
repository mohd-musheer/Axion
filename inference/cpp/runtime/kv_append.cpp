#include "kv_append.hpp"

#include <stdexcept>

namespace axion {

void append_kv_cache(
    LayerKVCache& cache,
    const Tensor& K,
    const Tensor& V
) {

    // -------------------------
    // FIRST INSERT
    // -------------------------

    if (cache.keys.owned_data.empty()) {

        cache.keys = K;
        cache.values = V;

        return;
    }

    // -------------------------
    // VALIDATION
    // -------------------------

    if (cache.keys.shape[1] != K.shape[1]) {

        throw std::runtime_error(
            "KV append hidden mismatch"
        );
    }

    // -------------------------
    // APPEND KEYS
    // -------------------------

    cache.keys.owned_data.insert(
        cache.keys.owned_data.end(),
        K.owned_data.begin(),
        K.owned_data.end()
    );

    // -------------------------
    // APPEND VALUES
    // -------------------------

    cache.values.owned_data.insert(
        cache.values.owned_data.end(),
        V.owned_data.begin(),
        V.owned_data.end()
    );

    // -------------------------
    // UPDATE SEQ LEN
    // -------------------------

    cache.keys.shape[0] +=
        K.shape[0];

    cache.values.shape[0] +=
        V.shape[0];
}

}