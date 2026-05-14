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

    if (cache.keys.data.empty()) {

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

    cache.keys.data.insert(
        cache.keys.data.end(),
        K.data.begin(),
        K.data.end()
    );

    // -------------------------
    // APPEND VALUES
    // -------------------------

    cache.values.data.insert(
        cache.values.data.end(),
        V.data.begin(),
        V.data.end()
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