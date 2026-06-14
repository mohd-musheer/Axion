#include "kv_append.hpp"

#include <stdexcept>
#include <cstring>

namespace axion {

namespace {

// Copies src contents into dst as contiguous FP32, regardless of
// the source storage class (owned / mmap / view / scheduler / fp16).
// The previous implementation read src.owned_data directly, which is
// silently empty for every non-owned tensor: shape[0] grew while no
// data was appended, corrupting attention for all later tokens.
void copy_tensor_data(
    float* dst,
    const Tensor& src
) {

    int64_t n = src.numel();

    if (!src.is_fp16 &&
        !src.is_strided) {

        std::memcpy(
            dst,
            src.data(),
            n * sizeof(float)
        );

        return;
    }

    for (int64_t i = 0; i < n; i++) {
        dst[i] = src.value(i);
    }
}

// Deep-copies any tensor into self-contained owned FP32 storage.
// KV cache entries must never alias caller memory: views, mmap
// pointers and scheduler blocks can all be invalidated while the
// cache is still alive.
Tensor materialize_owned_fp32(
    const Tensor& src
) {

    Tensor t;

    t.storage = TensorStorage::OWNED;
    t.dtype = DType::FLOAT32;
    t.shape = src.shape;
    t.name = src.name;

    t.owned_data.resize(
        static_cast<size_t>(src.numel())
    );

    copy_tensor_data(
        t.owned_data.data(),
        src
    );

    return t;
}

} // namespace

void append_kv_cache(
    LayerKVCache& cache,
    const Tensor& K,
    const Tensor& V
) {

    // -------------------------
    // VALIDATION
    // -------------------------

    if (K.shape.size() != 2 ||
        V.shape.size() != 2) {

        throw std::runtime_error(
            "KV append expects 2D [seq, hidden] tensors"
        );
    }

    if (K.shape[0] != V.shape[0] ||
        K.shape[1] != V.shape[1]) {

        throw std::runtime_error(
            "KV append K/V shape mismatch"
        );
    }

    // -------------------------
    // FIRST INSERT
    //
    // Detected via empty shape (default-constructed cache slot),
    // NOT via owned_data.empty(): non-owned tensors always have
    // empty owned_data, which broke this test.
    // -------------------------

    if (cache.keys.shape.empty()) {

        cache.keys =
            materialize_owned_fp32(K);

        cache.values =
            materialize_owned_fp32(V);

        cache.keys.name =
            "kv_cache_keys";

        cache.values.name =
            "kv_cache_values";

        return;
    }

    if (cache.keys.shape[1] != K.shape[1]) {

        throw std::runtime_error(
            "KV append hidden mismatch"
        );
    }

    // Cache slots are writable from Python (LayerKVCache.keys is
    // exposed via pybind); re-materialize defensively if a
    // non-owned tensor was assigned externally.

    if (!cache.keys.owns_data()) {

        cache.keys =
            materialize_owned_fp32(
                cache.keys
            );
    }

    if (!cache.values.owns_data()) {

        cache.values =
            materialize_owned_fp32(
                cache.values
            );
    }

    // -------------------------
    // APPEND KEYS
    // -------------------------

    size_t old_k =
        cache.keys.owned_data.size();

    cache.keys.owned_data.resize(
        old_k +
        static_cast<size_t>(K.numel())
    );

    copy_tensor_data(
        cache.keys.owned_data.data() + old_k,
        K
    );

    // -------------------------
    // APPEND VALUES
    // -------------------------

    size_t old_v =
        cache.values.owned_data.size();

    cache.values.owned_data.resize(
        old_v +
        static_cast<size_t>(V.numel())
    );

    copy_tensor_data(
        cache.values.owned_data.data() + old_v,
        V
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
