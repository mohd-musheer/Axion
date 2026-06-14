
#include "kv_cache.hpp"

#include <stdexcept>
#include <cstring>
#include <string>

namespace axion {

namespace {

// Copies src contents into dst as contiguous FP32, regardless of
// the source storage class (owned / mmap / view / scheduler / fp16).
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

// Deep-copies any tensor into self-contained owned FP32 storage so
// the cache never aliases memory that is later evicted or reused.
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

// Concatenates 2D [rows, hidden] entries along dim 0 into a single
// owned tensor. The previous implementation concatenated owned_data
// directly, which produced silently wrong results for any non-owned
// entry (empty owned_data appended nothing while shape[0] grew).
Tensor concat_rows(
    const std::vector<Tensor>& entries,
    const std::string& what
) {

    if (entries.empty()) {

        throw std::runtime_error(
            what + " cache empty"
        );
    }

    if (entries[0].shape.size() != 2) {

        throw std::runtime_error(
            what + " cache entries must be 2D"
        );
    }

    int64_t hidden =
        entries[0].shape[1];

    int64_t total_rows = 0;

    for (const auto& t : entries) {

        if (t.shape.size() != 2 ||
            t.shape[1] != hidden) {

            throw std::runtime_error(
                what + " cache entry shape mismatch"
            );
        }

        total_rows += t.shape[0];
    }

    Tensor out;

    out.storage = TensorStorage::OWNED;
    out.dtype = DType::FLOAT32;
    out.shape = { total_rows, hidden };

    out.owned_data.resize(
        static_cast<size_t>(
            total_rows * hidden
        )
    );

    int64_t offset = 0;

    for (const auto& t : entries) {

        copy_tensor_data(
            out.owned_data.data() + offset,
            t
        );

        offset += t.numel();
    }

    return out;
}

} // namespace

void KVCache::add(
    const Tensor& K,
    const Tensor& V
) {

    // Entries are deep-copied into owned storage so the cache can
    // never alias views, mmap pointers, or scheduler blocks that
    // may be invalidated while the cache is still alive.

    key_cache.push_back(
        materialize_owned_fp32(K)
    );

    value_cache.push_back(
        materialize_owned_fp32(V)
    );
}

Tensor KVCache::get_all_keys() const {

    Tensor output =
        concat_rows(
            key_cache,
            "Key"
        );

    output.name =
        "cached_keys";

    return output;
}

Tensor KVCache::get_all_values() const {

    Tensor output =
        concat_rows(
            value_cache,
            "Value"
        );

    output.name =
        "cached_values";

    return output;
}

void KVCache::clear() {

    key_cache.clear();

    value_cache.clear();
}

}
