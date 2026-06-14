#include "paged_kv.hpp"

#include "../core/tensor_factory.hpp"

#include <stdexcept>
#include <cstring>

namespace axion {

namespace {

// Copies one row (hidden_size elements) of src into dst,
// regardless of the source storage class.
void copy_row(
    float* dst,
    const Tensor& src,
    int64_t row,
    int64_t hidden_size
) {

    if (!src.is_fp16 &&
        !src.is_strided) {

        std::memcpy(
            dst,
            src.data() + row * hidden_size,
            hidden_size * sizeof(float)
        );

        return;
    }

    for (int64_t i = 0; i < hidden_size; i++) {

        dst[i] =
            src.value(
                row * hidden_size + i
            );
    }
}

} // namespace

void PagedKVCache::initialize(
    int64_t hidden,
    int64_t page
) {

    if (hidden <= 0 || page <= 0) {

        throw std::runtime_error(
            "PagedKVCache: invalid initialize parameters"
        );
    }

    hidden_size =
        hidden;

    page_size =
        page;
}

void PagedKVCache::append(
    const Tensor& K,
    const Tensor& V
) {

    if (hidden_size <= 0) {

        throw std::runtime_error(
            "PagedKVCache: not initialized"
        );
    }

    if (K.shape.size() != 2 ||
        V.shape.size() != 2 ||
        K.shape[1] != hidden_size ||
        V.shape[1] != hidden_size) {

        throw std::runtime_error(
            "PagedKVCache: expected [seq, hidden] tensors"
        );
    }

    if (K.shape[0] != V.shape[0]) {

        throw std::runtime_error(
            "PagedKVCache: K/V row count mismatch"
        );
    }

    int64_t rows =
        K.shape[0];

    // The previous implementation copied exactly one row per call,
    // silently dropping every row past the first during prefill.
    // Rows are now appended individually and split across page
    // boundaries as pages fill up.

    for (int64_t r = 0; r < rows; r++) {

        if (pages.empty() ||

            pages.back().used >= page_size) {

            KVPage page;

            page.keys =
                create_owned_tensor(
                    {
                        page_size,
                        hidden_size
                    }
                );

            page.values =
                create_owned_tensor(
                    {
                        page_size,
                        hidden_size
                    }
                );

            page.used = 0;

            pages.push_back(
                std::move(page)
            );
        }

        auto& page =
            pages.back();

        int64_t row =
            page.used;

        // -------------------------
        // COPY K ROW
        // -------------------------

        copy_row(
            page.keys.data() +
            row * hidden_size,
            K,
            r,
            hidden_size
        );

        // -------------------------
        // COPY V ROW
        // -------------------------

        copy_row(
            page.values.data() +
            row * hidden_size,
            V,
            r,
            hidden_size
        );

        page.used++;
    }
}

Tensor PagedKVCache::materialize_keys() {

    int64_t total = 0;

    for (auto& p : pages) {
        total += p.used;
    }

    Tensor out =
        create_owned_tensor(
            {
                total,
                hidden_size
            }
        );

    int64_t offset = 0;

    for (auto& p : pages) {

        int64_t count =
            p.used * hidden_size;

        std::memcpy(
            out.data() + offset,
            p.keys.data(),
            count * sizeof(float)
        );

        offset += count;
    }

    return out;
}

Tensor PagedKVCache::materialize_values() {

    int64_t total = 0;

    for (auto& p : pages) {
        total += p.used;
    }

    Tensor out =
        create_owned_tensor(
            {
                total,
                hidden_size
            }
        );

    int64_t offset = 0;

    for (auto& p : pages) {

        int64_t count =
            p.used * hidden_size;

        std::memcpy(
            out.data() + offset,
            p.values.data(),
            count * sizeof(float)
        );

        offset += count;
    }

    return out;
}

}
