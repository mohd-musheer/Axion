#include "paged_kv.hpp"

#include "../core/tensor_factory.hpp"

#include <stdexcept>
#include <cstring>

namespace axion {

void PagedKVCache::initialize(
    int64_t hidden,
    int64_t page
) {

    hidden_size =
        hidden;

    page_size =
        page;
}

void PagedKVCache::append(
    const Tensor& K,
    const Tensor& V
) {

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
    // COPY K
    // -------------------------

    std::memcpy(
        page.keys.data() +
        row * hidden_size,

        K.data(),

        hidden_size *
        sizeof(float)
    );

    // -------------------------
    // COPY V
    // -------------------------

    std::memcpy(
        page.values.data() +
        row * hidden_size,

        V.data(),

        hidden_size *
        sizeof(float)
    );

    page.used++;
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