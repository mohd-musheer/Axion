#include "arena.hpp"

#include <stdexcept>

namespace axion {

Arena::Arena(
    int64_t size
) {

    buffer.resize(size);
}

float* Arena::allocate(
    int64_t elements
) {

    if (offset + elements >
        (int64_t)buffer.size()) {

        throw std::runtime_error(
            "Arena out of memory"
        );
    }

    float* ptr =
        buffer.data() + offset;

    offset += elements;

    return ptr;
}

void Arena::reset() {

    offset = 0;
}

}