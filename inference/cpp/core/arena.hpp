#pragma once

#include <vector>
#include <cstdint>

namespace axion {

class Arena {

public:

    std::vector<float> buffer;

    int64_t offset = 0;

    Arena(
        int64_t size
    );

    float* allocate(
        int64_t elements
    );

    void reset();
};

}