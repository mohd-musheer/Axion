
#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace axion {

enum class DType {
    FLOAT16,
    FLOAT32,
    UNKNOWN
};

class Tensor {
public:

    std::string name;

    DType dtype = DType::UNKNOWN;

    std::vector<int64_t> shape;

    uint64_t offset_start = 0;
    uint64_t offset_end = 0;

    std::vector<float> data;

    Tensor();

    int64_t numel() const;

    size_t bytes() const;

    void print_info() const;
};

}

