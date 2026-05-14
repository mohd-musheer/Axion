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

    // --------------------------------
    // POINTER-BASED STORAGE
    // --------------------------------

    float* data_ptr = nullptr;

    // --------------------------------
    // OPTIONAL OWNED STORAGE
    // for temporary tensors
    // --------------------------------

    std::vector<float> owned_data;

    Tensor();

    int64_t numel() const;

    size_t bytes() const;

    bool owns_data() const;

    float* data();

    const float* data() const;

    void print_info() const;
};

}