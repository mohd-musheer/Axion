#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace axion {

class Arena;
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
    Arena* arena = nullptr;

    // --------------------------------
    // MMAP POINTER
    // --------------------------------

    float* data_ptr = nullptr;

    uint16_t* fp16_ptr = nullptr;

    bool is_fp16 = false;

    // --------------------------------
    // OWNED STORAGE
    // --------------------------------

    std::vector<float> owned_data;

    // --------------------------------
    // VIEW SUPPORT
    // --------------------------------

// --------------------------------
// VIEW SUPPORT
// --------------------------------

    bool is_view = false;

    bool is_strided = false;

    int64_t stride = 0;

    int64_t view_offset = 0;

    int64_t view_numel = 0;

    // parent owned storage reference

    std::vector<float>* parent_owned_data =
        nullptr;


    Tensor();

    int64_t numel() const;

    size_t bytes() const;

    bool owns_data() const;

    float* data();

    const float* data() const;
        float value(
        int64_t idx
    ) const;

    void print_info() const;

    bool valid() const;
};

}