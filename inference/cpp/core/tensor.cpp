#include "tensor.hpp"
#include "fp16.hpp"
#include <iostream>

namespace axion {

Tensor::Tensor() {}

int64_t Tensor::numel() const {

    if (is_view) {
        return view_numel;
    }

    int64_t total = 1;

    for (auto s : shape) {
        total *= s;
    }

    return total;
}

size_t Tensor::bytes() const {

    if (dtype == DType::FLOAT16) {
        return numel() * 2;
    }

    if (dtype == DType::FLOAT32) {
        return numel() * 4;
    }

    return 0;
}

bool Tensor::owns_data() const {

    return !is_view &&
           owned_data.data() != nullptr;
}

float* Tensor::data() {

    // -------------------------
    // OWNED STORAGE
    // -------------------------

    if (owns_data()) {

        return owned_data.data() +
               view_offset;
    }

    // -------------------------
    // VIEW INTO OWNED STORAGE
    // -------------------------

    if (parent_owned_data != nullptr) {

        return parent_owned_data->data() +
               view_offset;
    }

    // -------------------------
    // POINTER STORAGE
    // -------------------------

    return data_ptr + view_offset;
}

const float* Tensor::data() const {

    // -------------------------
    // OWNED STORAGE
    // -------------------------

    if (owns_data()) {

        return owned_data.data() +
               view_offset;
    }

    // -------------------------
    // VIEW INTO OWNED STORAGE
    // -------------------------

    if (parent_owned_data != nullptr) {

        return parent_owned_data->data() +
               view_offset;
    }

    // -------------------------
    // POINTER STORAGE
    // -------------------------

    return data_ptr + view_offset;
}

void Tensor::print_info() const {

    std::cout << "Tensor: "
              << name
              << std::endl;

    std::cout << "Shape: [";

    for (size_t i = 0; i < shape.size(); i++) {

        std::cout << shape[i];

        if (i + 1 != shape.size()) {
            std::cout << ", ";
        }
    }

    std::cout << "]" << std::endl;

    std::cout << "Elements: "
              << numel()
              << std::endl;

    std::cout << "Bytes: "
              << bytes()
              << std::endl;

    std::cout << "Owns Data: "
              << owns_data()
              << std::endl;

    std::cout << "Is View: "
              << is_view
              << std::endl;

    std::cout << "View Offset: "
              << view_offset
              << std::endl;
}




bool Tensor::valid() const {

    if (shape.empty()) {

        return false;
    }

    // -------------------------
    // FP16 MMAP
    // -------------------------

    if (is_fp16) {

        return fp16_ptr != nullptr;
    }

    // -------------------------
    // OWNED
    // -------------------------

    if (owns_data()) {

        return
            owned_data.size() >=
            static_cast<size_t>(numel());
    }

    // -------------------------
    // VIEW INTO OWNED
    // -------------------------

    if (parent_owned_data != nullptr) {

        return true;
    }

    // -------------------------
    // RAW POINTER
    // -------------------------

    return data_ptr != nullptr;
}


float Tensor::value(
    int64_t idx
) const {
    

    int64_t actual_idx;

    if (is_strided) {

        int64_t row =
            idx / shape[1];

        int64_t col =
            idx % shape[1];

        actual_idx =
            view_offset +
            row * stride +
            col;
    }
    else {

        actual_idx =
            idx + view_offset;
    }

    if (is_fp16) {

        return fp16_to_fp32(
            fp16_ptr[actual_idx]
        );
    }

    if (parent_owned_data != nullptr) {

        return (*parent_owned_data)
            [actual_idx];
    }

    if (data_ptr != nullptr) {

        return data_ptr[actual_idx];
    }

    return owned_data[actual_idx];
    }
    
}
