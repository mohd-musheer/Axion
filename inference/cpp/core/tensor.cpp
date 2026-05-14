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

    return !owned_data.empty();
}

float* Tensor::data() {

    if (owns_data()) {
        return owned_data.data() + view_offset;
    }

    return data_ptr + view_offset;
}

const float* Tensor::data() const {

    if (owns_data()) {
        return owned_data.data() + view_offset;
    }

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

    return data_ptr
        ? data_ptr[actual_idx]
        : owned_data[actual_idx];
}
}