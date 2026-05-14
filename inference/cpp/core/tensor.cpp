#include "tensor.hpp"

#include <iostream>

namespace axion {

Tensor::Tensor() {}

int64_t Tensor::numel() const {

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
        return owned_data.data();
    }

    return data_ptr;
}

const float* Tensor::data() const {

    if (owns_data()) {
        return owned_data.data();
    }

    return data_ptr;
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
}

}