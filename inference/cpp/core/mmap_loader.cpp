// inference/cpp/core/mmap_loader.cpp
#include "fp16.hpp"
#include "mmap_loader.hpp"

#include <fstream>
#include <iostream>
#include <cstring>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace axion {

bool MMapLoader::load_file(
    const std::string& path
) {

    std::ifstream file(
        path,
        std::ios::binary | std::ios::ate
    );

    if (!file.is_open()) {

        std::cerr << "Failed to open file"
                  << std::endl;

        return false;
    }

    std::streamsize size = file.tellg();

    file.seekg(0, std::ios::beg);

    file_data.resize(size);

    if (!file.read(file_data.data(), size)) {

        std::cerr << "Failed to read file"
                  << std::endl;

        return false;
    }

    std::memcpy(
        &header_size,
        file_data.data(),
        sizeof(uint64_t)
    );

    header_json = std::string(
        file_data.data() + 8,
        header_size
    );

    std::cout << "Header Size: "
              << header_size
              << std::endl;

    return true;
}

std::unordered_map<std::string, Tensor>
MMapLoader::parse_header() {

    std::unordered_map<std::string, Tensor> tensors;

    auto parsed = json::parse(header_json);

    for (auto& item : parsed.items()) {

        std::string tensor_name = item.key();

        if (tensor_name == "__metadata__") {
            continue;
        }

        auto tensor_info = item.value();

        Tensor t;

        t.name = tensor_name;

        std::string dtype =
            tensor_info["dtype"];

        if (dtype == "F16") {
            t.dtype = DType::FLOAT16;
        }
        else if (dtype == "F32") {
            t.dtype = DType::FLOAT32;
        }
        else {
            t.dtype = DType::UNKNOWN;
        }

        for (auto dim : tensor_info["shape"]) {
            t.shape.push_back(dim);
        }

        auto offsets = tensor_info["data_offsets"];

        t.offset_start =
            offsets[0];

        t.offset_end =
            offsets[1];

        tensors[tensor_name] = t;
    }

    return tensors;
}

Tensor MMapLoader::load_tensor(
    const std::string& tensor_name
) {

    auto tensors = parse_header();

    if (tensors.find(tensor_name)
        == tensors.end()) {

        throw std::runtime_error(
            "Tensor not found"
        );
    }

    Tensor tensor =
        tensors[tensor_name];

    tensor.print_info();

    return tensor;
}

Tensor MMapLoader::load_tensor_data(
    const std::string& tensor_name
) {

    auto tensors = parse_header();

    if (tensors.find(tensor_name)
        == tensors.end()) {

        throw std::runtime_error(
            "Tensor not found"
        );
    }

    Tensor tensor =
        tensors[tensor_name];

    uint64_t tensor_start =
        8 + header_size + tensor.offset_start;

    size_t total_elements =
        tensor.numel();

    tensor.data.resize(total_elements);

    if (tensor.dtype == DType::FLOAT16) {

        const uint16_t* raw_ptr =
            reinterpret_cast<const uint16_t*>(
                file_data.data() + tensor_start
            );

        for (size_t i = 0;
             i < total_elements;
             i++) {

            tensor.data[i] =
                fp16_to_fp32(raw_ptr[i]);
        }
    }
    else if (tensor.dtype == DType::FLOAT32) {

        const float* raw_ptr =
            reinterpret_cast<const float*>(
                file_data.data() + tensor_start
            );

        for (size_t i = 0;
             i < total_elements;
             i++) {

            tensor.data[i] =
                raw_ptr[i];
        }
    }

    return tensor;
}
std::vector<std::string>
MMapLoader::list_tensors() {

    auto parsed = json::parse(header_json);

    std::vector<std::string> names;

    for (auto& item : parsed.items()) {

        std::string tensor_name = item.key();

        if (tensor_name == "__metadata__") {
            continue;
        }

        names.push_back(tensor_name);
    }

    return names;
}



}