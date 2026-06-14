// inference/cpp/core/mmap_loader.cpp
//
// True platform-backed memory mapping (Phase 12 MR4).

#include "fp16.hpp"
#include "mmap_loader.hpp"

#include <iostream>
#include <cstring>
#include <stdexcept>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace axion {

bool MMapLoader::load_file(
    const std::string& path
) {

    directory.clear();
    directory_parsed = false;
    header_json.clear();
    header_size = 0;
    mapping_buffer_.reset();

    mapped_ =
        std::make_shared<MappedFile>();

    if (!mapped_->open(path)) {

        std::cerr << "Failed to map file"
                  << std::endl;

        mapped_.reset();
        return false;
    }

    if (mapped_->size() < 8) {

        std::cerr << "File too small for safetensors header"
                  << std::endl;

        mapped_.reset();
        return false;
    }

    std::memcpy(
        &header_size,
        mapped_->data(),
        sizeof(uint64_t)
    );

    if (8 + header_size > mapped_->size()) {

        std::cerr << "Corrupt safetensors header size"
                  << std::endl;

        mapped_.reset();
        return false;
    }

    header_json.assign(
        reinterpret_cast<const char*>(mapped_->data()) + 8,
        header_size
    );

    // MR4 objective 6: wrap the mapping in the shared ownership
    // model. The keepalive holds the MappedFile alive for as long
    // as any tensor references it.
    mapping_buffer_ =
        TensorBuffer::wrap_external(
            const_cast<uint8_t*>(mapped_->data()),
            mapped_->size(),
            mapped_
        );

    std::cout << "Header Size: "
              << header_size
              << std::endl;

    return true;
}

void MMapLoader::ensure_directory() {

    if (directory_parsed) {
        return;
    }

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

        t.offset_start = offsets[0];
        t.offset_end = offsets[1];

        directory[tensor_name] = t;
    }

    directory_parsed = true;
}

std::unordered_map<std::string, Tensor>
MMapLoader::parse_header() {

    ensure_directory();

    return directory;
}

Tensor MMapLoader::load_tensor(
    const std::string& tensor_name
) {

    ensure_directory();

    auto it = directory.find(tensor_name);

    if (it == directory.end()) {

        throw std::runtime_error(
            "Tensor not found"
        );
    }

    Tensor tensor = it->second;

    tensor.print_info();

    return tensor;
}

std::vector<std::string>
MMapLoader::list_tensors() {

    ensure_directory();

    std::vector<std::string> names;

    names.reserve(directory.size());

    for (auto& kv : directory) {
        names.push_back(kv.first);
    }

    return names;
}

Tensor MMapLoader::load_tensor_data(
    const std::string& tensor_name
) {

    if (!mapped_ || !mapped_->is_open()) {

        throw std::runtime_error(
            "No file mapped"
        );
    }

    ensure_directory();

    auto it = directory.find(tensor_name);

    if (it == directory.end()) {

        throw std::runtime_error(
            "Tensor not found"
        );
    }

    Tensor tensor = it->second;

    tensor.storage =
        TensorStorage::MMAP;

    uint64_t tensor_start =
        8 + header_size +
        tensor.offset_start;

    uint64_t tensor_len =
        tensor.offset_end -
        tensor.offset_start;

    if (tensor_start + tensor_len > mapped_->size()) {

        throw std::runtime_error(
            "Tensor data exceeds mapped file"
        );
    }

    // --------------------------------
    // LAZY VIEW INTO THE MAPPING
    //
    // No heap copy. Pages fault in on first access and can be
    // returned to the OS via release_tensor_pages().
    // --------------------------------

    const uint8_t* base =
        mapped_->data() + tensor_start;

    if (tensor.dtype == DType::FLOAT32) {

        tensor.data_ptr =
            reinterpret_cast<float*>(
                const_cast<uint8_t*>(base)
            );

        tensor.is_fp16 = false;
    }
    else if (tensor.dtype == DType::FLOAT16) {

        tensor.fp16_ptr =
            reinterpret_cast<uint16_t*>(
                const_cast<uint8_t*>(base)
            );

        tensor.is_fp16 = true;
    }
    else {

        throw std::runtime_error(
            "Unsupported tensor dtype"
        );
    }

    // MR4 objective 6: the tensor co-owns the mapping. The file is
    // unmapped only when the loader and every tensor/view are gone.
    tensor.buffer = mapping_buffer_;

    return tensor;
}

void MMapLoader::release_tensor_pages(
    const std::string& tensor_name
) {

    if (!mapped_) {
        return;
    }

    ensure_directory();

    auto it = directory.find(tensor_name);

    if (it == directory.end()) {
        return;
    }

    uint64_t start =
        8 + header_size +
        it->second.offset_start;

    uint64_t len =
        it->second.offset_end -
        it->second.offset_start;

    mapped_->release_pages(
        static_cast<size_t>(start),
        static_cast<size_t>(len)
    );
}

void MMapLoader::release_all_pages() {

    if (mapped_) {
        mapped_->release_all_pages();
    }
}

size_t MMapLoader::mapped_bytes() const {
    return mapped_ ? mapped_->size() : 0;
}

const uint8_t* MMapLoader::mapped_base() const {
    return mapped_ ? mapped_->data() : nullptr;
}

}
