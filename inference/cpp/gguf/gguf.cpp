#include "gguf.hpp"
#include "../core/tensor_factory.hpp"

#include <stdexcept>
#include <iostream>
#include <cstring>

namespace axion {

bool GGUFLoader::load_file(
    const std::string& path
) {

    file.open(
        path,
        std::ios::binary
    );

    if (!file.is_open()) {

        throw std::runtime_error(
            "Failed to open GGUF"
        );
    }

    // -------------------------
    // MAGIC
    // -------------------------

    char magic[4];

    file.read(
        magic,
        4
    );

    if (
        magic[0] != 'G' ||
        magic[1] != 'G' ||
        magic[2] != 'U' ||
        magic[3] != 'F'
    ) {

        throw std::runtime_error(
            "Invalid GGUF"
        );
    }

    // -------------------------
    // VERSION
    // -------------------------

    uint32_t version;

    file.read(
        reinterpret_cast<char*>(
            &version
        ),
        sizeof(version)
    );

    std::cout
        << "GGUF VERSION: "
        << version
        << std::endl;

    return true;
}

Tensor GGUFLoader::load_tensor(
    const std::string& name
) {

    if (!tensors.count(name)) {

        throw std::runtime_error(
            "Tensor not found"
        );
    }

    auto& info =
        tensors[name];

    // -------------------------
    // SEEK TO TENSOR
    // -------------------------

    file.seekg(
        info.offset,
        std::ios::beg
    );

    Tensor t =
        create_owned_tensor(
            info.shape,
            DType::FLOAT32
        );

    t.name =
        info.name;

    // -------------------------
    // STREAM READ
    // -------------------------

    file.read(
        reinterpret_cast<char*>(
            t.data()
        ),
        info.byte_size
    );

    return t;
}

std::vector<std::string>
GGUFLoader::tensor_names() const {

    std::vector<std::string> names;

    for (auto& kv : tensors) {

        names.push_back(
            kv.first
        );
    }

    return names;
}

GGUFLoader::~GGUFLoader() {}

}