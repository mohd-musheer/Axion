#pragma once

#include "../core/tensor.hpp"

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <cstdint>

namespace axion {

enum class GGUFType {
    F32  = 0,
    F16  = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15
};

struct GGUFMetadata {

    std::string key;

    uint32_t type;
};

struct GGUFTensorInfo {

    std::string name;

    GGUFType type;

    std::vector<int64_t> shape;

    uint64_t offset = 0;

    uint64_t element_count = 0;

    uint64_t byte_size = 0;
};

class GGUFLoader {

public:

    GGUFLoader() = default;

    ~GGUFLoader();

    bool load_file(
        const std::string& path
    );

    Tensor load_tensor(
        const std::string& name
    );

    std::vector<std::string>
    tensor_names() const;

private:

    void parse_metadata();
    
    void parse_tensor_directory();

    uint64_t aligned_offset(
        uint64_t offset,
        uint64_t alignment
    );

private:

    std::ifstream file;

    uint32_t version = 0;

    uint64_t tensor_count = 0;

    uint64_t metadata_count = 0;

    uint64_t tensor_data_offset = 0;

    std::unordered_map<
        std::string,
        GGUFMetadata
    > metadata;

    std::unordered_map<
        std::string,
        GGUFTensorInfo
    > tensors;
};

}