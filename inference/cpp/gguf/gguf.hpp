#pragma once

#include "../core/tensor.hpp"

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>

namespace axion {

enum class GGUFType {

    F32 = 0,
    F16 = 1,
    Q8_0 = 8
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

    std::ifstream file;

    std::unordered_map<
        std::string,
        GGUFTensorInfo
    > tensors;


};

}