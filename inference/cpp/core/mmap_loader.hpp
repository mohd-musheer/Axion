// inference/cpp/core/mmap_loader.hpp

#pragma once

#include "tensor.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace axion {

class MMapLoader {

public:

    std::vector<char> file_data;

    bool load_file(const std::string& path);

    std::unordered_map<std::string, Tensor> parse_header();

    Tensor load_tensor(
        const std::string& tensor_name
    );
    Tensor load_tensor_data( const std::string& tensor_name );

    private:

        uint64_t header_size = 0;

        std::string header_json;





    };
    

}
