#pragma once

#include "tensor.hpp"

#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>

namespace axion {

struct MemoryBlock {

    std::vector<float> storage;

    bool in_use = false;

    int64_t capacity = 0;


    DType dtype = DType::FLOAT32;

    int64_t id = 0;
};

class RuntimeMemoryScheduler {

public:

    RuntimeMemoryScheduler();

    Tensor request_tensor(
        const std::string& name,
        const std::vector<int64_t>& shape,
        DType dtype = DType::FLOAT32
    );

    void release_tensor(
        const std::string& name
    );

    void reset();

    void print_stats() const;

    int64_t current_memory_bytes() const;

    int64_t peak_memory_bytes() const;

private:

    std::vector<MemoryBlock> blocks;

    std::unordered_map<
        std::string,
        int64_t
    > active_tensors;

    int64_t next_block_id = 0;

    int64_t current_bytes = 0;

    int64_t tensor_counter = 0;

    int64_t peak_bytes = 0;

    int64_t calculate_numel(
        const std::vector<int64_t>& shape
    ) const;

    int64_t dtype_size(
        DType dtype
    ) const;

    MemoryBlock* find_reusable_block(
        int64_t required_elements,
        DType dtype
    );
};

}