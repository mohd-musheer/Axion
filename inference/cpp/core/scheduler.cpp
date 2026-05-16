#include "scheduler.hpp"

#include <iostream>
#include <stdexcept>

namespace axion {

RuntimeMemoryScheduler::RuntimeMemoryScheduler() {}

int64_t RuntimeMemoryScheduler::calculate_numel(
    const std::vector<int64_t>& shape
) const {

    int64_t total = 1;

    for (auto s : shape) {
        total *= s;
    }

    return total;
}

int64_t RuntimeMemoryScheduler::dtype_size(
    DType dtype
) const {

    switch (dtype) {

        case DType::FLOAT16:
            return 2;

        case DType::FLOAT32:
            return 4;

        default:
            return 4;
    }
}

MemoryBlock*
RuntimeMemoryScheduler::find_reusable_block(
    int64_t required_elements,
    DType dtype
) {

    for (auto& block : blocks) {

        if (
            !block.in_use &&
            block.dtype == dtype &&
            block.capacity >= required_elements
        ) {

            return &block;
        }
    }

    return nullptr;
}

Tensor RuntimeMemoryScheduler::request_tensor(
    const std::string& base_name,
    const std::vector<int64_t>& shape,
    DType dtype
) {

    std::string name =
        base_name + "_" +
        std::to_string(
            tensor_counter++
        );

    int64_t numel =
        calculate_numel(shape);

    MemoryBlock* reusable =
        find_reusable_block(
            numel,
            dtype
        );

    Tensor t;

    t.name = name;
    t.shape = shape;
    t.dtype = dtype;

    // --------------------------------
    // REUSE BLOCK
    // --------------------------------

    if (reusable != nullptr) {

        reusable->in_use = true;

        t.data_ptr =
            reusable->storage.data();

        active_tensors[name] =
            reusable->id;

        return t;
    }

    // --------------------------------
    // CREATE NEW BLOCK
    // --------------------------------

    MemoryBlock block;

    block.id =
        next_block_id++;

    block.capacity =
        numel;

    block.dtype =
        dtype;

    block.in_use = true;

    block.storage.resize(numel);

    current_bytes +=
        numel * dtype_size(dtype);

    if (current_bytes > peak_bytes) {

        peak_bytes =
            current_bytes;
    }

    blocks.push_back(
        std::move(block)
    );

    MemoryBlock& created =
        blocks.back();

    t.data_ptr =
        created.storage.data();

    active_tensors[name] =
        created.id;

    return t;
}

void RuntimeMemoryScheduler::release_tensor(
    const std::string& name
) {

    if (!active_tensors.count(name)) {
        return;
    }

    int64_t block_id =
        active_tensors[name];

    for (auto& block : blocks) {

        if (block.id == block_id) {

            block.in_use = false;

            break;
        }
    }

    active_tensors.erase(name);
}

void RuntimeMemoryScheduler::reset() {

    for (auto& block : blocks) {
        block.in_use = false;
    }

    active_tensors.clear();
}

void RuntimeMemoryScheduler::print_stats() const {

    std::cout
        << "\n=== RUNTIME MEMORY STATS ==="
        << std::endl;

    std::cout
        << "Blocks: "
        << blocks.size()
        << std::endl;

    std::cout
        << "Current Memory: "
        << current_bytes / (1024.0 * 1024.0)
        << " MB"
        << std::endl;

    std::cout
        << "Peak Memory: "
        << peak_bytes / (1024.0 * 1024.0)
        << " MB"
        << std::endl;
}

int64_t
RuntimeMemoryScheduler::current_memory_bytes() const {

    return current_bytes;
}

int64_t
RuntimeMemoryScheduler::peak_memory_bytes() const {

    return peak_bytes;
}

}