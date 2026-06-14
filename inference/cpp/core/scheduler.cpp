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

    MemoryBlock* best = nullptr;

    int64_t smallest_waste =
        INT64_MAX;

    for (auto& block : blocks) {

        if (block.in_use) {
            continue;
        }

        if (block.dtype != dtype) {
            continue;
        }

        if (block.capacity <
            required_elements) {

            continue;
        }

        int64_t waste =
            block.capacity -
            required_elements;

        if (waste < smallest_waste) {

            smallest_waste =
                waste;

            best = &block;
        }
    }

    return best;
}

Tensor RuntimeMemoryScheduler::request_tensor(
    const std::string& base_name,
    const std::vector<int64_t>& shape,
    DType dtype

) {
    execution_step++;

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
    t.storage =
        TensorStorage::SCHEDULER;

    t.name = name;
        lifetime_graph.register_tensor(
        name
    );
    t.shape = shape;
    t.dtype = dtype;

    // --------------------------------
    // REUSE BLOCK
    // --------------------------------

    if (reusable != nullptr) {

        reusable->reuse_count++;

        reusable->last_used_step =
            execution_step;

        reusable->wasted_elements =
            reusable->capacity - numel;

        reusable->in_use = true;

        t.data_ptr =
            reusable->storage.data();


        active_tensors[name] =
            reusable->id;
        tensor_registry[name] =
        t;

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

    block.last_used_step =
        execution_step;

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
    tensor_registry[name] =
        t;

    return t;
}

void RuntimeMemoryScheduler::release_tensor(
    const std::string& name
) {

    auto reg = tensor_registry.find(name);

    if (reg == tensor_registry.end()) {
        return;
    }

    Tensor& tensor = reg->second;

    if (tensor.storage ==
        TensorStorage::MMAP) {

        return;
    }

    // --------------------------------
    // DEFERRED RELEASE
    //
    // Blocked releases are queued and retried on unpin
    // or when a later release unblocks the lifetime graph.
    // Previously they were silently dropped, leaking the
    // block for the rest of the session.
    // --------------------------------

    if (tensor.pin_count > 0 ||
        !lifetime_graph.can_release(name)) {

        deferred_releases.insert(name);
        return;
    }

    auto it = active_tensors.find(name);

    if (it != active_tensors.end()) {

        int64_t block_id = it->second;

        for (auto& block : blocks) {

            if (block.id == block_id) {

                block.in_use = false;
                break;
            }
        }

        active_tensors.erase(it);
    }

    tensor_registry.erase(reg);
    lifetime_graph.release_tensor(name);
    deferred_releases.erase(name);

    // A successful release may unblock
    // lifetime-deferred tensors.
    drain_deferred();
}

void RuntimeMemoryScheduler::drain_deferred() {

    bool progress = true;

    while (progress) {

        progress = false;

        std::vector<std::string> candidates(
            deferred_releases.begin(),
            deferred_releases.end()
        );

        for (auto& name : candidates) {

            auto reg =
                tensor_registry.find(name);

            if (reg == tensor_registry.end()) {

                deferred_releases.erase(name);
                continue;
            }

            if (reg->second.pin_count == 0 &&
                lifetime_graph.can_release(name)) {

                deferred_releases.erase(name);
                release_tensor(name);
                progress = true;
            }
        }
    }
}

void RuntimeMemoryScheduler::reset() {

    for (auto& block : blocks) {
        block.in_use = false;
    }

    active_tensors.clear();
    tensor_registry.clear();
    deferred_releases.clear();
    lifetime_graph.reset();
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

    int64_t total_waste = 0;

    for (const auto& block : blocks) {

        total_waste +=
            block.wasted_elements;
    }

    std::cout
        << "Fragmentation Waste: "
        << total_waste * sizeof(float)
        / (1024.0 * 1024.0)
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

void RuntimeMemoryScheduler::pin_tensor(
    const std::string& name
) {

    auto it = tensor_registry.find(name);

    if (it == tensor_registry.end()) {
        return;
    }

    it->second.pin_count++;
}

void RuntimeMemoryScheduler::unpin_tensor(
    const std::string& name
) {

    auto it = tensor_registry.find(name);

    if (it == tensor_registry.end()) {
        return;
    }

    if (it->second.pin_count > 0) {

        it->second.pin_count--;

        if (it->second.pin_count == 0 &&
            deferred_releases.count(name)) {

            release_tensor(name);
        }
    }
}

}
