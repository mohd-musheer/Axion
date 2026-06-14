#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace axion {

class Arena;
class RuntimeMemoryScheduler;
class TensorBuffer;

enum class DType {
    FLOAT16,
    FLOAT32,
    UNKNOWN
};


enum class TensorStorage {

    OWNED,

    MMAP,

    SCHEDULER,

    ARENA,

    VIEW,

    // Phase 12: allocator-backed shared storage (TensorBuffer).
    BUFFER
};

class Tensor {

public:

    std::string name;

    DType dtype = DType::UNKNOWN;

    std::vector<int64_t> shape;

    uint64_t offset_start = 0;
    uint64_t offset_end = 0;
    Arena* arena = nullptr;
    TensorStorage storage =
    TensorStorage::OWNED;
    bool alive = true;
    int ref_count = 1;
    int pin_count = 0;
    std::string parent_tensor;

    // --------------------------------
    // MMAP POINTER
    // --------------------------------

    float* data_ptr = nullptr;

    uint16_t* fp16_ptr = nullptr;

    bool is_fp16 = false;

    // --------------------------------
    // OWNED STORAGE
    // (deprecated: migrating to TensorBuffer; removal planned
    //  for the end of Phase 12)
    // --------------------------------

    std::vector<float> owned_data;

    // --------------------------------
    // SHARED BUFFER STORAGE (Phase 12)
    //
    // When set, data_ptr aliases buffer->data() and the storage
    // is freed when the last Tensor/view referencing it dies.
    // Pin state shared across all copies lives in the buffer.
    // --------------------------------

    std::shared_ptr<TensorBuffer> buffer;

    // --------------------------------
    // VIEW SUPPORT
    // --------------------------------

    bool is_view = false;

    bool is_strided = false;

    int64_t stride = 0;

    int64_t view_offset = 0;

    int64_t view_numel = 0;

    // parent owned storage reference

    std::vector<float>* parent_owned_data =
        nullptr;


    Tensor();

    int64_t numel() const;

    size_t bytes() const;

    bool owns_data() const;

    float* data();

    const float* data() const;
        float value(
        int64_t idx
    ) const;

    void print_info() const;

    void release_view(
        RuntimeMemoryScheduler* scheduler
    );
    bool valid() const;
};



}
