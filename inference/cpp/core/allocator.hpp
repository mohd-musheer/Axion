#pragma once

#include <cstddef>
#include <cstdint>

namespace axion {

// Handle to one allocation. Returned by IAllocator::allocate and
// must be returned to the SAME allocator via deallocate().
struct AllocHandle {

    void* ptr = nullptr;

    size_t bytes = 0;

    size_t alignment = 0;

    uint64_t id = 0;

    bool valid() const {
        return ptr != nullptr;
    }
};

struct AllocatorStats {

    int64_t allocated_bytes = 0;     // currently outstanding

    int64_t peak_bytes = 0;

    int64_t allocation_count = 0;    // lifetime total

    int64_t active_allocations = 0;
};

// Byte-oriented allocator boundary.
//
// Byte-based (not element-based) so FP16/Q4/Q8 buffers are
// expressible; virtual so pool/arena backends and the Phase 17
// GPU backends implement the same interface.
class IAllocator {

public:

    // Cache line size; also satisfies AVX-512 load alignment.
    static constexpr size_t kDefaultAlignment = 64;

    virtual ~IAllocator() = default;

    virtual AllocHandle allocate(
        size_t bytes,
        size_t alignment = kDefaultAlignment
    ) = 0;

    virtual void deallocate(
        AllocHandle& handle
    ) = 0;

    virtual AllocatorStats stats() const = 0;

    virtual const char* name() const = 0;
};

// Platform shim: _aligned_malloc on Windows, posix_memalign on POSIX.
void* aligned_malloc(size_t bytes, size_t alignment);
void aligned_free(void* ptr);

class SystemAllocator : public IAllocator {

public:

    AllocHandle allocate(
        size_t bytes,
        size_t alignment = kDefaultAlignment
    ) override;

    void deallocate(
        AllocHandle& handle
    ) override;

    AllocatorStats stats() const override;

    const char* name() const override;

    // Process-wide default allocator.
    static SystemAllocator& instance();

private:

    AllocatorStats stats_;

    uint64_t next_id_ = 1;
};

}
