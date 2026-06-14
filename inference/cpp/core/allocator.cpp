#include "allocator.hpp"

#include <cstdlib>
#include <new>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace axion {

void* aligned_malloc(
    size_t bytes,
    size_t alignment
) {

#ifdef _WIN32
    return _aligned_malloc(bytes, alignment);
#else
    void* ptr = nullptr;

    if (posix_memalign(&ptr, alignment, bytes) != 0) {
        return nullptr;
    }

    return ptr;
#endif
}

void aligned_free(
    void* ptr
) {

#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

AllocHandle SystemAllocator::allocate(
    size_t bytes,
    size_t alignment
) {

    if (bytes == 0) {
        bytes = 1;
    }

    // posix_memalign requires alignment to be a power of two
    // and a multiple of sizeof(void*).
    if (alignment < alignof(void*)) {
        alignment = alignof(void*);
    }

    void* ptr =
        aligned_malloc(bytes, alignment);

    if (ptr == nullptr) {
        throw std::bad_alloc();
    }

    AllocHandle handle;

    handle.ptr = ptr;
    handle.bytes = bytes;
    handle.alignment = alignment;
    handle.id = next_id_++;

    stats_.allocated_bytes +=
        static_cast<int64_t>(bytes);

    stats_.allocation_count++;
    stats_.active_allocations++;

    if (stats_.allocated_bytes > stats_.peak_bytes) {
        stats_.peak_bytes = stats_.allocated_bytes;
    }

    return handle;
}

void SystemAllocator::deallocate(
    AllocHandle& handle
) {

    if (!handle.valid()) {
        return;
    }

    aligned_free(handle.ptr);

    stats_.allocated_bytes -=
        static_cast<int64_t>(handle.bytes);

    stats_.active_allocations--;

    handle = AllocHandle{};
}

AllocatorStats SystemAllocator::stats() const {
    return stats_;
}

const char* SystemAllocator::name() const {
    return "system";
}

SystemAllocator& SystemAllocator::instance() {

    static SystemAllocator inst;

    return inst;
}

}
