#pragma once

#include "allocator.hpp"

#include <atomic>
#include <memory>
#include <utility>

namespace axion {

// Shared storage control block (Phase 12).
//
// All Tensor copies and views of the same storage share one
// TensorBuffer through shared_ptr, so:
//   - storage is freed exactly once, when the last reference dies
//   - pin state lives here and cannot diverge across Tensor copies
//
// Two ownership modes:
//   - allocator-backed: allocator_ != nullptr; destructor returns
//     the handle to its allocator
//   - externally-owned: allocator_ == nullptr; the destructor frees
//     nothing, but an optional keepalive reference (e.g., the
//     MappedFile behind an mmap region) is held until the last
//     tensor/view dies, making mapped tensors lifetime-safe
class TensorBuffer {

public:

    TensorBuffer(
        IAllocator* allocator,
        AllocHandle handle,
        std::shared_ptr<void> keepalive = nullptr
    )
        : allocator_(allocator),
          handle_(handle),
          keepalive_(std::move(keepalive)) {}

    ~TensorBuffer() {

        if (allocator_ != nullptr &&
            handle_.valid()) {

            allocator_->deallocate(handle_);
        }

        // keepalive_ released here: for mapped memory this is
        // where the mapping owner may finally be destroyed.
    }

    TensorBuffer(const TensorBuffer&) = delete;
    TensorBuffer& operator=(const TensorBuffer&) = delete;

    float* data() {
        return static_cast<float*>(handle_.ptr);
    }

    const float* data() const {
        return static_cast<const float*>(handle_.ptr);
    }

    void* raw() {
        return handle_.ptr;
    }

    size_t bytes() const {
        return handle_.bytes;
    }

    size_t alignment() const {
        return handle_.alignment;
    }

    bool externally_owned() const {
        return allocator_ == nullptr;
    }

    void pin() {
        pins_.fetch_add(1, std::memory_order_relaxed);
    }

    void unpin() {

        int expected =
            pins_.load(std::memory_order_relaxed);

        while (expected > 0 &&
               !pins_.compare_exchange_weak(
                   expected,
                   expected - 1,
                   std::memory_order_relaxed)) {
        }
    }

    int pins() const {
        return pins_.load(std::memory_order_relaxed);
    }

    static std::shared_ptr<TensorBuffer> create(
        IAllocator& allocator,
        size_t bytes,
        size_t alignment = IAllocator::kDefaultAlignment
    ) {

        return std::make_shared<TensorBuffer>(
            &allocator,
            allocator.allocate(bytes, alignment)
        );
    }

    // Wraps externally-owned memory (e.g., a mapped file region)
    // in the shared ownership model. The optional keepalive holds
    // the memory's true owner alive until the last reference dies.
    static std::shared_ptr<TensorBuffer> wrap_external(
        void* ptr,
        size_t bytes,
        std::shared_ptr<void> keepalive = nullptr
    ) {

        AllocHandle handle;

        handle.ptr = ptr;
        handle.bytes = bytes;
        handle.alignment = 0;
        handle.id = 0;

        return std::make_shared<TensorBuffer>(
            nullptr,
            handle,
            std::move(keepalive)
        );
    }

private:

    IAllocator* allocator_ = nullptr;

    AllocHandle handle_;

    std::atomic<int> pins_{0};

    std::shared_ptr<void> keepalive_;
};

}
