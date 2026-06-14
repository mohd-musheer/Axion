#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace axion {

// Read-only, platform-backed memory mapping of a file.
//
//   POSIX:   mmap / munmap / madvise
//   Windows: CreateFileMapping / MapViewOfFile / UnmapViewOfFile /
//            VirtualUnlock
//
// This is the foundation of the larger-than-RAM goal: tensor data
// is accessed directly from the page cache, pages fault in on first
// touch, and release_pages() lets the runtime return physical pages
// to the OS while keeping the mapping valid (next access refaults
// from disk).
class MappedFile {

public:

    MappedFile() = default;

    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    bool open(const std::string& path);

    void close();

    const uint8_t* data() const {
        return data_;
    }

    size_t size() const {
        return size_;
    }

    bool is_open() const {
        return data_ != nullptr;
    }

    // Advise the OS that [offset, offset + length) is not needed.
    // Physical pages may be reclaimed; the mapping stays valid and
    // pages fault back in from the file on next access.
    void release_pages(size_t offset, size_t length);

    void release_all_pages();

private:

    uint8_t* data_ = nullptr;

    size_t size_ = 0;

#ifdef _WIN32
    void* file_handle_ = nullptr;     // HANDLE
    void* mapping_handle_ = nullptr;  // HANDLE
#else
    int fd_ = -1;
#endif
};

}
