#include "mapped_file.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace axion {

bool MappedFile::open(
    const std::string& path
) {

    close();

#ifdef _WIN32

    HANDLE file = CreateFileA(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );

    if (file == INVALID_HANDLE_VALUE) {
        return false;
    }

    LARGE_INTEGER sz;

    if (!GetFileSizeEx(file, &sz) || sz.QuadPart == 0) {
        CloseHandle(file);
        return false;
    }

    HANDLE mapping = CreateFileMappingA(
        file,
        nullptr,
        PAGE_READONLY,
        0,
        0,
        nullptr
    );

    if (mapping == nullptr) {
        CloseHandle(file);
        return false;
    }

    void* view = MapViewOfFile(
        mapping,
        FILE_MAP_READ,
        0,
        0,
        0
    );

    if (view == nullptr) {
        CloseHandle(mapping);
        CloseHandle(file);
        return false;
    }

    data_ = static_cast<uint8_t*>(view);
    size_ = static_cast<size_t>(sz.QuadPart);
    file_handle_ = file;
    mapping_handle_ = mapping;

    return true;

#else

    int fd = ::open(path.c_str(), O_RDONLY);

    if (fd < 0) {
        return false;
    }

    struct stat st;

    if (fstat(fd, &st) != 0 || st.st_size == 0) {
        ::close(fd);
        return false;
    }

    void* map = mmap(
        nullptr,
        static_cast<size_t>(st.st_size),
        PROT_READ,
        MAP_PRIVATE,
        fd,
        0
    );

    if (map == MAP_FAILED) {
        ::close(fd);
        return false;
    }

    data_ = static_cast<uint8_t*>(map);
    size_ = static_cast<size_t>(st.st_size);
    fd_ = fd;

    return true;

#endif
}

void MappedFile::close() {

    if (data_ == nullptr) {
        return;
    }

#ifdef _WIN32

    UnmapViewOfFile(data_);

    if (mapping_handle_ != nullptr) {
        CloseHandle(static_cast<HANDLE>(mapping_handle_));
        mapping_handle_ = nullptr;
    }

    if (file_handle_ != nullptr) {
        CloseHandle(static_cast<HANDLE>(file_handle_));
        file_handle_ = nullptr;
    }

#else

    munmap(data_, size_);

    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }

#endif

    data_ = nullptr;
    size_ = 0;
}

MappedFile::~MappedFile() {
    close();
}

void MappedFile::release_pages(
    size_t offset,
    size_t length
) {

    if (!is_open() ||
        length == 0 ||
        offset >= size_) {
        return;
    }

    if (offset + length > size_) {
        length = size_ - offset;
    }

    // Align the range down to a page boundary; advice APIs are
    // page-granular.
    size_t page = 4096;

#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    page = static_cast<size_t>(si.dwPageSize);
#else
    long p = sysconf(_SC_PAGESIZE);
    if (p > 0) {
        page = static_cast<size_t>(p);
    }
#endif

    size_t start = (offset / page) * page;
    size_t end = offset + length;

    if (end <= start) {
        return;
    }

    size_t aligned_len = end - start;

#ifdef _WIN32
    // Removes the range from the working set; pages are reloaded
    // from the file mapping on next access.
    VirtualUnlock(data_ + start, aligned_len);
#else
    madvise(data_ + start, aligned_len, MADV_DONTNEED);
#endif
}

void MappedFile::release_all_pages() {
    release_pages(0, size_);
}

}
