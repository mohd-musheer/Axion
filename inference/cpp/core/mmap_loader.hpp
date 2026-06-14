// inference/cpp/core/mmap_loader.hpp

#pragma once

#include "tensor.hpp"
#include "mapped_file.hpp"
#include "tensor_buffer.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace axion {

// Safetensors loader backed by a TRUE memory mapping (Phase 12 MR4).
//
// Tensors are lazy views into the mapping: no heap copies, pages
// fault in on first access, and physical pages can be released back
// to the OS while the mapping stays valid.
//
// Ownership (MR4 objective 6): the mapping is wrapped in a shared
// TensorBuffer. Every tensor returned by load_tensor_data holds a
// reference, so the file is unmapped only when the loader AND every
// tensor/view referencing it are gone. Mapped tensors are therefore
// lifetime-safe even if the loader is destroyed first.
class MMapLoader {

public:

    std::vector<std::string> list_tensors();

    bool load_file(const std::string& path);

    // Returns a copy of the cached tensor directory.
    std::unordered_map<std::string, Tensor> parse_header();

    // Metadata only (no data pointers).
    Tensor load_tensor(
        const std::string& tensor_name
    );

    // Lazy view into the mapping. NO heap copy. Carries shared
    // ownership of the mapping via TensorBuffer.
    Tensor load_tensor_data(
        const std::string& tensor_name
    );

    // Return the physical pages of one tensor (or the whole file)
    // to the OS. The mapping stays valid; next access refaults.
    void release_tensor_pages(
        const std::string& tensor_name
    );

    void release_all_pages();

    size_t mapped_bytes() const;

    const uint8_t* mapped_base() const;

private:

    std::shared_ptr<MappedFile> mapped_;

    // One shared control block for the whole mapping; holds the
    // MappedFile alive (keepalive) for as long as any tensor lives.
    std::shared_ptr<TensorBuffer> mapping_buffer_;

    uint64_t header_size = 0;

    std::string header_json;

    // Parsed exactly once per load_file; previously the JSON header
    // was reparsed on EVERY load_tensor_data call.
    std::unordered_map<std::string, Tensor> directory;

    bool directory_parsed = false;

    void ensure_directory();
};

}
